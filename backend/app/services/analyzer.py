from datetime import datetime
import logging
from pathlib import Path
from math import hypot

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal
from app.models import Detection, Video
from app.services.events import EventDetector, VEHICLE_CLASSES
from app.services.speech import speech_to_text_service


COCO_TARGET_CLASS_NAMES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
}


logger = logging.getLogger(__name__)


class LocalCentroidTracker:
    def __init__(self, max_distance: float = 80.0) -> None:
        self.max_distance = max_distance
        self._next_id = 100000
        self._tracks: dict[int, tuple[str, tuple[float, float]]] = {}

    def assign(self, class_name: str, center: tuple[float, float]) -> int:
        best_id: int | None = None
        best_distance = self.max_distance
        for track_id, (tracked_class, tracked_center) in self._tracks.items():
            if tracked_class != class_name:
                continue
            distance = hypot(center[0] - tracked_center[0], center[1] - tracked_center[1])
            if distance < best_distance:
                best_id = track_id
                best_distance = distance

        if best_id is None:
            best_id = self._next_id
            self._next_id += 1

        self._tracks[best_id] = (class_name, center)
        return best_id


class VideoAnalyzer:
    def __init__(self) -> None:
        self._model = None

    def analyze(self, db: Session, video: Video) -> None:
        logger.info("Starting analysis for video_id=%s path=%s", video.id, video.stored_path)
        video.status = "processing"
        video.error_message = None
        video.progress_percent = 0
        video.processed_at = None
        db.commit()

        try:
            self._analyze_impl(db, video)
            refreshed = db.get(Video, video.id)
            if refreshed is None:
                return
            refreshed.status = "completed"
            refreshed.progress_percent = 100
            refreshed.error_message = None
            refreshed.processed_at = datetime.utcnow()
            db.commit()
        except Exception as exc:
            db.rollback()
            refreshed = db.get(Video, video.id)
            if refreshed is not None:
                refreshed.status = "failed"
                refreshed.error_message = str(exc)
                refreshed.processed_at = datetime.utcnow()
                db.commit()
            raise

    def _analyze_impl(self, db: Session, video: Video) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is not installed. Install backend/requirements-ai.txt before analyzing videos."
            ) from exc

        path = Path(video.stored_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

            if width <= 0 or height <= 0:
                raise RuntimeError("Video metadata is invalid. Width or height is zero.")

            video.fps = float(fps)
            video.total_frames = total_frames
            video.duration_seconds = total_frames / fps if fps else None
            db.commit()
            logger.info(
                "Video metadata video_id=%s fps=%.2f frames=%s size=%sx%s",
                video.id,
                fps,
                total_frames,
                width,
                height,
            )

            detector = EventDetector(frame_width=max(width, 1), frame_height=max(height, 1))
            fallback_tracker = LocalCentroidTracker()
            model = self._load_model()
            target_ids = self._target_class_ids(model)
            if not target_ids:
                raise RuntimeError("YOLO model does not expose supported COCO target classes.")

            frame_index = 0
            processed_frames = 0
            batch_objects: list[Detection] = []
            batch_events = []

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_index % max(settings.frame_stride, 1) != 0:
                    frame_index += 1
                    continue

                timestamp = frame_index / fps if fps else 0.0
                result = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=target_ids,
                    conf=settings.confidence_threshold,
                    verbose=False,
                )[0]

                detections = self._result_to_detections(video.id, frame_index, timestamp, result, fallback_tracker)
                active_vehicle_count = sum(1 for item in detections if item.class_name in VEHICLE_CLASSES)

                for detection in detections:
                    batch_objects.append(detection)
                    batch_events.extend(
                        detector.observe(
                            video_id=video.id,
                            track_id=detection.track_id,
                            class_name=detection.class_name,
                            timestamp=timestamp,
                            confidence=detection.confidence,
                            bbox=(detection.x1, detection.y1, detection.x2, detection.y2),
                            active_vehicle_count=active_vehicle_count,
                        )
                    )

                if len(batch_objects) >= 250:
                    db.add_all(batch_objects)
                    db.add_all(batch_events)
                    video.progress_percent = self._progress(frame_index, total_frames)
                    db.commit()
                    batch_objects.clear()
                    batch_events.clear()

                processed_frames += 1
                if processed_frames % 100 == 0:
                    logger.info(
                        "Analysis progress video_id=%s processed_frames=%s frame_index=%s detections=%s",
                        video.id,
                        processed_frames,
                        frame_index,
                        len(batch_objects),
                    )
                frame_index += 1

            if processed_frames == 0:
                raise RuntimeError("No frames were processed. Check video codec or FRAME_STRIDE setting.")

            batch_events.extend(detector.finalize(video.id))
            if batch_objects:
                db.add_all(batch_objects)
            if batch_events:
                db.add_all(batch_events)
            video.progress_percent = 100
            db.commit()

            if settings.enable_whisper:
                speech_to_text_service.transcribe_video(db, video)

            try:
                from app.services.semantic_index import semantic_index_service

                semantic_index_service.index_video(db, video)
            except Exception:
                logger.exception("Semantic KIS indexing failed for video_id=%s", video.id)
            logger.info("Completed analysis for video_id=%s processed_frames=%s", video.id, processed_frames)
        finally:
            cap.release()

    def _load_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "Ultralytics YOLO is not installed. Install backend/requirements-ai.txt before analyzing videos."
                ) from exc

            try:
                self._model = YOLO(settings.yolo_model)
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot load YOLO model '{settings.yolo_model}'. "
                    "Check internet access for first download or place the model file locally and update YOLO_MODEL."
                ) from exc
        return self._model

    @staticmethod
    def _progress(frame_index: int, total_frames: int) -> int:
        if total_frames <= 0:
            return 0
        return max(0, min(99, int((frame_index / total_frames) * 100)))

    @staticmethod
    def _target_class_ids(model) -> list[int]:
        names = getattr(model, "names", {})
        return [int(class_id) for class_id, name in names.items() if name in COCO_TARGET_CLASS_NAMES]

    @staticmethod
    def _result_to_detections(
        video_id: str,
        frame_index: int,
        timestamp: float,
        result,
        fallback_tracker: LocalCentroidTracker,
    ) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
        classes = boxes.cls.cpu().numpy().astype(int).tolist() if boxes.cls is not None else []
        ids = boxes.id.cpu().numpy().astype(int).tolist() if boxes.id is not None else [None] * len(xyxy)
        names = getattr(result, "names", {})

        detections: list[Detection] = []
        for index, coords in enumerate(xyxy):
            if len(classes) <= index:
                continue
            class_id = int(classes[index])
            class_name = names.get(class_id, str(class_id))
            if class_name not in COCO_TARGET_CLASS_NAMES:
                continue
            x1, y1, x2, y2 = [float(value) for value in coords]
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            track_id = fallback_tracker.assign(class_name, center) if ids[index] is None else int(ids[index])
            confidence = float(confs[index]) if len(confs) > index else 1.0
            detections.append(
                Detection(
                    video_id=video_id,
                    frame_index=frame_index,
                    timestamp=float(timestamp),
                    track_id=track_id,
                    class_name=class_name,
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
        return detections


video_analyzer = VideoAnalyzer()


def analyze_video_task(video_id: str) -> None:
    db = SessionLocal()
    try:
        video = db.get(Video, video_id)
        if video is None:
            return
        video_analyzer.analyze(db, video)
    finally:
        db.close()
