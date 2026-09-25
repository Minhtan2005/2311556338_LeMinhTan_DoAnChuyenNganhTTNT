from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import Event, Video, VideoKeyframe


@dataclass(frozen=True)
class KeyframeExtractionResult:
    created: int
    total: int


class KeyframeExtractor:
    def extract_for_video(self, db: Session, video: Video) -> KeyframeExtractionResult:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for keyframe extraction.") from exc

        video_path = Path(video.stored_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for keyframe extraction: {video_path}")

        try:
            fps = float(video.fps or cap.get(cv2.CAP_PROP_FPS) or 25.0)
            total_frames = int(video.total_frames or cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            frame_ids = self._candidate_frame_ids(db, video.id, fps, total_frames)
            existing = {
                frame_id
                for (frame_id,) in db.query(VideoKeyframe.frame_id).filter(VideoKeyframe.video_id == video.id).all()
            }

            created = 0
            seen_hashes: list[int] = []
            output_dir = Path(settings.upload_dir) / "keyframes" / video.id
            output_dir.mkdir(parents=True, exist_ok=True)

            for frame_id in frame_ids:
                image_path = output_dir / f"frame_{frame_id:08d}.jpg"
                if frame_id in existing and image_path.exists():
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                if float(frame.std()) < settings.keyframe_min_stddev:
                    continue

                frame_hash = self._difference_hash(frame)
                if any(self._hamming_distance(frame_hash, item) <= settings.keyframe_hash_distance for item in seen_hashes):
                    continue
                seen_hashes.append(frame_hash)

                params = [int(cv2.IMWRITE_JPEG_QUALITY), int(settings.keyframe_image_quality)]
                if not cv2.imwrite(str(image_path), frame, params):
                    continue

                existing_row = (
                    db.query(VideoKeyframe)
                    .filter(VideoKeyframe.video_id == video.id, VideoKeyframe.frame_id == frame_id)
                    .one_or_none()
                )
                if existing_row is None:
                    db.add(
                        VideoKeyframe(
                            video_id=video.id,
                            frame_id=frame_id,
                            timestamp_sec=frame_id / fps if fps else 0.0,
                            image_path=str(image_path),
                            created_at=datetime.utcnow(),
                        )
                    )
                else:
                    existing_row.timestamp_sec = frame_id / fps if fps else existing_row.timestamp_sec
                    existing_row.image_path = str(image_path)
                created += 1

            db.commit()
            total = db.query(VideoKeyframe).filter(VideoKeyframe.video_id == video.id).count()
            return KeyframeExtractionResult(created=created, total=total)
        finally:
            cap.release()

    def _candidate_frame_ids(self, db: Session, video_id: str, fps: float, total_frames: int) -> list[int]:
        if total_frames <= 0:
            return []

        if settings.keyframe_target_fps > 0:
            interval_frames = max(1, int(round(max(fps, 1.0) / min(max(settings.keyframe_target_fps, 0.1), max(fps, 1.0)))))
        else:
            interval_frames = max(1, int(round(max(settings.keyframe_interval_seconds, 0.1) * max(fps, 1.0))))
        frame_ids = set(range(0, total_frames, interval_frames))

        event_rows = (
            db.query(Event.timestamp_start)
            .filter(Event.video_id == video_id)
            .order_by(Event.timestamp_start.asc())
            .limit(settings.max_keyframes_per_video)
            .all()
        )
        for (timestamp,) in event_rows:
            frame_ids.add(max(0, min(total_frames - 1, int(round(float(timestamp) * fps)))))

        frame_ids.add(max(0, total_frames - 1))
        return sorted(frame_ids)[: max(settings.max_keyframes_per_video, 1)]

    @staticmethod
    def _difference_hash(frame) -> int:
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        bits = resized[:, 1:] > resized[:, :-1]
        value = 0
        for bit in bits.flatten():
            value = (value << 1) | int(bool(bit))
        return value

    @staticmethod
    def _hamming_distance(left: int, right: int) -> int:
        return int((left ^ right).bit_count())


keyframe_extractor = KeyframeExtractor()
