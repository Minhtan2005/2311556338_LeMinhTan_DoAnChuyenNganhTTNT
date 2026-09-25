from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.services.open_vocab_query import build_cache_key, sanitize_prompt


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenVocabDetection:
    class_name: str
    confidence: float
    bbox: list[float]
    frame_id: int
    timestamp: float
    detector: str = "yolo_world"


@dataclass(frozen=True)
class OpenVocabSegment:
    class_name: str
    start_time: float
    end_time: float
    timestamp: float
    frame_id: int
    confidence: float
    bbox: list[float]
    detector: str = "yolo_world"


@dataclass(frozen=True)
class OpenVocabSearch:
    query_object: str
    detector: str
    model: str
    confidence: float
    sample_fps: float
    cache_hit: bool
    detections: list[OpenVocabDetection]
    segments: list[OpenVocabSegment]
    device: str = "unknown"


class YOLOWorldDetector:
    def __init__(self) -> None:
        self._model: Any | None = None
        self._loaded_model_name: str | None = None
        self._active_classes: tuple[str, ...] = ()
        self._device: str | None = None
        self._lock = threading.RLock()

    def load_model(self):
        if self._model is not None and self._loaded_model_name == settings.yolo_world_model:
            return self._model

        try:
            from ultralytics import YOLOWorld
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics YOLOWorld is not available. Install a compatible ultralytics version."
            ) from exc

        try:
            self._model = YOLOWorld(settings.yolo_world_model)
            self._loaded_model_name = settings.yolo_world_model
            self._active_classes = ()
            self._sync_model_device(self._model)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot load YOLO-World model '{settings.yolo_world_model}'."
            ) from exc
        return self._model

    def set_classes(self, classes: list[str]) -> None:
        model = self.load_model()
        clean_classes = tuple(sanitize_prompt(item) for item in classes if sanitize_prompt(item))
        if not clean_classes:
            raise ValueError("YOLO-World classes cannot be empty.")
        if clean_classes == self._active_classes:
            return
        if not hasattr(model, "set_classes"):
            raise RuntimeError("Loaded YOLO-World model does not expose set_classes().")
        old_userprofile = os.environ.get("USERPROFILE")
        old_home = os.environ.get("HOME")
        cache_home = str(Path(settings.yolo_world_clip_cache_home).resolve())
        try:
            os.environ["USERPROFILE"] = cache_home
            os.environ["HOME"] = cache_home
            self._sync_model_device(model)
            model.set_classes(list(clean_classes))
            self._sync_model_device(model)
        finally:
            if old_userprofile is None:
                os.environ.pop("USERPROFILE", None)
            else:
                os.environ["USERPROFILE"] = old_userprofile
            if old_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old_home
        self._active_classes = clean_classes

    def detect_frame(
        self,
        frame,
        classes: list[str],
        *,
        frame_id: int = 0,
        timestamp: float = 0.0,
        conf: float | None = None,
    ) -> list[OpenVocabDetection]:
        with self._lock:
            self.set_classes(classes)
            model = self.load_model()
            threshold = float(settings.yolo_world_conf if conf is None else conf)
            self._sync_model_device(model)
            result = model.predict(frame, conf=threshold, verbose=False, device=self.device)[0]
        return self._result_to_detections(result, classes, frame_id=frame_id, timestamp=timestamp)

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = self._resolve_device()
        return self._device

    @staticmethod
    def _resolve_device() -> str:
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _sync_model_device(self, model: Any) -> None:
        device = self.device
        if hasattr(model, "to"):
            model.to(device)

        world_model = getattr(model, "model", None)
        clip_model = getattr(world_model, "clip_model", None)
        if clip_model is None:
            return

        if hasattr(clip_model, "device"):
            clip_model.device = device
        nested_model = getattr(clip_model, "model", None)
        if nested_model is not None and hasattr(nested_model, "to"):
            nested_model.to(device)

    def search_video(
        self,
        *,
        video_id: str,
        video_path: str,
        query_object: str,
        conf: float | None = None,
        sample_fps: float | None = None,
    ) -> OpenVocabSearch:
        if not settings.yolo_world_enabled:
            raise RuntimeError("YOLO-World is disabled by configuration.")

        prompt = sanitize_prompt(query_object)
        if not prompt:
            raise ValueError("Invalid open-vocabulary object prompt.")

        threshold = float(settings.yolo_world_conf if conf is None else conf)
        sampling_fps = max(float(settings.yolo_world_sample_fps if sample_fps is None else sample_fps), 0.1)
        cache_key = build_cache_key(
            video_id=video_id,
            object_name=prompt,
            model=settings.yolo_world_model,
            confidence=threshold,
            sample_fps=sampling_fps,
        )

        cached = self._read_cache(cache_key)
        if cached is not None:
            return OpenVocabSearch(
                query_object=prompt,
                detector="yolo_world",
                model=settings.yolo_world_model,
                device=self.device,
                confidence=threshold,
                sample_fps=sampling_fps,
                cache_hit=True,
                detections=[OpenVocabDetection(**item) for item in cached.get("detections", [])],
                segments=[OpenVocabSegment(**item) for item in cached.get("segments", [])],
            )

        detections = self._scan_video(
            video_path=video_path,
            query_object=prompt,
            conf=threshold,
            sample_fps=sampling_fps,
        )
        segments = deduplicate_temporal_detections(detections, gap_seconds=settings.yolo_world_merge_gap_seconds)
        result = OpenVocabSearch(
            query_object=prompt,
            detector="yolo_world",
            model=settings.yolo_world_model,
            device=self.device,
            confidence=threshold,
            sample_fps=sampling_fps,
            cache_hit=False,
            detections=detections,
            segments=segments,
        )
        self._write_cache(cache_key, result)
        return result

    def _scan_video(
        self,
        *,
        video_path: str,
        query_object: str,
        conf: float,
        sample_fps: float,
    ) -> list[OpenVocabDetection]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is not installed. Cannot scan video for YOLO-World search.") from exc

        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        detections: list[OpenVocabDetection] = []
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            sample_stride = max(int(round(fps / sample_fps)), 1)
            frame_id = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_id % sample_stride != 0:
                    frame_id += 1
                    continue

                timestamp = frame_id / fps if fps > 0 else 0.0
                detections.extend(
                    self.detect_frame(
                        frame,
                        [query_object],
                        frame_id=frame_id,
                        timestamp=timestamp,
                        conf=conf,
                    )
                )
                frame_id += 1
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                self._try_clear_cuda_cache()
            raise
        finally:
            cap.release()

        return detections

    @staticmethod
    def _result_to_detections(result, classes: list[str], *, frame_id: int, timestamp: float) -> list[OpenVocabDetection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
        cls_values = boxes.cls.cpu().numpy().astype(int).tolist() if boxes.cls is not None else []
        names = getattr(result, "names", {}) or {}

        detections: list[OpenVocabDetection] = []
        for index, coords in enumerate(xyxy):
            class_id = int(cls_values[index]) if len(cls_values) > index else 0
            class_name = str(names.get(class_id) or classes[min(class_id, len(classes) - 1)])
            x1, y1, x2, y2 = [float(value) for value in coords]
            if x2 <= x1 or y2 <= y1:
                continue
            confidence = float(confs[index]) if len(confs) > index else 1.0
            detections.append(
                OpenVocabDetection(
                    class_name=class_name,
                    confidence=confidence,
                    bbox=[x1, y1, x2, y2],
                    frame_id=int(frame_id),
                    timestamp=float(timestamp),
                )
            )
        return detections

    def _read_cache(self, cache_key: str) -> dict[str, Any] | None:
        if not settings.yolo_world_cache:
            return None
        path = self._cache_path(cache_key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Ignoring corrupt YOLO-World cache file: %s", path)
            return None

    def _write_cache(self, cache_key: str, result: OpenVocabSearch) -> None:
        if not settings.yolo_world_cache:
            return
        path = self._cache_path(cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "query_object": result.query_object,
            "detector": result.detector,
            "model": result.model,
            "device": result.device,
            "confidence": result.confidence,
            "sample_fps": result.sample_fps,
            "detections": [asdict(item) for item in result.detections],
            "segments": [asdict(item) for item in result.segments],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _cache_path(cache_key: str) -> Path:
        digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
        return Path(settings.yolo_world_cache_dir) / f"{digest}.json"

    @staticmethod
    def _try_clear_cuda_cache() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return


def deduplicate_temporal_detections(
    detections: list[OpenVocabDetection],
    *,
    gap_seconds: float = 1.5,
) -> list[OpenVocabSegment]:
    if not detections:
        return []

    ordered = sorted(detections, key=lambda item: (item.timestamp, -item.confidence))
    groups: list[list[OpenVocabDetection]] = []
    current: list[OpenVocabDetection] = []
    last_time: float | None = None

    for detection in ordered:
        if last_time is None or detection.timestamp - last_time <= gap_seconds:
            current.append(detection)
        else:
            groups.append(current)
            current = [detection]
        last_time = detection.timestamp
    if current:
        groups.append(current)

    segments: list[OpenVocabSegment] = []
    for group in groups:
        best = max(group, key=lambda item: item.confidence)
        segments.append(
            OpenVocabSegment(
                class_name=best.class_name,
                start_time=float(group[0].timestamp),
                end_time=float(group[-1].timestamp),
                timestamp=float(best.timestamp),
                frame_id=int(best.frame_id),
                confidence=float(best.confidence),
                bbox=list(best.bbox),
            )
        )
    return segments


yolo_world_detector = YOLOWorldDetector()
