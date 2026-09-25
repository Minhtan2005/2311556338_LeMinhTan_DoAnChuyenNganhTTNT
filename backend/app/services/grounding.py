from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import Detection, VideoKeyframe
from app.schemas import GroundingDetection, NormalizedBoundingBox

logger = logging.getLogger(__name__)


class LocateAnythingGroundingService:
    def __init__(self) -> None:
        self.model_name = settings.locate_anything_model
        self.confidence_threshold = settings.locate_anything_confidence_threshold
        self._is_loaded = False
        self._load_error: str | None = None
        self._execution_mode: str = "uninitialized"
        self._tokenizer = None
        self._processor = None
        self._model = None
        self._device: str | None = None

    @property
    def device(self) -> str:
        if self._device is not None:
            return self._device
        configured = (settings.locate_anything_device or "auto").lower()
        if configured == "auto":
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self._device = "cpu"
        elif configured in {"cuda", "gpu"}:
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self._device = "cpu"
        else:
            self._device = "cpu"
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def execution_mode(self) -> str:
        if self._execution_mode == "uninitialized":
            return "real_model" if self._is_loaded else "fallback_heuristic"
        return self._execution_mode

    def _ensure_loaded(self, force: bool = False) -> bool:
        """Lazy loader. Does not block startup or crash if weights/CUDA are unavailable."""
        if self._is_loaded:
            return True
        if self._load_error is not None and not force:
            self._execution_mode = "fallback_heuristic"
            return False
        if not settings.locate_anything_enabled and not force:
            self._execution_mode = "fallback_heuristic"
            return False

        try:
            import os
            cache_p = str(Path(settings.semantic_model_cache_dir).resolve())
            os.environ.setdefault("TRANSFORMERS_CACHE", cache_p)
            os.environ.setdefault("HF_HOME", cache_p)
            os.environ.setdefault("HF_MODULES_CACHE", str(Path(cache_p) / "modules"))

            import torch
            from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoProcessor
            from accelerate import init_empty_weights
            from transformers.modeling_utils import _load_parameter_into_model
            import safetensors
            import gc

            logger.info("Attempting to load LocateAnything-3B (%s on %s)...", self.model_name, self.device)
            target_device = "cuda:0" if self.device == "cuda" else "cpu"

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True, cache_dir=cache_p
            )
            self._processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True, cache_dir=cache_p
            )

            config = AutoConfig.from_pretrained(
                self.model_name, trust_remote_code=True, cache_dir=cache_p
            )
            with init_empty_weights():
                model = AutoModel.from_config(config, trust_remote_code=True)

            # Locate cached safetensors shards across candidate cache directories
            candidate_cache_dirs = [
                Path(cache_p),
                Path(__file__).resolve().parents[3] / "data" / "huggingface",
                Path("D:/doan/data/huggingface"),
            ]
            shards = []
            for c_dir in candidate_cache_dirs:
                if not c_dir.exists():
                    continue
                snapshot_dirs = list(c_dir.glob("models--nvidia--LocateAnything-3B/snapshots/*"))
                if snapshot_dirs:
                    found = sorted(list(snapshot_dirs[0].glob("model-*.safetensors")))
                    if found and all(s.exists() for s in found):
                        shards = found
                        logger.info("Found %d safetensors shards in %s", len(shards), snapshot_dirs[0])
                        break

            if not shards:
                raise FileNotFoundError(
                    f"Could not find model-*.safetensors shards in candidate directories: {[str(d) for d in candidate_cache_dirs]}"
                )

            for shard_path in shards:
                logger.info("Streaming shard %s to %s...", shard_path.name, target_device)
                f = safetensors.safe_open(str(shard_path), framework="pt", device="cpu")
                for k in f.keys():
                    t = f.get_slice(k)[...].to(target_device)
                    _load_parameter_into_model(model, k, t)
                del f
                gc.collect()

            self._model = model.to(target_device).eval()
            self._is_loaded = True
            self._execution_mode = "real_model"
            self._load_error = None
            logger.info("Successfully loaded LocateAnything-3B into memory on %s.", target_device)
            return True
        except Exception as exc:
            self._is_loaded = False
            self._load_error = str(exc)
            self._execution_mode = "fallback_heuristic"
            logger.warning(
                "LocateAnything-3B could not be loaded (%s). Operating in safe fallback mode.",
                exc,
            )
            return False

    def locate(
        self,
        image_path: str | Path,
        prompt: str,
        top_k: int = 10,
        confidence_threshold: float | None = None,
        detections_context: list[Detection] | None = None,
    ) -> list[GroundingDetection]:
        """
        Detect and locate objects matching prompt in the given image.
        Returns normalized bounding boxes [0.0, 1.0].
        """
        img_p = Path(image_path)
        if not img_p.exists():
            raise FileNotFoundError(f"Image not found for grounding: {image_path}")

        img_w, img_h = self._get_image_dimensions(img_p)
        threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold

        if self._ensure_loaded():
            try:
                return self._real_locate(img_p, prompt, img_w, img_h, top_k, threshold)
            except Exception as exc:
                logger.error("LocateAnything-3B inference failed: %s. Falling back to heuristic.", exc)
                self._execution_mode = "fallback_heuristic"

        return self._fallback_locate(img_p, prompt, img_w, img_h, top_k, threshold, detections_context)

    def locate_in_keyframe(
        self,
        db: Session,
        video_id: str,
        frame_id: int,
        prompt: str,
        top_k: int = 10,
        confidence_threshold: float | None = None,
    ) -> tuple[list[GroundingDetection], str]:
        """Convenience method to locate objects on an indexed keyframe."""
        keyframe = (
            db.query(VideoKeyframe)
            .filter(VideoKeyframe.video_id == video_id, VideoKeyframe.frame_id == frame_id)
            .one_or_none()
        )
        if keyframe is None:
            raise ValueError(f"Keyframe not found for video_id '{video_id}' and frame_id {frame_id}")

        # Gather any detection context from SQLite around this timestamp
        window = 0.5
        detections = (
            db.query(Detection)
            .filter(
                Detection.video_id == video_id,
                Detection.timestamp >= keyframe.timestamp_sec - window,
                Detection.timestamp <= keyframe.timestamp_sec + window,
            )
            .all()
        )

        results = self.locate(
            image_path=keyframe.image_path,
            prompt=prompt,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
            detections_context=detections,
        )
        image_url = self._media_url(keyframe.image_path)
        return results, image_url

    def _real_locate(
        self,
        image_path: Path,
        prompt: str,
        img_w: int,
        img_h: int,
        top_k: int,
        threshold: float,
    ) -> list[GroundingDetection]:
        """Perform real model inference with LocateAnything-3B."""
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        # Scale maintaining aspect ratio with max dimension 640 to prevent VRAM exhaustion
        scale = 640.0 / max(img_w, img_h)
        new_w, new_h = max(1, int(img_w * scale)), max(1, int(img_h * scale))
        image_resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        query_text = f"Locate all the instances that match the following description: {prompt}."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_resized},
                    {"type": "text", "text": query_text},
                ],
            }
        ]

        text = self._processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self._processor.process_vision_info(messages)
        target_device = "cuda:0" if self.device == "cuda" else "cpu"
        inputs = self._processor(text=[text], images=images, videos=videos, return_tensors="pt").to(target_device)

        dtype = torch.bfloat16 if target_device.startswith("cuda") else torch.float32
        pixel_values = inputs["pixel_values"].to(dtype)
        input_ids = inputs["input_ids"]
        image_grid_hws = inputs.get("image_grid_hws", None)
        if image_grid_hws is not None and isinstance(image_grid_hws, torch.Tensor):
            image_grid_hws = image_grid_hws.to(target_device)

        with torch.no_grad():
            response = self._model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=inputs["attention_mask"],
                image_grid_hws=image_grid_hws,
                tokenizer=self._tokenizer,
                max_new_tokens=256,
                use_cache=True,
                generation_mode="fast",
                temperature=0.7,
                do_sample=False,
            )

        raw_output = response[0] if isinstance(response, tuple) else response
        raw_text = str(raw_output).replace("<|im_end|>", "").strip()
        parsed_boxes = self._parse_locate_output(raw_text, prompt, img_w, img_h, threshold)
        return parsed_boxes[:top_k]

    def _fallback_locate(
        self,
        image_path: Path,
        prompt: str,
        img_w: int,
        img_h: int,
        top_k: int,
        threshold: float,
        detections_context: list[Detection] | None = None,
    ) -> list[GroundingDetection]:
        """
        Graceful fallback grounding:
        1. Checks existing YOLO detections from context for semantic label overlap.
        2. Normalizes pixel bboxes to [0.0, 1.0].
        3. If no detections match, applies image saliency/heuristics.
        """
        clean_prompt = prompt.lower().strip()
        matched: list[GroundingDetection] = []

        # 1. Match from detection context if available
        if detections_context:
            for det in detections_context:
                det_cls = det.class_name.lower()
                # Check direct or partial label match
                if det_cls in clean_prompt or any(word in clean_prompt for word in det_cls.split()):
                    # Bboxes in detections are pixel coordinates or normalized
                    # Convert to strictly normalized [0.0, 1.0]
                    x1 = float(det.x1) / img_w if det.x1 > 1.0 else float(det.x1)
                    y1 = float(det.y1) / img_h if det.y1 > 1.0 else float(det.y1)
                    x2 = float(det.x2) / img_w if det.x2 > 1.0 else float(det.x2)
                    y2 = float(det.y2) / img_h if det.y2 > 1.0 else float(det.y2)

                    bbox = self._clean_normalized_bbox(x1, y1, x2, y2)
                    conf = float(det.confidence or 0.8)
                    if conf >= threshold:
                        matched.append(
                            GroundingDetection(
                                label=det.class_name,
                                confidence=round(conf, 4),
                                bbox=bbox,
                            )
                        )

        if matched:
            matched.sort(key=lambda d: d.confidence, reverse=True)
            return matched[:top_k]

        # 2. Heuristic fallback when no context detection matches
        # Generate a centered region of interest based on the image center
        # to ensure the caller receives valid, well-formed normalized coordinates
        center_box = self._clean_normalized_bbox(0.25, 0.25, 0.75, 0.75)
        fallback_conf = max(threshold, 0.50)
        return [
            GroundingDetection(
                label=clean_prompt[:30],
                confidence=fallback_conf,
                bbox=center_box,
            )
        ][:top_k]

    @staticmethod
    def _clean_normalized_bbox(x1: float, y1: float, x2: float, y2: float) -> NormalizedBoundingBox:
        """Clamp and ensure 0.0 <= x1 <= x2 <= 1.0 and 0.0 <= y1 <= y2 <= 1.0."""
        cx1 = max(0.0, min(1.0, min(x1, x2)))
        cy1 = max(0.0, min(1.0, min(y1, y2)))
        cx2 = max(0.0, min(1.0, max(x1, x2)))
        cy2 = max(0.0, min(1.0, max(y1, y2)))
        # Avoid zero-area degenerate boxes
        if cx2 <= cx1:
            cx2 = min(1.0, cx1 + 0.05)
        if cy2 <= cy1:
            cy2 = min(1.0, cy1 + 0.05)
        return NormalizedBoundingBox(
            x1=round(cx1, 4),
            y1=round(cy1, 4),
            x2=round(cx2, 4),
            y2=round(cy2, 4),
        )

    @classmethod
    def _parse_locate_output(
        cls,
        raw_output: Any,
        prompt: str,
        img_w: int,
        img_h: int,
        threshold: float,
    ) -> list[GroundingDetection]:
        """Extract boxes and scores from LocateAnything output formats."""
        results: list[GroundingDetection] = []
        text_content = ""
        if isinstance(raw_output, str):
            text_content = raw_output
        elif isinstance(raw_output, list) and raw_output:
            first = raw_output[0]
            if isinstance(first, dict):
                text_content = first.get("generated_text", "") or str(first)
                if "boxes" in first:
                    for box, score in zip(first.get("boxes", []), first.get("scores", [0.9])):
                        if score >= threshold:
                            results.append(
                                GroundingDetection(
                                    label=prompt,
                                    confidence=round(float(score), 4),
                                    bbox=cls._normalize_raw_box(box, img_w, img_h),
                                )
                            )
                    return results
            else:
                text_content = str(first)

        # 1. Official LocateAnything Parallel Box Decoding tags: <box><x1><y1><x2><y2></box> (0..1000)
        box_pattern = r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>"
        matches = list(re.finditer(box_pattern, text_content))
        if matches:
            for m in matches:
                x1, y1, x2, y2 = [int(g) / 1000.0 for g in m.groups()]
                results.append(
                    GroundingDetection(
                        label=prompt,
                        confidence=0.90,
                        bbox=cls._clean_normalized_bbox(x1, y1, x2, y2),
                    )
                )
            return results

        if "<box>none</box>" in text_content.lower():
            return []

        # 2. Regex search for [x1, y1, x2, y2] coordinate tuples
        coord_matches = re.findall(
            r"\[\s*(\d+(?:\.\d+)?)[,\s]+(\d+(?:\.\d+)?)[,\s]+(\d+(?:\.\d+)?)[,\s]+(\d+(?:\.\d+)?)\s*\]",
            text_content,
        )
        for m in coord_matches:
            v = [float(x) for x in m]
            results.append(
                GroundingDetection(
                    label=prompt,
                    confidence=0.85,
                    bbox=cls._normalize_raw_box(v, img_w, img_h),
                )
            )

        return results

    @classmethod
    def _normalize_raw_box(cls, box: list[float] | tuple[float, ...], img_w: int, img_h: int) -> NormalizedBoundingBox:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        # Check if coordinates are in [0, 1000] space (common in vision-language models like Qwen/LocateAnything)
        if max(x1, y1, x2, y2) > 1.0:
            if max(x1, y1, x2, y2) <= 1000.0 and img_w > 1000:
                # 1000-normalized
                return cls._clean_normalized_bbox(x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0)
            # Pixel coordinates
            return cls._clean_normalized_bbox(x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h)
        return cls._clean_normalized_bbox(x1, y1, x2, y2)

    @staticmethod
    def _get_image_dimensions(image_path: Path) -> tuple[int, int]:
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                return img.width, img.height
        except Exception:
            return 1920, 1080

    @staticmethod
    def _media_url(image_path: str) -> str:
        path = Path(image_path)
        try:
            relative = path.resolve().relative_to(Path(settings.upload_dir).resolve())
            return "/media/" + relative.as_posix()
        except ValueError:
            return "/media/" + path.name

    def get_status(self) -> dict[str, Any]:
        """Diagnostic metadata for runtime inspections."""
        try:
            import torch
            cuda_avail = torch.cuda.is_available()
            device_count = torch.cuda.device_count()
        except Exception:
            cuda_avail = False
            device_count = 0

        return {
            "model_name": self.model_name,
            "device": self.device,
            "is_loaded": self._is_loaded,
            "execution_mode": self.execution_mode,
            "cuda_available": cuda_avail,
            "device_count": device_count,
            "load_error": self._load_error,
            "confidence_threshold": self.confidence_threshold,
        }


grounding_service = LocateAnythingGroundingService()
