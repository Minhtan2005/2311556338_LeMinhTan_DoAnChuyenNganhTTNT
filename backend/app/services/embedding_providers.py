from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from app.core.config import settings
from app.services.nlp import COLOR_ALIASES, OBJECT_ALIASES, normalize_text


logger = logging.getLogger(__name__)


class SemanticEmbeddingProvider(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_version(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def encode_image(self, image: str | Path) -> np.ndarray:
        raise NotImplementedError

    def encode_text_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.encode_text(text) for text in texts]

    def encode_image_batch(self, images: list[str | Path]) -> list[np.ndarray]:
        return [self.encode_image(image) for image in images]


class LocalSemanticProvider(SemanticEmbeddingProvider):
    """Legacy Phase 2 baseline retained for offline fallback and regression tests."""

    _features = [
        "person",
        "car",
        "motorcycle",
        "bicycle",
        "bus",
        "truck",
        "traffic light",
        "backpack",
        "red",
        "blue",
        "green",
        "white",
        "black",
        "yellow",
        "gray",
        "street",
        "road",
        "traffic",
        "vehicle",
    ]

    @property
    def model_name(self) -> str:
        return "local-semantic-v1"

    @property
    def model_version(self) -> str:
        return "legacy"

    @property
    def embedding_dimension(self) -> int:
        return len(self._features)

    def encode_text(self, text: str) -> np.ndarray:
        normalized = normalize_text(text)
        vector = np.zeros(self.embedding_dimension, dtype=np.float32)
        for feature, aliases in OBJECT_ALIASES.items():
            if feature in self._features and any(alias in normalized for alias in aliases + [feature]):
                vector[self._features.index(feature)] += 1.0
        for feature, aliases in COLOR_ALIASES.items():
            if feature in self._features and any(alias in normalized for alias in aliases + [feature]):
                vector[self._features.index(feature)] += 0.8
        if any(word in normalized for word in ["đường", "street", "road"]):
            vector[self._features.index("street")] += 0.7
        if "xe" in normalized or "vehicle" in normalized:
            vector[self._features.index("vehicle")] += 0.7
        return normalize_vector(vector)

    def encode_image(self, image: str | Path) -> np.ndarray:
        vector = np.zeros(self.embedding_dimension, dtype=np.float32)
        color = dominant_color_name(image)
        if color in self._features:
            vector[self._features.index(color)] += 0.6
        return normalize_vector(vector)


class TransformersCLIPProvider(SemanticEmbeddingProvider):
    def __init__(self) -> None:
        self.model_id = settings.semantic_model_name_or_path
        self._model = None
        self._processor = None
        self._torch = None
        self._device = None
        self._embedding_dimension: int | None = None

    @property
    def model_name(self) -> str:
        return settings.semantic_model

    @property
    def model_version(self) -> str:
        return settings.semantic_model_version

    @property
    def embedding_dimension(self) -> int:
        self._ensure_loaded()
        assert self._embedding_dimension is not None
        return self._embedding_dimension

    def encode_text(self, text: str) -> np.ndarray:
        return self.encode_text_batch([text])[0]

    def encode_image(self, image: str | Path) -> np.ndarray:
        return self.encode_image_batch([image])[0]

    def encode_text_batch(self, texts: list[str]) -> list[np.ndarray]:
        self._ensure_loaded()
        assert self._torch is not None and self._model is not None and self._processor is not None
        translated = [prepare_clip_text(text) for text in texts]
        inputs = self._processor(text=translated, return_tensors="pt", padding=True, truncation=True).to(self._device)
        with self._torch.no_grad():
            features = self._model.get_text_features(**inputs)
            rows = feature_rows(features, "text_embeds", self._embedding_dimension, len(texts))
            if rows is None:
                outputs = self._model.text_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
                pooled = getattr(outputs, "pooler_output", None)
                if pooled is None:
                    pooled = outputs[1]
                rows = self._model.text_projection(pooled)
        return [normalize_vector(row.detach().cpu().numpy().astype(np.float32)) for row in rows]

    def encode_image_batch(self, images: list[str | Path]) -> list[np.ndarray]:
        self._ensure_loaded()
        assert self._torch is not None and self._model is not None and self._processor is not None
        from PIL import Image

        pil_images = [Image.open(path).convert("RGB") for path in images]
        inputs = self._processor(images=pil_images, return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            features = self._model.get_image_features(**inputs)
            rows = feature_rows(features, "image_embeds", self._embedding_dimension, len(images))
            if rows is None:
                outputs = self._model.vision_model(pixel_values=inputs["pixel_values"])
                pooled = getattr(outputs, "pooler_output", None)
                if pooled is None:
                    pooled = outputs[1]
                rows = self._model.visual_projection(pooled)
        for image in pil_images:
            image.close()
        return [normalize_vector(row.detach().cpu().numpy().astype(np.float32)) for row in rows]

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        cache_dir = settings.semantic_model_cache_dir
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(Path(cache_dir).resolve()))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(cache_dir).resolve()))

        import torch
        from transformers import CLIPModel, CLIPProcessor

        if settings.semantic_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = settings.semantic_device

        logger.info("Loading semantic embedding model %s on %s", self.model_id, device)
        self._processor = CLIPProcessor.from_pretrained(self.model_id, cache_dir=cache_dir)
        self._model = CLIPModel.from_pretrained(self.model_id, cache_dir=cache_dir)
        self._model.eval()
        self._model.to(device)
        self._torch = torch
        self._device = device
        self._embedding_dimension = int(getattr(self._model.config, "projection_dim", 512))


def get_embedding_provider() -> SemanticEmbeddingProvider:
    configured = settings.semantic_model.lower().strip()
    if configured == "local-semantic-v1":
        return LocalSemanticProvider()
    if configured in {"clip-vit-base-patch32", "openai/clip-vit-base-patch32"}:
        return TransformersCLIPProvider()
    raise RuntimeError(f"Unsupported SEMANTIC_MODEL: {settings.semantic_model}")


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return vector
    return (vector / norm).astype(np.float32)


def feature_rows(features, preferred_key: str, expected_dim: int | None, batch_size: int):
    candidates = []
    if hasattr(features, "detach"):
        candidates.append(features)
    if hasattr(features, preferred_key):
        candidates.append(getattr(features, preferred_key))
    if hasattr(features, "pooler_output"):
        candidates.append(getattr(features, "pooler_output"))
    if isinstance(features, dict):
        for key in (preferred_key, "pooler_output"):
            value = features.get(key)
            if value is not None:
                candidates.append(value)
        candidates.extend(value for value in features.values() if hasattr(value, "detach"))

    for tensor in candidates:
        if tensor is None or not hasattr(tensor, "detach"):
            continue
        if getattr(tensor, "ndim", 0) == 1:
            tensor = tensor.unsqueeze(0)
        if getattr(tensor, "ndim", 0) != 2:
            continue
        if tensor.shape[0] != batch_size:
            continue
        if expected_dim is not None and tensor.shape[-1] != expected_dim:
            continue
        return tensor
    return None


def prepare_clip_text(text: str) -> str:
    normalized = normalize_text(text)
    replacements = [
        ("một người đàn ông", "a man"),
        ("người đàn ông", "man"),
        ("một người phụ nữ", "a woman"),
        ("người phụ nữ", "woman"),
        ("một người", "a person"),
        ("người", "person"),
        ("áo đỏ", "red shirt"),
        ("áo trắng", "white shirt"),
        ("áo đen", "black shirt"),
        ("áo xanh", "blue shirt"),
        ("mặc", "wearing"),
        ("đứng cạnh", "standing next to"),
        ("bên cạnh", "next to"),
        ("cạnh", "next to"),
        ("gần", "near"),
        ("đang đi qua đường", "crossing the street"),
        ("đi qua đường", "crossing the street"),
        ("băng qua đường", "crossing the street"),
        ("đường", "street"),
        ("xe máy", "motorcycle"),
        ("mô tô", "motorcycle"),
        ("ô tô", "car"),
        ("xe hơi", "car"),
        ("xe tải", "truck"),
        ("xe buýt", "bus"),
        ("xe đạp", "bicycle"),
        ("ba lô", "backpack"),
        ("balo", "backpack"),
        ("túi", "bag"),
        ("cầm", "holding"),
        ("mang", "carrying"),
        ("đeo", "wearing"),
        ("màu trắng", "white"),
        ("màu đỏ", "red"),
        ("màu đen", "black"),
        ("màu xanh", "blue"),
        ("màu vàng", "yellow"),
        ("đậu", "parked"),
        ("vỉa hè", "sidewalk"),
        ("tìm cảnh", "scene"),
        ("cảnh", "scene"),
    ]
    translated = normalized
    for source, target in replacements:
        translated = translated.replace(source, target)
    if translated == normalized:
        return text
    return f"{translated}. Original Vietnamese query: {text}"


def dominant_color_name(image_path: str | Path) -> str | None:
    try:
        import cv2
    except ImportError:
        return None

    image = cv2.imread(str(image_path))
    if image is None:
        return None
    small = cv2.resize(image, (1, 1), interpolation=cv2.INTER_AREA)[0, 0]
    b, g, r = [int(value) for value in small]
    brightness = (r + g + b) / 3
    if brightness > 210:
        return "white"
    if brightness < 55:
        return "black"
    if abs(r - g) < 20 and abs(g - b) < 20:
        return "gray"
    if r > g * 1.25 and r > b * 1.25:
        return "red"
    if g > r * 1.15 and g > b * 1.15:
        return "green"
    if b > r * 1.15 and b > g * 1.15:
        return "blue"
    if r > 150 and g > 130 and b < 100:
        return "yellow"
    return "gray"
