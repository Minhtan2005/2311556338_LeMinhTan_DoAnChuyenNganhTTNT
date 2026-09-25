from __future__ import annotations

from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.services.grounding import LocateAnythingGroundingService, grounding_service
from app.schemas import NormalizedBoundingBox


@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    img_path = tmp_path / "test_frame.jpg"
    img = Image.new("RGB", (640, 480), color=(73, 109, 137))
    img.save(img_path)
    return img_path


def test_lazy_loading_does_not_load_on_init() -> None:
    service = LocateAnythingGroundingService()
    assert service.is_loaded is False
    assert service._model is None
    assert service.model_name == "nvidia/LocateAnything-3B"


def test_device_selection_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    service = LocateAnythingGroundingService()
    # Test explicit CPU
    monkeypatch.setattr("app.core.config.settings.locate_anything_device", "cpu")
    service._device = None
    assert service.device == "cpu"

    # Test auto fallback
    monkeypatch.setattr("app.core.config.settings.locate_anything_device", "auto")
    service._device = None
    assert service.device in {"cuda", "cpu"}


def test_clean_normalized_bbox() -> None:
    # Standard valid box
    box = LocateAnythingGroundingService._clean_normalized_bbox(0.1, 0.2, 0.5, 0.6)
    assert 0.0 <= box.x1 <= box.x2 <= 1.0
    assert 0.0 <= box.y1 <= box.y2 <= 1.0

    # Inverted box (x1 > x2)
    box_inv = LocateAnythingGroundingService._clean_normalized_bbox(0.8, 0.9, 0.2, 0.3)
    assert box_inv.x1 <= box_inv.x2
    assert box_inv.y1 <= box_inv.y2
    assert box_inv.x1 == 0.2
    assert box_inv.x2 == 0.8

    # Clamping out-of-bounds box
    box_oob = LocateAnythingGroundingService._clean_normalized_bbox(-0.5, -0.2, 1.5, 1.8)
    assert box_oob.x1 == 0.0
    assert box_oob.y1 == 0.0
    assert box_oob.x2 == 1.0
    assert box_oob.y2 == 1.0


def test_fallback_grounding_on_image(sample_image: Path) -> None:
    service = LocateAnythingGroundingService()
    # Ensure offline / fallback
    service._is_loaded = False
    service._load_error = "Mock offline testing"

    results = service.locate(sample_image, prompt="motorcycle", top_k=5)
    assert len(results) >= 1
    det = results[0]
    assert 0.0 <= det.confidence <= 1.0
    assert 0.0 <= det.bbox.x1 <= det.bbox.x2 <= 1.0
    assert 0.0 <= det.bbox.y1 <= det.bbox.y2 <= 1.0
    assert service.execution_mode == "fallback_heuristic"


def test_api_locate_endpoint(sample_image: Path) -> None:
    client = TestClient(app)
    payload = {
        "prompt": "traffic light",
        "image_path": str(sample_image),
        "top_k": 3,
        "confidence_threshold": 0.2,
    }
    response = client.post("/api/retrieval/locate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prompt"] == "traffic light"
    assert data["model"] == "nvidia/LocateAnything-3B"
    assert data["execution_mode"] in {"real_model", "fallback_heuristic"}
    assert isinstance(data["detections"], list)
    for item in data["detections"]:
        bbox = item["bbox"]
        assert 0.0 <= bbox["x1"] <= bbox["x2"] <= 1.0
        assert 0.0 <= bbox["y1"] <= bbox["y2"] <= 1.0


def test_api_locate_status_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/api/retrieval/locate/status")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "nvidia/LocateAnything-3B"
    assert "device" in data
    assert "is_loaded" in data
    assert "execution_mode" in data
    assert "cuda_available" in data
