from __future__ import annotations

from datetime import datetime

import numpy as np
import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.session import Base
from app.models import Detection, Video
from app.schemas import StructuredQuery
from app.services.open_vocab_query import build_cache_key, route_object_query
from app.services.query_router import query_routing_service
from app.services.yolo_world_detector import (
    OpenVocabDetection,
    OpenVocabSearch,
    OpenVocabSegment,
    YOLOWorldDetector,
    deduplicate_temporal_detections,
)


def test_vietnamese_alias_normalization_primary_classes() -> None:
    assert route_object_query("T\u00ecm \u00f4 t\u00f4").object_name == "car"
    assert route_object_query("T\u00ecm xe m\u00e1y").object_name == "motorcycle"
    assert route_object_query("T\u00ecm ng\u01b0\u1eddi").object_name == "person"


def test_open_vocabulary_routing_examples() -> None:
    assert route_object_query("T\u00ecm con ch\u00f3").detector == "yolo_world"
    assert route_object_query("T\u00ecm con ch\u00f3").object_name == "dog"
    assert route_object_query("T\u00ecm xe c\u1ee9u th\u01b0\u01a1ng").object_name == "ambulance"
    assert route_object_query("T\u00ecm traffic cone").object_name == "traffic cone"
    assert route_object_query("T\u00ecm c\u00e1i \u00f4").object_name == "umbrella"
    assert route_object_query("C\u00f3 m\u00e8o kh\u00f4ng?").object_name == "cat"
    assert route_object_query("T\u00ecm vali").object_name == "suitcase"
    assert route_object_query("T\u00ecm gh\u1ebf").object_name == "chair"
    assert route_object_query("T\u00ecm bi\u1ec3n b\u00e1o").object_name == "traffic sign"


def test_unsupported_attribute_and_relationship_are_preserved() -> None:
    color = route_object_query("xe \u00f4 t\u00f4 m\u00e0u v\u00e0ng")
    relation = route_object_query("ng\u01b0\u1eddi \u0111ang \u0111i xe m\u00e1y")

    assert color.status == "unsupported_attribute"
    assert relation.status == "unsupported_relationship"


def test_malformed_object_query_is_rejected() -> None:
    routed = route_object_query("tim")

    assert routed.detector == "none"
    assert routed.status == "unsupported_query"


def test_cache_key_includes_required_dimensions() -> None:
    key = build_cache_key(
        video_id="video-1",
        object_name="Dog",
        model="yolov8s-worldv2.pt",
        confidence=0.25,
        sample_fps=2.0,
    )

    assert key == "video-1|dog|yolov8s-worldv2.pt|0.2500|2.0000"


def test_temporal_dedup_groups_nearby_detections() -> None:
    detections = [
        OpenVocabDetection("dog", 0.50, [0, 0, 10, 10], 10, 20.0),
        OpenVocabDetection("dog", 0.81, [1, 1, 11, 11], 11, 20.5),
        OpenVocabDetection("dog", 0.70, [2, 2, 12, 12], 12, 21.0),
        OpenVocabDetection("dog", 0.60, [3, 3, 13, 13], 30, 25.0),
    ]

    segments = deduplicate_temporal_detections(detections, gap_seconds=1.5)

    assert len(segments) == 2
    assert segments[0].start_time == 20.0
    assert segments[0].end_time == 21.0
    assert segments[0].timestamp == 20.5
    assert segments[0].confidence == 0.81


def test_normalized_detection_schema_from_model_result() -> None:
    class ArrayLike:
        def __init__(self, values):
            self.values = np.array(values)

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class Boxes:
        xyxy = ArrayLike([[1, 2, 30, 40]])
        conf = ArrayLike([0.73])
        cls = ArrayLike([0])

    class Result:
        boxes = Boxes()
        names = {0: "ambulance"}

    detections = YOLOWorldDetector._result_to_detections(
        Result(),
        ["ambulance"],
        frame_id=1234,
        timestamp=41.13,
    )

    assert detections == [
        OpenVocabDetection(
            class_name="ambulance",
            confidence=0.73,
            bbox=[1.0, 2.0, 30.0, 40.0],
            frame_id=1234,
            timestamp=41.13,
        )
    ]


def test_query_router_uses_yolo_world_for_open_vocab(monkeypatch, tmp_path) -> None:
    db = _make_db()
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"not used by mocked detector")
    video = _add_video(db, str(video_path))

    def fake_search_video(**kwargs):
        return OpenVocabSearch(
            query_object=kwargs["query_object"],
            detector="yolo_world",
            model="mock-yolo-world",
            confidence=0.25,
            sample_fps=2.0,
            cache_hit=False,
            detections=[
                OpenVocabDetection("dog", 0.9, [1, 2, 3, 4], 5, 1.0),
            ],
            segments=[
                OpenVocabSegment("dog", 1.0, 1.0, 1.0, 5, 0.9, [1, 2, 3, 4]),
            ],
        )

    monkeypatch.setattr("app.services.query_router.yolo_world_detector.search_video", fake_search_video)
    parsed = StructuredQuery(task_type="LEGACY", normalized_query_vi="T\u00ecm con ch\u00f3")

    response = query_routing_service._route_legacy(
        db,
        video.id,
        "T\u00ecm con ch\u00f3",
        "local_fallback",
        parsed,
        None,
    )

    assert response.status == "ok"
    assert response.debug.entities["detector"] == "yolo_world"
    assert response.segments[0].detector == "yolo_world"
    assert response.segments[0].frame_id == 5


def test_query_router_uses_primary_metadata_for_unaccented_car_query(tmp_path) -> None:
    db = _make_db()
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"not used")
    video = _add_video(db, str(video_path))
    db.add(
        Detection(
            video_id=video.id,
            frame_index=0,
            timestamp=0.0,
            track_id=1,
            class_name="car",
            confidence=0.8,
            x1=0,
            y1=0,
            x2=10,
            y2=10,
        )
    )
    db.commit()
    parsed = StructuredQuery(task_type="LEGACY", normalized_query_vi="Tim o to")

    response = query_routing_service._route_legacy(
        db,
        video.id,
        "Tim o to",
        "local_fallback",
        parsed,
        None,
    )

    assert response.debug.intent == "search_object"
    assert response.debug.entities["object"] == "car"
    assert response.segments
    assert response.segments[0].label == "car track #1"


def test_yolo_world_disabled_behavior(monkeypatch, tmp_path) -> None:
    db = _make_db()
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"not used")
    video = _add_video(db, str(video_path))
    monkeypatch.setattr("app.services.query_router.settings.yolo_world_enabled", False)
    parsed = StructuredQuery(task_type="LEGACY", normalized_query_vi="T\u00ecm con ch\u00f3")

    response = query_routing_service._route_open_vocab(
        db,
        video.id,
        "T\u00ecm con ch\u00f3",
        "local_fallback",
        parsed,
        "dog",
    )

    assert response.status == "yolo_world_disabled"
    assert response.segments == []


def test_yolo_world_detector_keeps_device_consistent_across_prompt_changes(monkeypatch) -> None:
    class EmptyResult:
        boxes = None

    class FakeClip:
        def __init__(self) -> None:
            self.device = "stale"
            self.model = torch.nn.Linear(1, 1)

    class FakeWorld:
        def __init__(self) -> None:
            self.clip_model = FakeClip()

    class FakeYOLOWorld:
        def __init__(self) -> None:
            self.model = FakeWorld()
            self.set_class_records = []
            self.predict_devices = []

        def to(self, device: str):
            self.model.clip_model.model.to(device)
            return self

        def set_classes(self, classes: list[str]) -> None:
            self.set_class_records.append((tuple(classes), self.model.clip_model.device))

        def predict(self, frame, *, conf: float, verbose: bool, device: str):
            self.predict_devices.append(device)
            return [EmptyResult()]

    detector = YOLOWorldDetector()
    fake_model = FakeYOLOWorld()
    detector._model = fake_model
    detector._loaded_model_name = settings.yolo_world_model
    monkeypatch.setattr(detector, "_resolve_device", lambda: "cpu")

    detector.detect_frame(object(), ["white car"])
    fake_model.model.clip_model.device = "stale"
    detector.detect_frame(object(), ["black car"])

    assert fake_model.set_class_records == [(("white car",), "cpu"), (("black car",), "cpu")]
    assert fake_model.predict_devices == ["cpu", "cpu"]


def _make_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return testing_session_local()


def _add_video(db, stored_path: str) -> Video:
    video = Video(
        original_filename="sample.mp4",
        stored_filename="sample.mp4",
        stored_path=stored_path,
        content_type="video/mp4",
        status="completed",
        duration_seconds=1.0,
        fps=5.0,
        total_frames=5,
        progress_percent=100,
        created_at=datetime.utcnow(),
    )
    db.add(video)
    db.commit()
    db.refresh(video)
    return video
