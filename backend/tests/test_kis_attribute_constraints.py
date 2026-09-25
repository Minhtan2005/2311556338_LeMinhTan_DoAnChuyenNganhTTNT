from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import Base
from app.models import Detection, SemanticEmbedding, Video, VideoKeyframe
from app.schemas import StructuredObject, StructuredQuery
from app.services.kis import (
    ColorConstraint,
    RelationConstraint,
    extract_query_signals,
    kis_service,
)
from app.services.semantic_index import normalize_vector, semantic_index_service


@pytest.fixture()
def db_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    monkeypatch.setattr("app.core.config.settings.upload_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr("app.core.config.settings.kis_object_crop_semantic_weight", 0.0)
    kis_service._detections_cache.clear()
    kis_service._color_cache.clear()
    kis_service._crop_vector_cache.clear()
    Path(tmp_path / "uploads").mkdir(parents=True, exist_ok=True)
    try:
        yield session
    finally:
        session.close()


def test_query_aliases_do_not_confuse_tim_with_purple_or_canh_with_next_to() -> None:
    signals = extract_query_signals("Tim canh co o to mau xanh")

    assert ColorConstraint("car", "blue", "full") in signals.color_constraints
    assert all(item.color != "purple" for item in signals.color_constraints)
    assert signals.relation_constraints == ()
    assert not signals.wants_near


def test_structured_query_extracts_multiple_objects_color_and_relation() -> None:
    parsed = StructuredQuery(
        task_type="KIS",
        normalized_query_vi="Nguoi ao do dung canh xe may",
        semantic_query_en="a person wearing a red shirt standing next to a motorcycle",
        objects=[
            StructuredObject(type="person", attributes={"upper_color": "red"}),
            StructuredObject(type="motorcycle", attributes={}),
        ],
        actions=["standing"],
        relations=[{"subject": "person", "relation": "next_to", "object": "motorcycle"}],
        confidence=0.95,
    )

    signals = extract_query_signals(parsed.semantic_query_en or "", structured_query=parsed)

    assert signals.required_objects == {"person", "motorcycle"}
    assert ColorConstraint("person", "red", "upper") in signals.color_constraints
    assert RelationConstraint("person", "next_to", "motorcycle") in signals.relation_constraints


def test_attribute_aware_kis_rejects_wrong_color_and_missing_object(db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(semantic_index_service, "provider", FakeProvider())

    video = Video(
        original_filename="L00_V000.mp4",
        stored_filename="L00_V000.mp4",
        stored_path=str(tmp_path / "video.mp4"),
        content_type="video/mp4",
        status="completed",
        duration_seconds=1.0,
        fps=5.0,
        total_frames=5,
        progress_percent=100,
        created_at=datetime.utcnow(),
    )
    db_session.add(video)
    db_session.commit()
    db_session.refresh(video)

    red_car_keyframe = add_keyframe_with_detection(db_session, tmp_path, video.id, frame_id=1, color_bgr=(0, 0, 255), class_name="car")
    blue_car_keyframe = add_keyframe_with_detection(db_session, tmp_path, video.id, frame_id=2, color_bgr=(255, 0, 0), class_name="car")
    add_embedding(db_session, video.id, red_car_keyframe.id)
    add_embedding(db_session, video.id, blue_car_keyframe.id)

    parsed = StructuredQuery(
        task_type="KIS",
        normalized_query_vi="O to mau xanh",
        semantic_query_en="a blue car",
        objects=[StructuredObject(type="car", attributes={"vehicle_color": "blue"})],
        confidence=0.9,
    )

    results = kis_service.search(db_session, "a blue car", top_k=5, structured_query=parsed)

    assert [item.frame_id for item in results] == [2]


def test_attribute_aware_kis_requires_near_relation(db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(semantic_index_service, "provider", FakeProvider())

    video = Video(
        original_filename="L00_V001.mp4",
        stored_filename="L00_V001.mp4",
        stored_path=str(tmp_path / "video.mp4"),
        content_type="video/mp4",
        status="completed",
        duration_seconds=1.0,
        fps=5.0,
        total_frames=5,
        progress_percent=100,
        created_at=datetime.utcnow(),
    )
    db_session.add(video)
    db_session.commit()
    db_session.refresh(video)

    far = add_relation_keyframe(db_session, tmp_path, video.id, frame_id=1, close=False)
    close = add_relation_keyframe(db_session, tmp_path, video.id, frame_id=3, close=True)
    add_embedding(db_session, video.id, far.id)
    add_embedding(db_session, video.id, close.id)

    parsed = StructuredQuery(
        task_type="KIS",
        normalized_query_vi="Nguoi dung canh xe may",
        semantic_query_en="a person standing next to a motorcycle",
        objects=[StructuredObject(type="person", attributes={}), StructuredObject(type="motorcycle", attributes={})],
        relations=[{"subject": "person", "relation": "next_to", "object": "motorcycle"}],
        confidence=0.9,
    )

    results = kis_service.search(db_session, "a person standing next to a motorcycle", top_k=5, structured_query=parsed)

    assert [item.frame_id for item in results] == [3]


def add_keyframe_with_detection(
    db: Session,
    tmp_path: Path,
    video_id: str,
    frame_id: int,
    color_bgr: tuple[int, int, int],
    class_name: str,
) -> VideoKeyframe:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.rectangle(image, (40, 35), (120, 90), color_bgr, -1)
    path = tmp_path / f"frame_{frame_id}.jpg"
    cv2.imwrite(str(path), image)
    keyframe = VideoKeyframe(video_id=video_id, frame_id=frame_id, timestamp_sec=float(frame_id), image_path=str(path))
    db.add(keyframe)
    db.flush()
    db.add(
        Detection(
            video_id=video_id,
            frame_index=frame_id,
            timestamp=float(frame_id),
            track_id=frame_id,
            class_name=class_name,
            confidence=0.9,
            x1=40,
            y1=35,
            x2=120,
            y2=90,
        )
    )
    db.commit()
    db.refresh(keyframe)
    return keyframe


def add_relation_keyframe(db: Session, tmp_path: Path, video_id: str, frame_id: int, close: bool) -> VideoKeyframe:
    image = np.zeros((120, 180, 3), dtype=np.uint8)
    path = tmp_path / f"relation_{frame_id}.jpg"
    cv2.imwrite(str(path), image)
    keyframe = VideoKeyframe(video_id=video_id, frame_id=frame_id, timestamp_sec=float(frame_id), image_path=str(path))
    db.add(keyframe)
    db.flush()
    motorcycle_x1 = 75 if close else 140
    db.add_all(
        [
            Detection(
                video_id=video_id,
                frame_index=frame_id,
                timestamp=float(frame_id),
                track_id=frame_id * 10,
                class_name="person",
                confidence=0.9,
                x1=20,
                y1=25,
                x2=60,
                y2=105,
            ),
            Detection(
                video_id=video_id,
                frame_index=frame_id,
                timestamp=float(frame_id),
                track_id=frame_id * 10 + 1,
                class_name="motorcycle",
                confidence=0.9,
                x1=motorcycle_x1,
                y1=45,
                x2=motorcycle_x1 + 35,
                y2=95,
            ),
        ]
    )
    db.commit()
    db.refresh(keyframe)
    return keyframe


def add_embedding(db: Session, video_id: str, keyframe_id: int) -> None:
    vector = normalize_vector(np.asarray([1.0, 0.0], dtype=np.float32))
    db.add(
        SemanticEmbedding(
            keyframe_id=keyframe_id,
            video_id=video_id,
            model_name="clip-vit-base-patch32",
            model_version="test",
            vector_dim=2,
            vector_json="[1.0,0.0]",
            descriptor_text="test",
        )
    )
    db.commit()


class FakeProvider:
    @property
    def model_name(self) -> str:
        return "clip-vit-base-patch32"

    @property
    def model_version(self) -> str:
        return "test"

    @property
    def embedding_dimension(self) -> int:
        return 2

    def encode_text(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)

    def encode_image(self, image: str | Path) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)

    def encode_image_batch(self, images: list[str | Path]) -> list[np.ndarray]:
        return [self.encode_image(image) for image in images]
