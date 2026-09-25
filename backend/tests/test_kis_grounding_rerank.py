from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import Base
from app.models import Detection, SemanticEmbedding, Video, VideoKeyframe
from app.schemas import (
    GroundingDetection,
    NormalizedBoundingBox,
    StructuredObject,
    StructuredQuery,
    StructuredRelation,
)
from app.services.kis import derive_grounding_prompt, kis_service
from app.services.semantic_index import semantic_index_service


ATTRIBUTE_COLORS = ("red", "blue", "green", "white", "black", "yellow", "gray", "orange", "brown")


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


class FakeColorClipProvider:
    @property
    def model_name(self) -> str:
        return "clip-vit-base-patch32"

    @property
    def model_version(self) -> str:
        return "test"

    @property
    def embedding_dimension(self) -> int:
        return 1 + len(ATTRIBUTE_COLORS)

    def encode_text(self, text: str) -> np.ndarray:
        vector = np.zeros(self.embedding_dimension, dtype=np.float32)
        lowered = text.lower()
        if lowered.startswith("a "):
            for index, color in enumerate(ATTRIBUTE_COLORS, start=1):
                if f" {color} " in f" {lowered} ":
                    vector[index] = 1.0
                    return vector
        vector[0] = 1.0
        return vector

    def encode_image(self, image: str | Path) -> np.ndarray:
        array = cv2.imread(str(image))
        vector = np.zeros(self.embedding_dimension, dtype=np.float32)
        if array is None:
            return vector
        b, g, r = [float(value) for value in array.reshape(-1, 3).mean(axis=0)]
        color = "red" if r > b else "blue"
        vector[ATTRIBUTE_COLORS.index(color) + 1] = 1.0
        return vector

    def encode_text_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.encode_text(text) for text in texts]

    def encode_image_batch(self, images: list[str | Path]) -> list[np.ndarray]:
        return [self.encode_image(image) for image in images]


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_derive_grounding_prompt_for_required_queries():
    # 1. "blue car"
    assert derive_grounding_prompt("blue car") == "blue car"

    # 2. "red car"
    assert derive_grounding_prompt("red car") == "red car"

    # 3. "person wearing red shirt"
    assert derive_grounding_prompt("person wearing red shirt") == "person wearing red shirt"

    # 4. "motorcycle"
    assert derive_grounding_prompt("motorcycle") == "motorcycle"

    # 5. "person next to car"
    assert derive_grounding_prompt("person next to car") == "person next to car"


def test_derive_grounding_prompt_vietnamese():
    assert derive_grounding_prompt("xe hoi mau xanh") == "blue car"
    assert derive_grounding_prompt("nguoi mac ao do") == "person wearing red shirt"
    assert derive_grounding_prompt("xe may") == "motorcycle"
    assert derive_grounding_prompt("nguoi dung canh xe may") == "person next to motorcycle"


def test_derive_grounding_prompt_from_structured_query():
    sq = StructuredQuery(
        task_type="KIS",
        normalized_query_vi="tim xe hoi mau xanh",
        semantic_query_en="a blue car on the street",
        objects=[StructuredObject(type="car", attributes={"color": "blue"})],
    )
    prompt = derive_grounding_prompt("tim xe hoi mau xanh", structured_query=sq)
    assert prompt == "blue car"

    sq_rel = StructuredQuery(
        task_type="KIS",
        normalized_query_vi="nguoi ben canh xe hoi",
        semantic_query_en="a person standing next to a car",
        relations=[StructuredRelation(subject="person", relation="next_to", object="car")],
    )
    prompt_rel = derive_grounding_prompt("nguoi ben canh xe hoi", structured_query=sq_rel)
    assert prompt_rel == "person next to car"


def _setup_test_keyframes(db: Session, count: int = 3, class_name: str = "car"):
    video = Video(
        original_filename="L01_V001.mp4",
        stored_filename="L01_V001.mp4",
        stored_path="uploads/L01_V001.mp4",
        status="completed",
        duration_seconds=10.0,
        fps=5.0,
    )
    db.add(video)
    db.flush()

    items = []
    for i in range(count):
        kf = VideoKeyframe(
            video_id=video.id,
            frame_id=i * 10,
            timestamp_sec=float(i * 2.0),
            image_path=f"uploads/keyframes/test_frame_{i}.jpg",
        )
        db.add(kf)
        db.flush()

        db.add(
            Detection(
                video_id=video.id,
                frame_index=i * 10,
                timestamp=float(i * 2.0),
                track_id=i + 1,
                class_name=class_name,
                confidence=0.9,
                x1=10,
                y1=10,
                x2=50,
                y2=50,
            )
        )

        # Decreasing base semantic scores: 0.90, 0.80, 0.70
        s = 0.90 - i * 0.10
        emb = SemanticEmbedding(
            keyframe_id=kf.id,
            video_id=video.id,
            model_name="clip-vit-base-patch32",
            model_version="test",
            vector_dim=2,
            vector_json=f"[{s:.2f},0.0]",
            descriptor_text="test",
        )
        db.add(emb)
        items.append((kf, emb))

    db.commit()
    return video, items


def test_kis_grounding_rerank_promotes_verified_candidate(db_session: Session, monkeypatch):
    video, items = _setup_test_keyframes(db_session, count=3, class_name="car")

    monkeypatch.setattr(semantic_index_service, "provider", FakeProvider())

    mock_grounding = MagicMock()
    mock_grounding.is_loaded = True
    mock_grounding.execution_mode = "real_model"

    def fake_locate(image_path, prompt, **kwargs):
        if "test_frame_1" in str(image_path):
            return [
                GroundingDetection(
                    label="blue car",
                    confidence=0.92,
                    bbox=NormalizedBoundingBox(x1=0.2, y1=0.3, x2=0.6, y2=0.7),
                )
            ]
        return []

    mock_grounding.locate.side_effect = fake_locate

    with patch("app.services.grounding.grounding_service", mock_grounding):
        results, meta = kis_service.search_with_meta(
            db=db_session,
            query="blue car",
            top_k=3,
            enable_grounding_rerank=True,
            grounding_top_n=2,
        )

    assert meta["execution_mode"] == "real_model"
    assert meta["prompt"] == "blue car"
    assert len(results) == 3

    # Frame 1 must be rank 1 (promoted & verified)
    assert results[0].frame_id == 10
    assert results[0].grounding_status == "verified"
    assert len(results[0].grounding_detections) == 1
    assert results[0].competition_output == "L01_V001,10"

    # Frame 0 was evaluated in top_n and demoted
    assert results[0].score > results[1].score


def test_kis_grounding_rerank_demotes_absent_candidate(db_session: Session, monkeypatch):
    video, items = _setup_test_keyframes(db_session, count=3, class_name="motorcycle")

    monkeypatch.setattr(semantic_index_service, "provider", FakeProvider())

    mock_grounding = MagicMock()
    mock_grounding.is_loaded = True
    mock_grounding.execution_mode = "real_model"

    def fake_locate(image_path, prompt, **kwargs):
        if "test_frame_1" in str(image_path):
            return [
                GroundingDetection(
                    label="motorcycle",
                    confidence=0.88,
                    bbox=NormalizedBoundingBox(x1=0.1, y1=0.1, x2=0.5, y2=0.5),
                )
            ]
        return []

    mock_grounding.locate.side_effect = fake_locate

    with patch("app.services.grounding.grounding_service", mock_grounding):
        results, meta = kis_service.search_with_meta(
            db=db_session,
            query="motorcycle",
            top_k=3,
            enable_grounding_rerank=True,
            grounding_top_n=2,
        )

    # Frame 1 is promoted to rank 1
    assert results[0].frame_id == 10
    assert results[0].grounding_status == "verified"

    # Frame 0 is demoted
    demoted_item = next(r for r in results if r.frame_id == 0)
    assert demoted_item.grounding_status == "demoted"


def test_kis_grounding_color_attribute_reranks_with_crop_clip(db_session: Session, tmp_path: Path, monkeypatch):
    monkeypatch.setattr("app.core.config.settings.upload_dir", str(tmp_path / "uploads"))
    Path(tmp_path / "uploads").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(semantic_index_service, "provider", FakeColorClipProvider())
    monkeypatch.setattr(kis_service, "_color_match", lambda *_args, **_kwargs: (0.0, "not_required"))

    video = Video(
        original_filename="L01_V002.mp4",
        stored_filename="L01_V002.mp4",
        stored_path=str(tmp_path / "uploads" / "L01_V002.mp4"),
        status="completed",
        duration_seconds=10.0,
        fps=5.0,
    )
    db_session.add(video)
    db_session.flush()

    frames = [
        (0, "red", (0, 0, 255), 0.90),
        (10, "blue", (255, 0, 0), 0.80),
    ]
    for frame_id, _color, bgr, semantic_score in frames:
        image = np.zeros((100, 140, 3), dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (100, 80), bgr, -1)
        path = tmp_path / f"frame_{frame_id}.jpg"
        cv2.imwrite(str(path), image)
        keyframe = VideoKeyframe(
            video_id=video.id,
            frame_id=frame_id,
            timestamp_sec=float(frame_id),
            image_path=str(path),
        )
        db_session.add(keyframe)
        db_session.flush()
        db_session.add(
            Detection(
                video_id=video.id,
                frame_index=frame_id,
                timestamp=float(frame_id),
                track_id=frame_id + 1,
                class_name="car",
                confidence=0.9,
                x1=20,
                y1=20,
                x2=100,
                y2=80,
            )
        )
        vector = [semantic_score] + [0.0] * len(ATTRIBUTE_COLORS)
        db_session.add(
            SemanticEmbedding(
                keyframe_id=keyframe.id,
                video_id=video.id,
                model_name="clip-vit-base-patch32",
                model_version="test",
                vector_dim=1 + len(ATTRIBUTE_COLORS),
                vector_json=str(vector).replace(" ", ""),
                descriptor_text="test",
            )
        )

    db_session.commit()

    mock_grounding = MagicMock()
    mock_grounding.is_loaded = True
    mock_grounding.execution_mode = "real_model"
    mock_grounding.locate.return_value = [
        GroundingDetection(
            label="blue car",
            confidence=0.9,
            bbox=NormalizedBoundingBox(x1=20 / 140, y1=20 / 100, x2=100 / 140, y2=80 / 100),
        )
    ]

    with patch("app.services.grounding.grounding_service", mock_grounding):
        results, meta = kis_service.search_with_meta(
            db=db_session,
            query="blue car",
            top_k=2,
            enable_grounding_rerank=True,
            grounding_top_n=2,
        )

    assert meta["execution_mode"] == "real_model"
    assert [item.frame_id for item in results] == [10, 0]
    assert results[0].requested_color == "blue"
    assert results[0].grounded_bbox is not None
    assert results[0].attribute_scores["blue"] > results[0].attribute_scores["red"]
    assert results[0].predicted_color == "blue"
    assert results[0].attribute_verified is True
    assert results[0].final_rerank_score == results[0].score
    assert results[0].debug_crop_path is not None
    assert Path(results[0].debug_crop_path).exists()


def test_kis_grounding_fallback_preserves_clip_ranking_when_heuristic_or_offline(db_session: Session, monkeypatch):
    video, items = _setup_test_keyframes(db_session, count=3, class_name="car")

    monkeypatch.setattr(semantic_index_service, "provider", FakeProvider())

    # Execution mode is fallback_heuristic: must NOT use fallback boxes; preserve CLIP order
    mock_grounding = MagicMock()
    mock_grounding.is_loaded = False
    mock_grounding.execution_mode = "fallback_heuristic"

    with patch("app.services.grounding.grounding_service", mock_grounding):
        results, meta = kis_service.search_with_meta(
            db=db_session,
            query="blue car",
            top_k=3,
            enable_grounding_rerank=True,
        )

    assert meta["status"] == "skipped_real_model_unavailable"
    # Original CLIP order: frame 0 > frame 10 > frame 20
    assert [r.frame_id for r in results] == [0, 10, 20]
    assert results[0].grounding_status is None


def test_kis_grounding_fallback_on_exception(db_session: Session, monkeypatch):
    video, items = _setup_test_keyframes(db_session, count=3, class_name="motorcycle")

    monkeypatch.setattr(semantic_index_service, "provider", FakeProvider())

    mock_grounding = MagicMock()
    mock_grounding.is_loaded = True
    mock_grounding.execution_mode = "real_model"
    mock_grounding.locate.side_effect = RuntimeError("CUDA Out of Memory")

    with patch("app.services.grounding.grounding_service", mock_grounding):
        results, meta = kis_service.search_with_meta(
            db=db_session,
            query="motorcycle",
            top_k=3,
            enable_grounding_rerank=True,
            grounding_top_n=2,
        )

    assert len(results) == 3
    assert results[0].competition_output == "L01_V001,0"
