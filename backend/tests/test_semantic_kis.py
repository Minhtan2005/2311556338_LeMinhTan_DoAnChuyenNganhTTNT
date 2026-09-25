from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import Base, get_db
from app.main import app
from app.models import Detection, SemanticEmbedding, Video, VideoKeyframe
from app.services.keyframes import keyframe_extractor
from app.services.kis import kis_service
from app.services.kis_benchmark import run_kis_benchmark
from app.services.embedding_providers import TransformersCLIPProvider
from app.services.semantic_index import normalize_vector, semantic_index_service
from app.services.video_names import competition_video_name


@pytest.fixture()
def db_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    monkeypatch.setattr("app.core.config.settings.upload_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr("app.core.config.settings.keyframe_interval_seconds", 1.0)
    monkeypatch.setattr("app.core.config.settings.max_keyframes_per_video", 5)
    Path(tmp_path / "uploads").mkdir(parents=True, exist_ok=True)
    try:
        yield session
    finally:
        session.close()


def create_video_file(path: Path) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 5.0, (160, 90))
    assert writer.isOpened()
    for index in range(15):
        frame = np.zeros((90, 160, 3), dtype=np.uint8)
        cv2.rectangle(frame, (20 + index, 20), (60 + index, 70), (255, 255, 255), -1)
        cv2.rectangle(frame, (90, 30), (140, 65), (0, 0, 255), -1)
        writer.write(frame)
    writer.release()


def add_completed_video(db: Session, video_path: Path, name: str = "L00_V000.mp4", stored_name: str | None = None) -> Video:
    video = Video(
        original_filename=name,
        stored_filename=stored_name or video_path.name,
        stored_path=str(video_path),
        content_type="video/mp4",
        status="completed",
        duration_seconds=3.0,
        fps=5.0,
        total_frames=15,
        progress_percent=100,
        created_at=datetime.utcnow(),
    )
    db.add(video)
    db.commit()
    db.refresh(video)
    return video


def add_person_motorcycle_detection(db: Session, video_id: str, frame_index: int = 0) -> None:
    db.add_all(
        [
            Detection(
                video_id=video_id,
                frame_index=frame_index,
                timestamp=frame_index / 5.0,
                track_id=1,
                class_name="person",
                confidence=0.9,
                x1=20,
                y1=20,
                x2=60,
                y2=80,
            ),
            Detection(
                video_id=video_id,
                frame_index=frame_index,
                timestamp=frame_index / 5.0,
                track_id=2,
                class_name="motorcycle",
                confidence=0.88,
                x1=70,
                y1=30,
                x2=130,
                y2=75,
            ),
        ]
    )
    db.commit()


def test_keyframe_extraction_metadata(db_session: Session, tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    create_video_file(video_path)
    video = add_completed_video(db_session, video_path)

    result = keyframe_extractor.extract_for_video(db_session, video)

    assert result.total > 0
    keyframe = db_session.query(VideoKeyframe).filter(VideoKeyframe.video_id == video.id).first()
    assert keyframe is not None
    assert keyframe.frame_id >= 0
    assert keyframe.timestamp_sec >= 0
    assert Path(keyframe.image_path).exists()


def test_embedding_normalization() -> None:
    vector = normalize_vector(np.asarray([3.0, 4.0], dtype=np.float32))

    assert np.linalg.norm(vector) == pytest.approx(1.0)


def test_real_clip_embedding_dimensions_and_normalization(tmp_path: Path) -> None:
    image_path = tmp_path / "red.jpg"
    cv2.imwrite(str(image_path), np.full((96, 96, 3), (0, 0, 255), dtype=np.uint8))
    provider = TransformersCLIPProvider()

    text_vector = provider.encode_text("a red square")
    image_vector = provider.encode_image(image_path)

    assert provider.embedding_dimension == 512
    assert text_vector.shape == image_vector.shape == (512,)
    assert np.linalg.norm(text_vector) == pytest.approx(1.0, abs=1e-5)
    assert np.linalg.norm(image_vector) == pytest.approx(1.0, abs=1e-5)


def test_real_clip_related_text_scores_higher_than_unrelated(tmp_path: Path) -> None:
    image_path = tmp_path / "red.jpg"
    cv2.imwrite(str(image_path), np.full((96, 96, 3), (0, 0, 255), dtype=np.uint8))
    provider = TransformersCLIPProvider()

    image_vector = provider.encode_image(image_path)
    related = float(np.dot(provider.encode_text("a red image"), image_vector))
    unrelated = float(np.dot(provider.encode_text("a black dog"), image_vector))

    assert related > unrelated


def test_vector_similarity_ranking_and_persistence(db_session: Session, tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    create_video_file(video_path)
    video = add_completed_video(db_session, video_path)
    add_person_motorcycle_detection(db_session, video.id)

    semantic_index_service.index_video(db_session, video)
    before = db_session.query(SemanticEmbedding).count()

    db_session.expire_all()
    results = kis_service.search(db_session, "Tìm cảnh một người đứng cạnh xe máy", top_k=3)

    assert before > 0
    assert results
    assert results[0].video_name == "L00_V000"
    assert results[0].original_filename == "L00_V000.mp4"
    assert results[0].frame_id >= 0
    assert results[0].score > 0
    assert results[0].competition_output == f"{results[0].video_name},{results[0].frame_id}"


def test_kis_endpoint_and_top_k(db_session: Session, tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    create_video_file(video_path)
    video = add_completed_video(db_session, video_path)
    add_person_motorcycle_detection(db_session, video.id)
    semantic_index_service.index_video(db_session, video)

    def override_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_db
    try:
        client = TestClient(app)
        response = client.post("/api/retrieval/kis", json={"query": "người cạnh xe máy", "top_k": 1})
        assert response.status_code == 200
        payload = response.json()
        assert len(payload["results"]) == 1
        assert payload["model"]
        assert payload["results"][0]["semantic_score"] > 0
        assert payload["competition_preview"][0].startswith("L00_V000,")
    finally:
        app.dependency_overrides.clear()


def test_kis_handles_no_indexed_videos(db_session: Session) -> None:
    results = kis_service.search(db_session, "người cạnh xe máy", top_k=20)

    assert results == []


def test_vietnamese_unicode_query_embedding() -> None:
    vector = semantic_index_service.encode_text("Người mặc áo đỏ đứng cạnh xe máy")

    assert float(np.linalg.norm(vector)) > 0


def test_model_version_persistence(db_session: Session, tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    create_video_file(video_path)
    video = add_completed_video(db_session, video_path)

    semantic_index_service.index_video(db_session, video, force=True)
    row = db_session.query(SemanticEmbedding).first()

    assert row is not None
    assert row.model_name == semantic_index_service.model_name
    assert row.model_version == semantic_index_service.model_version
    assert row.vector_dim == semantic_index_service.embedding_dimension


def test_cross_video_ranking_and_uuid_storage_name(db_session: Session, tmp_path: Path) -> None:
    first_path = tmp_path / "first.mp4"
    second_path = tmp_path / "second.mp4"
    create_video_file(first_path)
    create_video_file(second_path)
    first = add_completed_video(db_session, first_path, name="L01_V028.mp4", stored_name="uuid-one.mp4")
    second = add_completed_video(db_session, second_path, name="L02_V001.mp4", stored_name="uuid-two.mp4")
    add_person_motorcycle_detection(db_session, first.id)
    db_session.add(
        Detection(
            video_id=second.id,
            frame_index=0,
            timestamp=0.0,
            track_id=3,
            class_name="car",
            confidence=0.9,
            x1=10,
            y1=10,
            x2=80,
            y2=80,
        )
    )
    db_session.commit()

    semantic_index_service.index_video(db_session, first, force=True)
    semantic_index_service.index_video(db_session, second, force=True)
    results = kis_service.search(db_session, "A person standing next to a motorcycle", top_k=10)
    names = {item.video_name for item in results}

    assert "L01_V028" in names
    assert "uuid-one" not in names
    assert all(isinstance(item.frame_id, int) for item in results)


def test_competition_video_name_removes_extension(db_session: Session, tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    create_video_file(video_path)
    video = add_completed_video(db_session, video_path, name="L03_V005.mp4", stored_name="internal-uuid.mp4")

    assert competition_video_name(video) == "L03_V005"


def test_empty_manual_benchmark_fixture(db_session: Session) -> None:
    fixture = Path(__file__).parent / "fixtures" / "kis_benchmark.json"

    metrics = run_kis_benchmark(db_session, fixture)

    assert metrics.labels_count == 0
    assert metrics.recall_at_1 == 0.0
