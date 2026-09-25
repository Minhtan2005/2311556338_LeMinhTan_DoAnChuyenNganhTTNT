from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import Base
from app.models import SemanticEmbedding, Video, VideoKeyframe
from app.schemas import (
    KISResult,
    QueryRequest,
    StructuredQuery,
)
from app.services.query_router import query_routing_service
from app.services.visual_qa import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    MAX_QA_ANSWER_LENGTH,
    VisualQAResult,
    visual_qa_service,
)


@pytest.fixture()
def db_session(tmp_path: Path):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()

    # Create dummy video and keyframe
    video = Video(
        id="vid-test-01",
        original_filename="L01_V001.mp4",
        stored_filename="L01_V001.mp4",
        stored_path="uploads/L01_V001.mp4",
        status="completed",
        duration_seconds=10.0,
        fps=5.0,
    )
    session.add(video)
    session.flush()

    dummy_img = tmp_path / "frame_00000010.jpg"
    dummy_img.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb")

    kf = VideoKeyframe(
        id=1,
        video_id=video.id,
        frame_id=10,
        timestamp_sec=2.0,
        image_path=str(dummy_img),
    )
    session.add(kf)
    session.commit()

    try:
        yield session
    finally:
        session.close()


def test_answer_length_enforcement():
    # Long text exceeding 100 characters must be truncated to <= 100
    long_text = "Đây là một câu trả lời rất dài được sinh ra nhằm mục đích kiểm tra xem hệ thống có cắt bớt câu trả lời khi vượt quá một trăm ký tự hay không."
    cleaned = visual_qa_service._clean_and_truncate_answer(long_text)
    assert len(cleaned) <= MAX_QA_ANSWER_LENGTH
    assert len(cleaned) <= 100


def test_visual_qa_routing_and_competition_output(db_session: Session, monkeypatch: pytest.MonkeyPatch):
    # Mock kis_service.search to return our dummy keyframe result
    mock_kis_result = KISResult(
        video_id="vid-test-01",
        video_name="L01_V001",
        original_filename="L01_V001.mp4",
        frame_id=10,
        timestamp=2.0,
        semantic_score=0.95,
        rerank_score=0.1,
        score=0.98,
        image_url="/media/keyframes/frame_10.jpg",
        image_path="uploads/keyframes/frame_10.jpg",
        competition_output="L01_V001,10",
    )
    monkeypatch.setattr("app.services.query_router.kis_service.search", lambda *args, **kwargs: [mock_kis_result])

    # Mock visual_qa_service.answer_question to return factual answer
    mock_qa_result = VisualQAResult(
        answer="Xe ô tô màu xanh dương",
        confidence=0.95,
        model="gemini-2.5-flash",
        latency_ms=450.0,
    )
    monkeypatch.setattr("app.services.visual_qa.visual_qa_service.answer_question", lambda *args, **kwargs: mock_qa_result)

    # Mock query understanding to return Q&A task
    parsed_q = StructuredQuery(
        task_type="Q&A",
        normalized_query_vi="chiec xe o to mau gi",
        semantic_query_en="a car on the road",
        question="Chiếc xe ô tô màu gì?",
    )
    monkeypatch.setattr(
        query_routing_service,
        "analyze_query_with_fallback",
        lambda q: ("gemini", parsed_q, None, 100.0),
    )

    resp = query_routing_service.route_query(db_session, video_id="vid-test-01", question="Chiếc xe ô tô màu gì?")

    assert resp.task_type == "Q&A"
    assert resp.answer == "Xe ô tô màu xanh dương"
    assert len(resp.answer) <= 100
    assert len(resp.competition_preview) == 1

    # Strictly check competition format: <video_name>,<frame_id>,<answer> without .mp4
    comp_row = resp.competition_preview[0]
    assert comp_row == "L01_V001,10,Xe ô tô màu xanh dương"
    assert ".mp4" not in comp_row


def test_visual_qa_insufficient_evidence_when_no_keyframes(db_session: Session, monkeypatch: pytest.MonkeyPatch):
    # When KIS returns empty candidate list
    monkeypatch.setattr("app.services.query_router.kis_service.search", lambda *args, **kwargs: [])

    parsed_q = StructuredQuery(
        task_type="Q&A",
        normalized_query_vi="bien so xe la bao nhieu",
        semantic_query_en="license plate",
        question="Biển số xe là bao nhiêu?",
    )
    monkeypatch.setattr(
        query_routing_service,
        "analyze_query_with_fallback",
        lambda q: ("gemini", parsed_q, None, 50.0),
    )

    resp = query_routing_service.route_query(db_session, video_id="vid-test-01", question="Biển số xe là bao nhiêu?")
    assert resp.task_type == "Q&A"
    assert "Không tìm thấy" in resp.answer
    assert resp.competition_preview == []


def test_visual_qa_graceful_fallback_on_api_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # When Gemini API raises an exception
    dummy_img = tmp_path / "test.jpg"
    dummy_img.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb")

    def mock_fail(*args, **kwargs):
        raise RuntimeError("API quota exceeded")

    monkeypatch.setattr(visual_qa_service, "_call_gemini_vision", mock_fail)

    result = visual_qa_service.answer_question(
        image_path=dummy_img,
        question="Người trong hình đang cầm gì?",
    )

    # Must not crash, returns safe insufficient evidence message
    assert result.is_insufficient_evidence is True
    assert INSUFFICIENT_EVIDENCE_ANSWER in result.answer
    assert len(result.answer) <= 100


def test_visual_qa_local_counting_fallback(tmp_path: Path):
    dummy_img = tmp_path / "test2.jpg"
    dummy_img.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb")

    car_track_1a = MagicMock()
    car_track_1a.class_name = "car"
    car_track_1a.track_id = 10
    car_track_1b = MagicMock()
    car_track_1b.class_name = "car"
    car_track_1b.track_id = 10
    car_track_2 = MagicMock()
    car_track_2.class_name = "car"
    car_track_2.track_id = 11

    result = visual_qa_service._local_fallback(
        question="Có bao nhiêu xe ô tô?",
        context={"detections": [car_track_1a, car_track_1b, car_track_2], "detection_scope": "frame"},
    )

    assert result.answer == "Có 2 ô tô trong khung hình."
    assert len(result.answer) <= 100
    assert result.model == "yolo_detection_fallback"


def test_visual_qa_local_presence_fallback():
    motorcycle = MagicMock()
    motorcycle.class_name = "motorcycle"
    motorcycle.track_id = 7

    result = visual_qa_service._local_fallback(
        question="Có xe máy không?",
        context={"detections": [motorcycle], "detection_scope": "frame"},
    )

    assert result.answer == "Có xe máy trong khung hình."
    assert result.is_insufficient_evidence is False


def test_visual_qa_local_person_count_fallback():
    person_a = MagicMock()
    person_a.class_name = "person"
    person_a.track_id = 1
    person_b = MagicMock()
    person_b.class_name = "person"
    person_b.track_id = 2

    result = visual_qa_service._local_fallback(
        question="Có bao nhiêu người?",
        context={"detections": [person_a, person_b], "detection_scope": "frame"},
    )

    assert result.answer == "Có 2 người trong khung hình."


def test_visual_qa_uses_detection_fallback_after_insufficient_gemini(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dummy_img = tmp_path / "test3.jpg"
    dummy_img.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb")

    monkeypatch.setattr("app.services.visual_qa.settings.gemini_enabled", True)
    monkeypatch.setattr("app.services.visual_qa.settings.gemini_vision_enabled", True)
    monkeypatch.setattr("app.services.visual_qa.settings.gemini_api_key", "demo-key")
    monkeypatch.setattr(
        visual_qa_service,
        "_call_gemini_vision",
        lambda *args, **kwargs: VisualQAResult(
            answer=INSUFFICIENT_EVIDENCE_ANSWER,
            confidence=0.5,
            model="gemini",
            latency_ms=10.0,
            is_insufficient_evidence=True,
        ),
    )

    car = MagicMock()
    car.class_name = "car"
    car.track_id = 5

    result = visual_qa_service.answer_question(
        image_path=dummy_img,
        question="Có bao nhiêu ô tô?",
        context={"detections": [car], "detection_scope": "frame"},
    )

    assert result.answer == "Có 1 ô tô trong khung hình."
    assert result.model == "yolo_detection_fallback"
