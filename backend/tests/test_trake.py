from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import Base
from app.models import SemanticEmbedding, Video, VideoKeyframe
from app.schemas import (
    StructuredEvent,
    StructuredQuery,
    TRAKEResponse,
)
from app.services.query_router import query_routing_service
from app.services.semantic_index import semantic_index_service
from app.services.trake import INSUFFICIENT_EVIDENCE_MESSAGE, trake_service


class MockCLIPProvider:
    """Deterministic embedding provider for TRAKE testing."""
    def __init__(self):
        # 4-dimensional unit vectors
        self.vectors = {
            "event_1": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "event_2": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            "event_3": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            "event_4": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "unknown": np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        }

    def encode_text(self, text: str) -> np.ndarray:
        for k, v in self.vectors.items():
            if k in text.lower():
                return v
        return self.vectors["unknown"]

    def encode_image(self, image) -> np.ndarray:
        return self.vectors["unknown"]


@pytest.fixture()
def db_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()

    mock_provider = MockCLIPProvider()
    monkeypatch.setattr(semantic_index_service, "provider", mock_provider)

    try:
        yield session
    finally:
        session.close()


def add_video_with_keyframes(
    db: Session,
    video_id: str,
    original_filename: str,
    frame_configs: list[tuple[int, float, np.ndarray]],  # (frame_id, timestamp, vector)
) -> Video:
    video = Video(
        id=video_id,
        original_filename=original_filename,
        stored_filename=original_filename,
        stored_path=f"uploads/{original_filename}",
        status="completed",
        duration_seconds=30.0,
        fps=5.0,
        created_at=datetime.utcnow(),
    )
    db.add(video)
    db.flush()

    for fid, ts, vec in frame_configs:
        kf = VideoKeyframe(
            video_id=video.id,
            frame_id=fid,
            timestamp_sec=ts,
            image_path=f"uploads/keyframes/{video.id}/frame_{fid:08d}.jpg",
        )
        db.add(kf)
        db.flush()

        emb = SemanticEmbedding(
            video_id=video.id,
            keyframe_id=kf.id,
            vector_dim=len(vec),
            vector_json=json.dumps(vec.tolist()),
            model_name="clip-mock",
            model_version="test",
        )
        db.add(emb)

    db.commit()
    db.refresh(video)
    return video


def test_trake_two_event_sequence(db_session: Session):
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    v_other = np.array([0.0, 0.0, 0.5, 0.5], dtype=np.float32)

    frames = [
        (10, 2.0, v_other),
        (25, 5.0, v1),      # Matches event_1
        (40, 8.0, v_other),
        (60, 12.0, v2),     # Matches event_2
        (80, 16.0, v_other),
    ]
    video = add_video_with_keyframes(db_session, "vid-01", "L01_V001.mp4", frames)

    events = [
        StructuredEvent(order=1, description_vi="event_1", semantic_query_en="event_1"),
        StructuredEvent(order=2, description_vi="event_2", semantic_query_en="event_2"),
    ]

    res = trake_service.align_events(db_session, events=events, query="test query")

    assert res.status == "ok"
    assert res.video_id == video.id
    assert res.video_name == "L01_V001"
    assert res.frame_ids == [25, 60]
    assert len(res.frame_ids) == 2
    assert res.frame_ids[0] < res.frame_ids[1]
    assert res.competition_output == "L01_V001,25,60"
    assert ".mp4" not in res.competition_output


def test_trake_three_plus_events_monotonic(db_session: Session):
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    v4 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    frames = [
        (5, 1.0, v1),       # Event 1
        (15, 3.0, v2),      # Event 2
        (35, 7.0, v3),      # Event 3
        (55, 11.0, v4),     # Event 4
    ]
    add_video_with_keyframes(db_session, "vid-02", "L02_V010.mp4", frames)

    events = [
        StructuredEvent(order=1, description_vi="event_1", semantic_query_en="event_1"),
        StructuredEvent(order=2, description_vi="event_2", semantic_query_en="event_2"),
        StructuredEvent(order=3, description_vi="event_3", semantic_query_en="event_3"),
        StructuredEvent(order=4, description_vi="event_4", semantic_query_en="event_4"),
    ]

    res = trake_service.align_events(db_session, events=events)

    assert res.status == "ok"
    assert res.frame_ids == [5, 15, 35, 55]
    assert len(res.frame_ids) == 4
    # Strictly monotonically increasing
    for i in range(len(res.frame_ids) - 1):
        assert res.frame_ids[i] < res.frame_ids[i + 1]
    assert res.competition_output == "L02_V010,5,15,35,55"


def test_trake_prevents_duplicate_and_inverted_frames(db_session: Session):
    # Setup scenario where naive independent retrieval would invert or duplicate frames:
    # Frame 10: strong match for event_2
    # Frame 30: strong match for event_1
    # Frame 50: moderate match for event_2
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2_strong = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    v2_mod = np.array([0.0, 0.8, 0.2, 0.0], dtype=np.float32)

    frames = [
        (10, 2.0, v2_strong),  # Event 2 has highest raw score here
        (30, 6.0, v1),         # Event 1 is here
        (50, 10.0, v2_mod),    # Event 2 is also here with lower score
    ]
    add_video_with_keyframes(db_session, "vid-03", "L03_V005.mp4", frames)

    events = [
        StructuredEvent(order=1, description_vi="event_1", semantic_query_en="event_1"),
        StructuredEvent(order=2, description_vi="event_2", semantic_query_en="event_2"),
    ]

    res = trake_service.align_events(db_session, events=events)

    assert res.status == "ok"
    # Dynamic Programming must choose frame 30 for Event 1 and frame 50 for Event 2!
    # Frame 10 for Event 2 would violate monotonicity (30 < 10 is False)!
    assert res.frame_ids == [30, 50]
    assert res.frame_ids[0] < res.frame_ids[1]
    assert len(set(res.frame_ids)) == 2  # No duplicates


def test_trake_same_video_constraint(db_session: Session):
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    v_weak = np.array([0.2, 0.2, 0.0, 0.0], dtype=np.float32)

    # Video A: strong for both events
    frames_a = [
        (10, 2.0, v1),
        (30, 6.0, v2),
    ]
    vid_a = add_video_with_keyframes(db_session, "vid-a", "L01_VA.mp4", frames_a)

    # Video B: weak for both events
    frames_b = [
        (15, 3.0, v_weak),
        (35, 7.0, v_weak),
    ]
    add_video_with_keyframes(db_session, "vid-b", "L01_VB.mp4", frames_b)

    events = [
        StructuredEvent(order=1, description_vi="event_1", semantic_query_en="event_1"),
        StructuredEvent(order=2, description_vi="event_2", semantic_query_en="event_2"),
    ]

    res = trake_service.align_events(db_session, events=events)

    # Must choose Video A entirely, never mix Video A and Video B
    assert res.status == "ok"
    assert res.video_id == vid_a.id
    assert res.video_name == "L01_VA"
    assert res.frame_ids == [10, 30]


def test_trake_insufficient_evidence_refusal(db_session: Session):
    # Video with only unrelated frames
    v_unrelated = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    frames = [
        (10, 2.0, v_unrelated),
        (20, 4.0, v_unrelated),
    ]
    add_video_with_keyframes(db_session, "vid-empty", "L99_V099.mp4", frames)

    events = [
        StructuredEvent(order=1, description_vi="event_1", semantic_query_en="event_1"),
        StructuredEvent(order=2, description_vi="event_2", semantic_query_en="event_2"),
    ]

    res = trake_service.align_events(db_session, events=events)

    assert res.status == "insufficient_evidence"
    assert res.message == INSUFFICIENT_EVIDENCE_MESSAGE
    assert res.frame_ids == []
    assert res.competition_output is None


def test_trake_query_router_offline_parsing_and_competition_output(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
):
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    frames = [
        (12, 2.4, v1),
        (34, 6.8, v2),
        (56, 11.2, v3),
    ]
    add_video_with_keyframes(db_session, "vid-router", "L05_V001.mp4", frames)

    # Test offline query parsing with Vietnamese transition words: "rồi", "sau đó"
    query = "event_1 rồi event_2 sau đó event_3"

    # Simulate Gemini offline error so fallback is triggered
    monkeypatch.setattr(
        "app.services.query_router.gemini_nlp_service.analyze_query",
        MagicMock(side_effect=RuntimeError("API offline")),
    )

    resp = query_routing_service.route_query(db_session, video_id=None, question=query)

    assert resp.task_type == "TRAKE"
    assert resp.status == "ok"
    assert len(resp.competition_preview) == 1
    assert resp.competition_preview[0] == "L05_V001,12,34,56"
    assert ".mp4" not in resp.competition_preview[0]
