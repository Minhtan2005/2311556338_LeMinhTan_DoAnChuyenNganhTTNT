from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.session import Base, get_db
from app.main import app
from app.models import Detection, Video
from app.schemas import KISResult, StructuredQuery
from app.services.gemini_nlp import GeminiNLPError, GeminiNLPResult, gemini_nlp_service
from app.services.query_router import query_routing_service
from app.services.yolo_world_detector import OpenVocabDetection, OpenVocabSearch, OpenVocabSegment


def load_fixture_cases() -> list[dict]:
    fixture = Path(__file__).parent / "fixtures" / "gemini_nlp_cases.json"
    return json.loads(fixture.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", load_fixture_cases())
def test_mocked_gemini_structured_query_schema(case: dict) -> None:
    parsed = StructuredQuery.model_validate(case["mocked_response"])

    assert parsed.task_type == case["expected_task_type"]
    assert 0 <= parsed.confidence <= 1
    if case["expected_task_type"] == "KIS":
        assert parsed.semantic_query_en
    if case.get("expected_events") is not None:
        assert len(parsed.events) == case["expected_events"]


def test_nlp_analyze_endpoint_uses_mocked_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    mocked = StructuredQuery.model_validate(load_fixture_cases()[1]["mocked_response"])

    def fake_analyze(query: str, context=None) -> GeminiNLPResult:
        return GeminiNLPResult(parsed_query=mocked, latency_ms=42.5)

    monkeypatch.setattr(gemini_nlp_service, "analyze_query", fake_analyze)
    client = TestClient(app)

    response = client.post("/api/nlp/analyze", json={"query": "Tim canh nguoi ao do canh xe may"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "gemini"
    assert payload["latency_ms"] == 42.5
    assert payload["parsed_query"]["task_type"] == "KIS"
    assert payload["parsed_query"]["semantic_query_en"] == mocked.semantic_query_en


def test_query_router_sends_gemini_semantic_query_to_kis(monkeypatch: pytest.MonkeyPatch) -> None:
    mocked = StructuredQuery.model_validate(load_fixture_cases()[1]["mocked_response"])
    captured: dict[str, str] = {}

    def fake_analyze(query: str, context=None) -> GeminiNLPResult:
        return GeminiNLPResult(parsed_query=mocked, latency_ms=15.0)

    def fake_search(db, query: str, top_k: int = 20, structured_query=None) -> list[KISResult]:
        captured["query"] = query
        return [
            KISResult(
                video_id="video-1",
                video_name="L00_V000",
                original_filename="L00_V000.mp4",
                frame_id=12,
                timestamp=2.4,
                semantic_score=0.91,
                rerank_score=0.05,
                score=0.824,
                image_url="/media/keyframes/video-1/12.jpg",
                image_path="uploads/keyframes/video-1/12.jpg",
                competition_output="L00_V000,12",
            )
        ]

    monkeypatch.setattr(gemini_nlp_service, "analyze_query", fake_analyze)
    monkeypatch.setattr("app.services.query_router.kis_service.search", fake_search)

    response = query_routing_service.route_query(db=None, video_id="video-1", question=mocked.normalized_query_vi)

    assert captured["query"] == mocked.semantic_query_en
    assert response.nlp_provider == "gemini"
    assert response.task_type == "KIS"
    assert response.kis_results[0]["competition_output"] == "L00_V000,12"


def test_gemini_object_car_routes_to_yolov8_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    session = make_memory_session()
    video = add_completed_video(session)
    session.add(
        Detection(
            video_id=video.id,
            frame_index=0,
            timestamp=0.0,
            track_id=7,
            class_name="car",
            confidence=0.86,
            x1=0,
            y1=0,
            x2=10,
            y2=10,
        )
    )
    session.commit()
    parsed = StructuredQuery(
        task_type="LEGACY",
        legacy_intent="search_object",
        normalized_query_vi="T\u00ecm \u00f4 t\u00f4",
        objects=[{"type": "car", "attributes": {}}],
        confidence=0.95,
    )

    monkeypatch.setattr(
        gemini_nlp_service,
        "analyze_query",
        lambda query, context=None: GeminiNLPResult(parsed_query=parsed, latency_ms=12.0),
    )

    response = query_routing_service.route_query(session, video.id, "T\u00ecm \u00f4 t\u00f4")

    assert response.nlp_provider == "gemini"
    assert response.debug.intent == "search_object"
    assert response.debug.entities["object"] == "car"
    assert response.segments[0].label == "car track #7"


def test_gemini_object_dog_routes_to_yolo_world(monkeypatch: pytest.MonkeyPatch) -> None:
    session = make_memory_session()
    video = add_completed_video(session)
    parsed = StructuredQuery(
        task_type="LEGACY",
        legacy_intent="search_object",
        normalized_query_vi="T\u00ecm con ch\u00f3",
        objects=[{"type": "dog", "attributes": {}}],
        confidence=0.95,
    )

    def fake_search_video(**kwargs):
        return OpenVocabSearch(
            query_object=kwargs["query_object"],
            detector="yolo_world",
            model="mock-yolo-world",
            confidence=0.25,
            sample_fps=2.0,
            cache_hit=False,
            detections=[OpenVocabDetection("dog", 0.9, [1, 2, 30, 40], 4, 1.0)],
            segments=[OpenVocabSegment("dog", 1.0, 1.0, 1.0, 4, 0.9, [1, 2, 30, 40])],
        )

    monkeypatch.setattr(
        gemini_nlp_service,
        "analyze_query",
        lambda query, context=None: GeminiNLPResult(parsed_query=parsed, latency_ms=12.0),
    )
    monkeypatch.setattr("app.services.query_router.yolo_world_detector.search_video", fake_search_video)

    response = query_routing_service.route_query(session, video.id, "T\u00ecm con ch\u00f3")

    assert response.nlp_provider == "gemini"
    assert response.debug.intent == "open_vocab_search"
    assert response.debug.entities["object"] == "dog"
    assert response.segments[0].detector == "yolo_world"


def test_gemini_long_dog_query_routes_to_yolo_world(monkeypatch: pytest.MonkeyPatch) -> None:
    session = make_memory_session()
    video = add_completed_video(session)
    parsed = StructuredQuery(
        task_type="LEGACY",
        legacy_intent="search_object",
        normalized_query_vi="Cho t\u00f4i xem \u0111o\u1ea1n video c\u00f3 m\u1ed9t ch\u00fa ch\u00f3 xu\u1ea5t hi\u1ec7n",
        objects=[{"type": "dog", "attributes": {}}],
        confidence=0.95,
    )

    monkeypatch.setattr(
        gemini_nlp_service,
        "analyze_query",
        lambda query, context=None: GeminiNLPResult(parsed_query=parsed, latency_ms=12.0),
    )
    monkeypatch.setattr(
        "app.services.query_router.yolo_world_detector.search_video",
        lambda **kwargs: OpenVocabSearch(
            query_object=kwargs["query_object"],
            detector="yolo_world",
            model="mock-yolo-world",
            confidence=0.25,
            sample_fps=2.0,
            cache_hit=False,
            detections=[],
            segments=[],
        ),
    )

    response = query_routing_service.route_query(
        session,
        video.id,
        "Cho t\u00f4i xem \u0111o\u1ea1n video c\u00f3 m\u1ed9t ch\u00fa ch\u00f3 xu\u1ea5t hi\u1ec7n",
    )

    assert response.nlp_provider == "gemini"
    assert response.status == "no_results"
    assert response.debug.entities["detector"] == "yolo_world"


def test_gemini_count_motorcycle_uses_primary_track_count(monkeypatch: pytest.MonkeyPatch) -> None:
    session = make_memory_session()
    video = add_completed_video(session)
    for track_id in [10, 11, 11]:
        session.add(
            Detection(
                video_id=video.id,
                frame_index=track_id,
                timestamp=float(track_id),
                track_id=track_id,
                class_name="motorcycle",
                confidence=0.8,
                x1=0,
                y1=0,
                x2=10,
                y2=10,
            )
        )
    session.commit()
    parsed = StructuredQuery(
        task_type="LEGACY",
        legacy_intent="count_object",
        normalized_query_vi="C\u00f3 bao nhi\u00eau xe m\u00e1y?",
        objects=[{"type": "motorcycle", "attributes": {}}],
        confidence=0.95,
    )
    monkeypatch.setattr(
        gemini_nlp_service,
        "analyze_query",
        lambda query, context=None: GeminiNLPResult(parsed_query=parsed, latency_ms=12.0),
    )

    response = query_routing_service.route_query(session, video.id, "C\u00f3 bao nhi\u00eau xe m\u00e1y?")

    assert response.nlp_provider == "gemini"
    assert response.debug.intent == "count_object"
    assert any("2" in fact for fact in response.raw_facts)


def test_gemini_color_attribute_stays_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    session = make_memory_session()
    video = add_completed_video(session)
    monkeypatch.setattr("app.services.query_router.settings.yolo_world_attribute_prompts", False)
    parsed = StructuredQuery(
        task_type="LEGACY",
        legacy_intent="search_object",
        normalized_query_vi="xe \u00f4 t\u00f4 m\u00e0u v\u00e0ng",
        objects=[{"type": "car", "attributes": {"color": "yellow"}}],
        confidence=0.95,
    )
    monkeypatch.setattr(
        gemini_nlp_service,
        "analyze_query",
        lambda query, context=None: GeminiNLPResult(parsed_query=parsed, latency_ms=12.0),
    )

    response = query_routing_service.route_query(session, video.id, "xe \u00f4 t\u00f4 m\u00e0u v\u00e0ng")

    assert response.status == "unsupported_attribute"
    assert response.nlp_provider == "gemini"


@pytest.mark.parametrize(
    ("query", "object_type", "color", "expected_prompt"),
    [
        ("t\u00ecm xe \u00f4 t\u00f4 m\u00e0u tr\u1eafng", "car", "tr\u1eafng", "white car"),
        ("t\u00ecm xe \u00f4 t\u00f4 m\u00e0u \u0111\u1ecf", "car", "\u0111\u1ecf", "red car"),
        ("t\u00ecm xe bu\u00fdt m\u00e0u tr\u1eafng", "bus", "tr\u1eafng", "white bus"),
    ],
)
def test_gemini_color_attribute_routes_to_yolo_world_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    object_type: str,
    color: str,
    expected_prompt: str,
) -> None:
    session = make_memory_session()
    video = add_completed_video(session)
    monkeypatch.setattr("app.services.query_router.settings.yolo_world_attribute_prompts", True)
    parsed = StructuredQuery(
        task_type="LEGACY",
        legacy_intent="search_object",
        normalized_query_vi=query,
        objects=[{"type": object_type, "attributes": {"color": color}}],
        confidence=0.95,
    )

    def fake_search_video(**kwargs):
        assert kwargs["query_object"] == expected_prompt
        return OpenVocabSearch(
            query_object=kwargs["query_object"],
            detector="yolo_world",
            model="mock-yolo-world",
            confidence=0.25,
            sample_fps=2.0,
            cache_hit=False,
            detections=[OpenVocabDetection(expected_prompt, 0.88, [1, 2, 30, 40], 4, 1.0)],
            segments=[OpenVocabSegment(expected_prompt, 1.0, 1.0, 1.0, 4, 0.88, [1, 2, 30, 40])],
        )

    monkeypatch.setattr(
        gemini_nlp_service,
        "analyze_query",
        lambda _query, context=None: GeminiNLPResult(parsed_query=parsed, latency_ms=12.0),
    )
    monkeypatch.setattr("app.services.query_router.yolo_world_detector.search_video", fake_search_video)

    response = query_routing_service.route_query(session, video.id, query)

    assert response.nlp_provider == "gemini"
    assert response.debug.intent == "open_vocab_search"
    assert response.debug.entities["detector"] == "yolo_world"
    assert response.debug.entities["object"] == expected_prompt
    assert response.debug.entities["yolo_world_prompt"] == expected_prompt
    assert response.segments[0].detector == "yolo_world"
    assert response.segments[0].timestamp == 1.0


def test_gemini_person_clothing_color_remains_unsupported_when_attribute_prompts_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = make_memory_session()
    video = add_completed_video(session)
    monkeypatch.setattr("app.services.query_router.settings.yolo_world_attribute_prompts", True)
    parsed = StructuredQuery(
        task_type="LEGACY",
        legacy_intent="search_object",
        normalized_query_vi="t\u00ecm ng\u01b0\u1eddi \u00e1o xanh",
        objects=[{"type": "person", "attributes": {"upper_color": "blue"}}],
        confidence=0.95,
    )
    monkeypatch.setattr(
        gemini_nlp_service,
        "analyze_query",
        lambda _query, context=None: GeminiNLPResult(parsed_query=parsed, latency_ms=12.0),
    )

    response = query_routing_service.route_query(session, video.id, "t\u00ecm ng\u01b0\u1eddi \u00e1o xanh")

    assert response.status == "unsupported_attribute"
    assert response.nlp_provider == "gemini"


def test_gemini_relationship_stays_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    session = make_memory_session()
    video = add_completed_video(session)
    parsed = StructuredQuery(
        task_type="LEGACY",
        normalized_query_vi="ng\u01b0\u1eddi \u0111ang \u0111i xe m\u00e1y",
        objects=[
            {"type": "person", "attributes": {}},
            {"type": "motorcycle", "attributes": {}},
        ],
        relations=[{"subject": "person", "relation": "riding", "object": "motorcycle"}],
        confidence=0.95,
    )
    monkeypatch.setattr(
        gemini_nlp_service,
        "analyze_query",
        lambda query, context=None: GeminiNLPResult(parsed_query=parsed, latency_ms=12.0),
    )

    response = query_routing_service.route_query(session, video.id, "ng\u01b0\u1eddi \u0111ang \u0111i xe m\u00e1y")

    assert response.status == "unsupported_relationship"
    assert response.nlp_provider == "gemini"


@pytest.mark.parametrize(
    "error",
    [
        GeminiNLPError("timeout"),
        GeminiNLPError("429 quota"),
        GeminiNLPError("invalid json"),
        GeminiNLPError("Gemini API key is not configured."),
    ],
)
def test_gemini_failures_use_local_fallback(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    monkeypatch.setattr(
        gemini_nlp_service,
        "analyze_query",
        lambda query, context=None: (_ for _ in ()).throw(error),
    )

    provider, parsed, fallback_error, latency = query_routing_service.analyze_query_with_fallback("T\u00ecm \u00f4 t\u00f4")

    assert provider == "local_fallback"
    assert parsed.task_type == "LEGACY"
    assert fallback_error
    assert latency is None


def test_query_router_recognizes_unimplemented_future_engines() -> None:
    dummy_query = StructuredQuery(
        task_type="LEGACY",
        normalized_query_vi="test query",
    )
    response = query_routing_service._not_implemented(
        question="test query",
        provider="local",
        parsed=dummy_query,
        engine="future_engine",
    )
    assert response.status == "engine_not_implemented"
    assert response.segments == []


def test_query_endpoint_falls_back_to_local_nlp(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    video = Video(
        original_filename="L00_V000.mp4",
        stored_filename="L00_V000.mp4",
        stored_path="uploads/L00_V000.mp4",
        content_type="video/mp4",
        status="completed",
        duration_seconds=1.0,
        fps=5.0,
        total_frames=5,
        progress_percent=100,
        created_at=datetime.utcnow(),
    )
    session.add(video)
    session.commit()
    session.refresh(video)

    def fake_analyze(query: str, context=None) -> GeminiNLPResult:
        raise GeminiNLPError("mocked Gemini outage")

    def override_db():
        try:
            yield session
        finally:
            pass

    monkeypatch.setattr(gemini_nlp_service, "analyze_query", fake_analyze)
    app.dependency_overrides[get_db] = override_db
    try:
        client = TestClient(app)
        response = client.post("/api/query", json={"video_id": video.id, "question": "Tom tat video"})
    finally:
        app.dependency_overrides.clear()
        session.close()

    assert response.status_code == 200
    payload = response.json()
    assert payload["nlp_provider"] == "local_fallback"
    assert payload["parsed_query"]["task_type"] == "LEGACY"
    assert "nlp_fallback_reason" in payload["debug"]["entities"]


@pytest.mark.skipif(os.getenv("RUN_LIVE_GEMINI_TEST") != "1", reason="Optional live Gemini test is disabled.")
def test_live_gemini_structured_query_contract() -> None:
    result = gemini_nlp_service.analyze_query("Tim canh nguoi mac ao do dung canh xe may")

    assert result.parsed_query.task_type in {"KIS", "LEGACY"}
    assert result.latency_ms > 0


def make_memory_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return testing_session_local()


def add_completed_video(session) -> Video:
    video = Video(
        original_filename="L00_V000.mp4",
        stored_filename="L00_V000.mp4",
        stored_path="uploads/L00_V000.mp4",
        content_type="video/mp4",
        status="completed",
        duration_seconds=10.0,
        fps=5.0,
        total_frames=50,
        progress_percent=100,
        created_at=datetime.utcnow(),
    )
    session.add(video)
    session.commit()
    session.refresh(video)
    return video
