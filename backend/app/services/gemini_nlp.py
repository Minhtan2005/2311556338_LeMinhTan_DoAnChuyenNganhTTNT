from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Literal

from app.core.config import settings
from app.schemas import StructuredEvent, StructuredObject, StructuredQuery, StructuredRelation
from pydantic import BaseModel, Field


GEMINI_SYSTEM_PROMPT = """You are a query parser for an AI video retrieval system.

Your task is NOT to answer the user's question directly.
Your task is NOT to invent video names, timestamps, frame IDs, detections, scores, or SQL.
Do not claim an object exists in a video.
Do not analyze image, frame, or video content.

Convert the user's Vietnamese or English query into structured JSON for downstream video retrieval.
Classify the request into exactly one task_type: KIS, Q&A, TRAKE, or LEGACY.

Definitions:
- KIS: the user describes a visible scene/event to find in video frames.
- Q&A: the user asks a question that requires answering something visible in a frame/video.
- TRAKE: the user describes ordered multiple events that must be found chronologically.
- LEGACY: existing object/event/count/timestamp/summary database queries.

Use LEGACY for simple object search, object timestamp search, object counting, and object existence queries.
Use Q&A only when the user asks for visual details that cannot be answered from object detections alone.
For object search/count/existence, extract objects using concise English detector labels.
Examples:
- "Tìm ô tô" -> task_type LEGACY, legacy_intent search_object, object car.
- "Có bao nhiêu xe máy?" -> task_type LEGACY, legacy_intent count_object, object motorcycle.
- "Tìm con chó" -> task_type LEGACY, legacy_intent search_object, object dog.
- "Có xe cứu thương không?" -> task_type LEGACY, legacy_intent search_object, object ambulance.
- "Tìm cái ô" -> task_type LEGACY, legacy_intent search_object, object umbrella.

Extract visible objects, explicit attributes, actions, spatial relations, questions, and ordered events.
When generating semantic_query_en, write a concise literal visual description suitable for CLIP image retrieval.
Do not add facts not present in the query.
Return only data matching the required schema.

Supported object examples: person, bicycle, motorcycle, car, bus, truck, backpack, traffic light, traffic sign.
Supported action examples: standing, walking, running, riding, entering, exiting, carrying, crossing, stopping, turning, sitting.
Supported relation examples: next_to, riding, inside, entering, exiting, carrying, behind, in_front_of, near, crossing.
"""


@dataclass(frozen=True)
class GeminiNLPResult:
    parsed_query: StructuredQuery
    latency_ms: float


class GeminiNLPError(RuntimeError):
    pass


class GeminiObjectAttributes(BaseModel):
    upper_color: str | None = None
    lower_color: str | None = None
    color: str | None = None
    backpack: bool | None = None
    has_backpack: bool | None = None
    gender_description: str | None = None
    clothing_description: str | None = None
    vehicle_type: str | None = None
    vehicle_color: str | None = None
    object_description: str | None = None


class GeminiStructuredObject(BaseModel):
    type: str
    attributes: GeminiObjectAttributes = Field(default_factory=GeminiObjectAttributes)


class GeminiStructuredRelation(BaseModel):
    subject: str | None = None
    relation: str
    object: str | None = None


class GeminiStructuredEvent(BaseModel):
    order: int
    description_vi: str
    semantic_query_en: str
    objects: list[GeminiStructuredObject] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    relations: list[GeminiStructuredRelation] = Field(default_factory=list)


class GeminiStructuredQuery(BaseModel):
    task_type: Literal["KIS", "Q&A", "TRAKE", "LEGACY"]
    legacy_intent: str | None = None
    normalized_query_vi: str
    semantic_query_en: str | None = None
    objects: list[GeminiStructuredObject] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    relations: list[GeminiStructuredRelation] = Field(default_factory=list)
    spatial: list[str] = Field(default_factory=list)
    count: int | None = None
    question: str | None = None
    events: list[GeminiStructuredEvent] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


def _gemini_to_structured_query(parsed: GeminiStructuredQuery) -> StructuredQuery:
    return StructuredQuery(
        task_type=parsed.task_type,
        legacy_intent=parsed.legacy_intent,
        normalized_query_vi=parsed.normalized_query_vi,
        semantic_query_en=parsed.semantic_query_en,
        objects=[
            StructuredObject(
                type=item.type,
                attributes=item.attributes.model_dump(exclude_none=True),
            )
            for item in parsed.objects
        ],
        actions=parsed.actions,
        relations=[
            StructuredRelation(subject=item.subject, relation=item.relation, object=item.object)
            for item in parsed.relations
        ],
        spatial=parsed.spatial,
        count=parsed.count,
        question=parsed.question,
        events=[
            StructuredEvent(
                order=event.order,
                description_vi=event.description_vi,
                semantic_query_en=event.semantic_query_en,
                objects=[
                    StructuredObject(
                        type=item.type,
                        attributes=item.attributes.model_dump(exclude_none=True),
                    )
                    for item in event.objects
                ],
                actions=event.actions,
                relations=[
                    StructuredRelation(subject=item.subject, relation=item.relation, object=item.object)
                    for item in event.relations
                ],
            )
            for event in parsed.events
        ],
        confidence=parsed.confidence,
    )


class GeminiNLPService:
    def __init__(self) -> None:
        self._cache: dict[str, tuple[float, StructuredQuery]] = {}

    def analyze_query(self, query: str, context: dict[str, Any] | None = None) -> GeminiNLPResult:
        normalized = self._normalize(query)
        if not normalized:
            raise GeminiNLPError("Query is empty.")
        if len(normalized) > settings.gemini_max_query_length:
            raise GeminiNLPError("Query exceeds configured Gemini NLP length limit.")
        if not settings.gemini_enabled:
            raise GeminiNLPError("Gemini NLP is disabled.")
        if not settings.gemini_api_key:
            raise GeminiNLPError("Gemini API key is not configured.")

        cached = self._get_cached(normalized)
        if cached is not None:
            return GeminiNLPResult(parsed_query=cached, latency_ms=0.0)

        started = time.perf_counter()
        try:
            raw_text = self._call_gemini(normalized, context=context)
            parsed = self._parse_model_output(raw_text)
        except GeminiNLPError:
            raise
        except Exception as exc:
            raise GeminiNLPError(f"Gemini NLP request failed: {exc.__class__.__name__}") from exc

        latency_ms = (time.perf_counter() - started) * 1000.0
        self._set_cached(normalized, parsed)
        return GeminiNLPResult(parsed_query=parsed, latency_ms=latency_ms)

    def health_check(self) -> dict[str, Any]:
        return {
            "enabled": settings.gemini_enabled,
            "configured": bool(settings.gemini_api_key),
            "model": settings.gemini_model,
            "timeout_seconds": settings.gemini_timeout_seconds,
            "cache_ttl_seconds": settings.gemini_cache_ttl_seconds,
            "max_query_length": settings.gemini_max_query_length,
        }

    def _call_gemini(self, query: str, context: dict[str, Any] | None = None) -> str:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=int(settings.gemini_timeout_seconds * 1000)),
        )
        payload = {
            "query": query,
            "context": context or {},
            "output_contract": "Return a StructuredQuery JSON object only.",
        }
        prompt = json.dumps(payload, ensure_ascii=False)

        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=GEMINI_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=GeminiStructuredQuery,
                temperature=0,
            ),
        )
        if getattr(response, "parsed", None) is not None:
            parsed = response.parsed
            if isinstance(parsed, GeminiStructuredQuery):
                return _gemini_to_structured_query(parsed).model_dump_json()
            if isinstance(parsed, StructuredQuery):
                return parsed.model_dump_json()
            return json.dumps(parsed, ensure_ascii=False)
        text = getattr(response, "text", None)
        if not text:
            raise GeminiNLPError("Gemini returned an empty NLP response.")
        return str(text)

    @staticmethod
    def _parse_model_output(raw_text: str) -> StructuredQuery:
        try:
            parsed = StructuredQuery.model_validate_json(raw_text)
        except Exception as exc:
            raise GeminiNLPError("Gemini returned invalid StructuredQuery JSON.") from exc
        return parsed

    def _get_cached(self, key: str) -> StructuredQuery | None:
        if settings.gemini_cache_ttl_seconds <= 0:
            return None
        cached = self._cache.get(key)
        if cached is None:
            return None
        created_at, parsed = cached
        if time.monotonic() - created_at > settings.gemini_cache_ttl_seconds:
            self._cache.pop(key, None)
            return None
        return parsed.model_copy(deep=True)

    def _set_cached(self, key: str, parsed: StructuredQuery) -> None:
        if settings.gemini_cache_ttl_seconds <= 0:
            return
        self._cache[key] = (time.monotonic(), parsed.model_copy(deep=True))

    @staticmethod
    def _normalize(query: str) -> str:
        return " ".join(query.strip().split())


gemini_nlp_service = GeminiNLPService()
