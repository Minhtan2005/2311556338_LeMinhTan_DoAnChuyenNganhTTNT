from __future__ import annotations

import re
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import Detection
from app.schemas import (
    QueryDebug,
    QueryResponse,
    StructuredEvent,
    StructuredObject,
    StructuredQuery,
)
from app.services.gemini_nlp import GeminiNLPError, gemini_nlp_service
from app.services.kis import kis_service, normalize_color_name, unsupported_query
from app.services.nlp import QueryUnderstanding, query_parser
from app.services.open_vocab_query import (
    PRIMARY_CLASSES,
    looks_like_object_query,
    normalize_for_object_query,
    route_object_query,
)
from app.services.openai_response import natural_answer_service
from app.services.retrieval import retrieval_service
from app.services.yolo_world_detector import yolo_world_detector


SUPPORTED_ATTRIBUTE_PROMPT_COLORS = {"white", "black", "red", "blue", "green", "yellow", "gray", "orange"}
ATTRIBUTE_COLOR_ALIASES: dict[str, tuple[str, ...]] = {
    "white": ("white", "mau trang", "trang"),
    "black": ("black", "mau den", "den"),
    "red": ("red", "mau do", "do"),
    "blue": ("blue", "mau xanh duong", "xanh duong"),
    "green": ("green", "mau xanh la", "xanh la"),
    "yellow": ("yellow", "mau vang", "vang"),
    "gray": ("gray", "grey", "mau xam", "xam"),
    "orange": ("orange", "mau cam", "cam"),
}


class QueryRoutingService:
    def analyze_query_with_fallback(self, query: str) -> tuple[str, StructuredQuery, str | None, float | None]:
        try:
            result = gemini_nlp_service.analyze_query(query)
            return "gemini", result.parsed_query, None, result.latency_ms
        except (GeminiNLPError, Exception) as exc:
            return "local_fallback", self._local_structured_query(query), str(exc), None

    def route_query(self, db: Session, video_id: str | None, question: str) -> QueryResponse:
        provider, parsed, error, _latency_ms = self.analyze_query_with_fallback(question)

        structured_object_response = self._route_structured_object_query(db, video_id or "", question, provider, parsed)
        if structured_object_response is not None:
            if error:
                structured_object_response.debug.entities["nlp_fallback_reason"] = error
            return structured_object_response

        if parsed.task_type == "KIS":
            return self._route_kis(db, question, provider, parsed)
        if parsed.task_type == "Q&A":
            return self._route_qa(db, video_id, question, provider, parsed)
        if parsed.task_type == "TRAKE":
            return self._route_trake(db, video_id, question, provider, parsed)
        if provider == "local_fallback" or parsed.task_type == "LEGACY":
            return self._route_legacy(db, video_id or "", question, provider, parsed, error)

        return self._route_legacy(db, video_id or "", question, provider, parsed, error)

    def _route_structured_object_query(
        self,
        db: Session,
        video_id: str,
        question: str,
        provider: str,
        parsed: StructuredQuery,
    ) -> QueryResponse | None:
        if parsed.task_type == "TRAKE" or parsed.events:
            return None

        attribute_prompt = self._attribute_prompt(parsed) or self._attribute_prompt_from_query(question)
        if attribute_prompt is not None:
            if settings.yolo_world_attribute_prompts:
                return self._route_open_vocab(
                    db,
                    video_id,
                    question,
                    provider,
                    parsed,
                    attribute_prompt,
                    debug_extra={"yolo_world_prompt": attribute_prompt},
                )
            return self._unsupported_response(
                question,
                provider,
                parsed,
                "unsupported_attribute",
                "Chua du du lieu de xac minh thuoc tinh mau.",
                "color_attribute_requires_reliable_visual_verification",
            )

        if not parsed.objects:
            return None

        object_name = str(parsed.objects[0].type).strip()
        if not object_name:
            return None

        if parsed.task_type == "KIS" and not self._is_simple_kis_object_query(question, parsed):
            return None

        if self._parsed_has_attribute_constraint(parsed):
            return self._unsupported_response(
                question,
                provider,
                parsed,
                "unsupported_attribute",
                "Chua du du lieu de xac minh thuoc tinh mau.",
                "color_attribute_requires_reliable_visual_verification",
            )

        if self._parsed_has_relationship_constraint(parsed):
            return self._unsupported_response(
                question,
                provider,
                parsed,
                "unsupported_relationship",
                "Truy van quan he phuc tap chua duoc ho tro day du.",
                "relationship_reasoning_not_enabled",
            )

        if parsed.task_type == "KIS" and (parsed.actions or parsed.spatial):
            return None
        if parsed.task_type == "Q&A" and not self._structured_asks_count_or_presence(question, parsed):
            return None

        route = route_object_query(question, {"object": object_name})
        if route.status == "unsupported_query":
            route = route_object_query(object_name, {"object": object_name})

        if route.status in {"unsupported_attribute", "unsupported_relationship"}:
            return self._unsupported_response(question, provider, parsed, route.status, route.message, route.reason)
        if route.detector == "yolo_world" and route.object_name:
            return self._route_open_vocab(db, video_id, question, provider, parsed, route.object_name)
        if route.detector == "primary" and route.object_name:
            understanding = self._understanding_from_structured_query(question, parsed, route.object_name)
            return self._route_primary_metadata(db, video_id, question, provider, parsed, understanding, error=None)

        return None

    def _route_primary_metadata(
        self,
        db: Session,
        video_id: str,
        question: str,
        provider: str,
        parsed: StructuredQuery,
        understanding: QueryUnderstanding,
        error: str | None,
    ) -> QueryResponse:
        segments, facts = retrieval_service.retrieve(db, video_id, question, understanding)
        answer = natural_answer_service.generate(
            question=question,
            understanding=understanding,
            facts=facts,
            segments=segments,
        )
        entities: dict[str, Any] = dict(understanding.entities)
        if error:
            entities["nlp_fallback_reason"] = error
        return QueryResponse(
            answer=answer,
            debug=QueryDebug(intent=understanding.intent, entities=entities),
            segments=segments,
            raw_facts=facts,
            query=question,
            nlp_provider=provider,
            task_type=parsed.task_type,
            parsed_query=parsed,
        )

    @staticmethod
    def _understanding_from_structured_query(
        question: str,
        parsed: StructuredQuery,
        object_name: str,
    ) -> QueryUnderstanding:
        q_norm = QueryRoutingService._normalize_for_match(question)
        legacy_intent = parsed.legacy_intent or ""
        asks_count = any(token in q_norm for token in ["bao nhieu", "so luong", "dem", "how many"])
        asks_timestamp = any(token in q_norm for token in ["khi nao", "luc nao", "thoi diem", "timestamp"])

        if legacy_intent in {"count_object", "count", "existence", "search_object", "find_timestamp"}:
            intent = {
                "count": "count_object",
                "existence": "search_object",
            }.get(legacy_intent, legacy_intent)
        elif asks_count:
            intent = "count_object"
        elif asks_timestamp:
            intent = "find_timestamp"
        else:
            intent = "search_object"

        entities: dict[str, Any] = {"object": object_name}
        first_object = parsed.objects[0] if parsed.objects else None
        if first_object is not None:
            for key, value in first_object.attributes.items():
                if value is not None:
                    entities[key] = value
            if object_name in PRIMARY_CLASSES - {"person"}:
                entities["vehicle_type"] = object_name
        if parsed.actions:
            entities["action"] = parsed.actions[0]
        return QueryUnderstanding(intent=intent, entities=entities)

    @staticmethod
    def _parsed_has_attribute_constraint(parsed: StructuredQuery) -> bool:
        color_keys = {"color", "vehicle_color", "upper_color", "lower_color"}
        for item in parsed.objects:
            if any(item.attributes.get(key) for key in color_keys):
                return True
        return False

    @staticmethod
    def _attribute_prompt(parsed: StructuredQuery) -> str | None:
        if parsed.relations or parsed.actions or parsed.spatial:
            return None
        if len(parsed.objects) != 1:
            return None

        item = parsed.objects[0]
        object_name = str(item.type).strip().lower()
        if not object_name or object_name == "person":
            return None
        if item.attributes.get("upper_color") or item.attributes.get("lower_color"):
            return None

        raw_color = item.attributes.get("vehicle_color") or item.attributes.get("color")
        if raw_color is None:
            return None
        color = normalize_color_name(str(raw_color))
        if color not in SUPPORTED_ATTRIBUTE_PROMPT_COLORS:
            return None

        route = route_object_query(object_name, {"object": object_name})
        if route.status != "ok" or not route.object_name:
            return None
        return f"{color} {route.object_name}"

    @staticmethod
    def _attribute_prompt_from_query(question: str) -> str | None:
        normalized = normalize_for_object_query(question)
        if not normalized:
            return None

        color = None
        stripped = f" {normalized} "
        for canonical, aliases in ATTRIBUTE_COLOR_ALIASES.items():
            for alias in aliases:
                normalized_alias = normalize_for_object_query(alias)
                if f" {normalized_alias} " in stripped:
                    color = canonical
                    stripped = stripped.replace(f" {normalized_alias} ", " ")
                    break
            if color is not None:
                break
        if color is None:
            return None

        object_query = re.sub(r"\b(mau|color|colored|co|khong)\b", " ", stripped)
        object_query = re.sub(r"\s+", " ", object_query).strip()
        route = route_object_query(object_query)
        if route.status != "ok" or not route.object_name or route.object_name == "person":
            return None
        return f"{color} {route.object_name}"

    @staticmethod
    def _parsed_has_relationship_constraint(parsed: StructuredQuery) -> bool:
        if parsed.relations:
            return True
        relation_actions = {"riding", "person on motorcycle", "person riding motorcycle"}
        if any(str(action).lower() in relation_actions for action in parsed.actions):
            object_types = {item.type for item in parsed.objects}
            return "person" in object_types and bool(object_types.intersection(PRIMARY_CLASSES - {"person"}))
        return False

    @staticmethod
    def _is_simple_kis_object_query(question: str, parsed: StructuredQuery) -> bool:
        return (
            len(parsed.objects) == 1
            and not parsed.objects[0].attributes
            and not parsed.actions
            and not parsed.relations
            and not parsed.spatial
            and looks_like_object_query(question)
        )

    @staticmethod
    def _structured_asks_count_or_presence(question: str, parsed: StructuredQuery) -> bool:
        q_norm = QueryRoutingService._normalize_for_match(question)
        if parsed.legacy_intent in {"count_object", "count", "existence", "search_object", "find_timestamp"}:
            return True
        return (
            any(token in q_norm for token in ["bao nhieu", "so luong", "dem", "how many"])
            or any(token in q_norm for token in [" co ", "khong", "co thay", "xuat hien"])
            or q_norm.startswith("co ")
        )

    def _route_legacy(
        self,
        db: Session,
        video_id: str,
        question: str,
        provider: str,
        parsed: StructuredQuery,
        error: str | None,
    ) -> QueryResponse:
        understanding = query_parser.parse(question)
        route = route_object_query(question, understanding.entities)
        if route.status in {"unsupported_attribute", "unsupported_relationship"}:
            return self._unsupported_response(question, provider, parsed, route.status, route.message, route.reason)
        if route.status == "unsupported_query" and looks_like_object_query(question):
            return self._unsupported_response(question, provider, parsed, route.status, route.message, route.reason)
        if route.detector == "yolo_world" and route.object_name and looks_like_object_query(question):
            return self._route_open_vocab(db, video_id, question, provider, parsed, route.object_name)
        if route.detector == "primary" and route.object_name:
            entities = dict(understanding.entities)
            entities["object"] = route.object_name
            intent = understanding.intent
            if intent not in {"count_object", "find_timestamp", "search_action"}:
                intent = "search_object"
            understanding = QueryUnderstanding(intent=intent, entities=entities)

        if (
            route.detector == "primary"
            and route.object_name
            and settings.yolo_world_fallback_when_primary_empty
        ):
            segments, _facts = retrieval_service.retrieve(db, video_id, question, understanding)
            if segments:
                return self._route_primary_metadata(db, video_id, question, provider, parsed, understanding, error)
            return self._route_open_vocab(db, video_id, question, provider, parsed, route.object_name)
        return self._route_primary_metadata(db, video_id, question, provider, parsed, understanding, error)

    @staticmethod
    def _route_open_vocab(
        db: Session,
        video_id: str,
        question: str,
        provider: str,
        parsed: StructuredQuery,
        object_name: str,
        debug_extra: dict[str, Any] | None = None,
    ) -> QueryResponse:
        from app.models import Video
        from app.schemas import RetrievedSegment

        if not settings.yolo_world_enabled:
            return QueryRoutingService._unsupported_response(
                question,
                provider,
                parsed,
                "yolo_world_disabled",
                "Phat hien mo rong dang tat trong cau hinh.",
                "feature_flag_disabled",
            )

        video = db.get(Video, video_id) if video_id else None
        if video is None:
            return QueryRoutingService._unsupported_response(
                question,
                provider,
                parsed,
                "video_not_found",
                "Khong tim thay video de chay phat hien mo rong.",
                "missing_video",
            )

        try:
            result = yolo_world_detector.search_video(
                video_id=video.id,
                video_path=video.stored_path,
                query_object=object_name,
            )
        except Exception as exc:
            return QueryResponse(
                answer="Khong the tai hoac chay mo hinh phat hien mo rong.",
                debug=QueryDebug(
                    intent="open_vocab_search",
                    entities={
                        "object": object_name,
                        "detector": "yolo_world",
                        "error": str(exc),
                    },
                ),
                segments=[],
                raw_facts=[],
                query=question,
                nlp_provider=provider,
                task_type=parsed.task_type,
                parsed_query=parsed,
                status="yolo_world_unavailable",
                model=settings.yolo_world_model,
                competition_preview=[],
            )

        segments = [
            RetrievedSegment(
                start_time=item.start_time,
                end_time=item.end_time,
                label=f"{item.class_name} ({item.detector})",
                confidence=item.confidence,
                evidence=(
                    f"{item.class_name} detected from {item.start_time:.1f}s to "
                    f"{item.end_time:.1f}s; best frame {item.frame_id} at {item.timestamp:.1f}s."
                ),
                frame_id=item.frame_id,
                timestamp=item.timestamp,
                detector=item.detector,
                class_name=item.class_name,
                bbox=item.bbox,
            )
            for item in result.segments
        ]
        if segments:
            answer = f"Tim thay {len(segments)} doan co {result.query_object} bang YOLO-World."
            status = "ok"
        else:
            answer = f"Khong tim thay {result.query_object} trong video voi YOLO-World."
            status = "no_results"

        return QueryResponse(
            answer=answer,
            debug=QueryDebug(
                intent="open_vocab_search",
                entities={
                    "object": result.query_object,
                    "detector": result.detector,
                    "model": result.model,
                    "device": result.device,
                    "confidence": result.confidence,
                    "sample_fps": result.sample_fps,
                    "cache_hit": result.cache_hit,
                    "raw_detection_count": len(result.detections),
                    **(debug_extra or {}),
                },
            ),
            segments=segments,
            raw_facts=[segment.evidence for segment in segments],
            query=question,
            nlp_provider=provider,
            task_type=parsed.task_type,
            parsed_query=parsed,
            status=status,
            model=result.model,
            model_version="ultralytics-yolo-world",
            competition_preview=[],
        )

    @staticmethod
    def _unsupported_response(
        question: str,
        provider: str,
        parsed: StructuredQuery,
        status: str,
        message: str | None,
        reason: str | None,
    ) -> QueryResponse:
        return QueryResponse(
            answer=message or "Truy van chua duoc ho tro.",
            debug=QueryDebug(intent=status, entities={"reason": reason}),
            segments=[],
            raw_facts=[],
            query=question,
            nlp_provider=provider,
            task_type=parsed.task_type,
            parsed_query=parsed,
            status=status,
            competition_preview=[],
        )

    def _route_kis(self, db: Session, question: str, provider: str, parsed: StructuredQuery) -> QueryResponse:
        semantic_query = parsed.semantic_query_en or parsed.normalized_query_vi or question
        meta = unsupported_query(
            semantic_query,
            structured_query=parsed,
            allow_attribute_verification=True,
        ) or {"status": "ok"}
        if meta.get("status") != "ok":
            results = []
        else:
            results = kis_service.search(db, semantic_query, top_k=20, structured_query=parsed)
        competition_preview = [item.competition_output for item in results[:20]]
        status = meta.get("status") or "ok"
        message = meta.get("message")
        return QueryResponse(
            answer=message or f"Search found {len(results)} matching keyframes.",
            debug=QueryDebug(
                intent="semantic_kis",
                entities={
                    "semantic_query_en": semantic_query,
                    "objects": [item.model_dump() for item in parsed.objects],
                    "actions": parsed.actions,
                    "relations": [item.model_dump() for item in parsed.relations],
                    "reason": meta.get("reason"),
                },
            ),
            segments=[],
            raw_facts=[],
            query=question,
            nlp_provider=provider,
            task_type=parsed.task_type,
            parsed_query=parsed,
            kis_results=[item.model_dump() for item in results],
            status=status,
            model=kis_service.model_name,
            model_version=kis_service.model_version,
            competition_preview=competition_preview,
        )

    def _route_qa(
        self,
        db: Session,
        video_id: str | None,
        question: str,
        provider: str,
        parsed: StructuredQuery,
    ) -> QueryResponse:
        detection_answer = self._route_detection_qa(db, video_id, question, provider, parsed)
        if detection_answer is not None:
            return detection_answer

        route = route_object_query(
            question,
            {"object": parsed.objects[0].type} if parsed.objects else None,
        )
        if route.status in {"unsupported_attribute", "unsupported_relationship"}:
            return self._unsupported_response(question, provider, parsed, route.status, route.message, route.reason)
        if route.detector == "yolo_world" and route.object_name and looks_like_object_query(question):
            return self._route_open_vocab(db, video_id or "", question, provider, parsed, route.object_name)

        from app.services.visual_qa import visual_qa_service

        retrieval_query = parsed.semantic_query_en or parsed.normalized_query_vi or question
        results = kis_service.search(db, retrieval_query, top_k=5, structured_query=parsed)

        target_result = None
        if video_id:
            for r in results:
                if r.video_id == video_id:
                    target_result = r
                    break
        if target_result is None and results:
            target_result = results[0]

        if target_result is None:
            return QueryResponse(
                answer="Không tìm thấy khung hình phù hợp để trả lời câu hỏi.",
                debug=QueryDebug(intent="visual_qa", entities={"error": "no_candidate_keyframes"}),
                segments=[],
                raw_facts=[],
                query=question,
                nlp_provider=provider,
                task_type="Q&A",
                parsed_query=parsed,
                status="no_keyframes",
                competition_preview=[],
            )

        # Visually answer question from the selected frame image
        qa_context = self._qa_detection_context(db, target_result.video_id, target_result.frame_id, target_result.timestamp)
        qa_result = visual_qa_service.answer_question(
            image_path=target_result.image_path,
            question=parsed.question or question,
            context=qa_context,
        )

        competition_row = f"{target_result.video_name},{target_result.frame_id},{qa_result.answer}"

        return QueryResponse(
            answer=qa_result.answer,
            debug=QueryDebug(
                intent="visual_qa",
                entities={
                    "video_id": target_result.video_id,
                    "video_name": target_result.video_name,
                    "frame_id": target_result.frame_id,
                    "image_url": target_result.image_url,
                    "confidence": qa_result.confidence,
                    "model": qa_result.model,
                    "latency_ms": qa_result.latency_ms,
                    "is_insufficient_evidence": qa_result.is_insufficient_evidence,
                    "detections_used": qa_context["detections_used"],
                    "detection_scope": qa_context["detection_scope"],
                },
            ),
            segments=[],
            raw_facts=[qa_result.evidence] if qa_result.evidence else [],
            query=question,
            nlp_provider=provider,
            task_type="Q&A",
            parsed_query=parsed,
            status="ok",
            model=qa_result.model,
            kis_results=[item.model_dump() for item in results],
            competition_preview=[competition_row],
        )

    @staticmethod
    def _route_detection_qa(
        db: Session,
        video_id: str | None,
        question: str,
        provider: str,
        parsed: StructuredQuery,
    ) -> QueryResponse | None:
        if not video_id or not parsed.objects:
            return None

        q_norm = QueryRoutingService._normalize_for_match(question)
        asks_count = any(token in q_norm for token in ["bao nhieu", "so luong", "dem", "how many"])
        asks_presence = any(token in q_norm for token in [" co ", "khong", "co thay", "xuat hien"]) or q_norm.startswith("co ")
        if not asks_count and not asks_presence:
            return None

        object_type = parsed.objects[0].type
        supported = {"person", "car", "motorcycle", "bicycle", "bus", "truck"}
        if object_type not in supported:
            return None

        detections = (
            db.query(Detection)
            .filter(Detection.video_id == video_id, Detection.class_name == object_type)
            .order_by(Detection.timestamp.asc())
            .all()
        )
        track_ids = {int(item.track_id) for item in detections if item.track_id is not None}
        count = len(track_ids) if track_ids else len(detections)
        label = {
            "person": "người",
            "car": "ô tô",
            "motorcycle": "xe máy",
            "bicycle": "xe đạp",
            "bus": "xe buýt",
            "truck": "xe tải",
        }[object_type]

        if asks_count:
            answer = f"Có {count} {label} trong video."
        else:
            answer = f"Có {label} trong video." if count > 0 else f"Không thấy {label} trong video."

        first = detections[0] if detections else None
        last = detections[-1] if detections else None
        segments = []
        if first is not None and last is not None:
            from app.schemas import RetrievedSegment

            segments.append(
                RetrievedSegment(
                    start_time=float(first.timestamp),
                    end_time=float(last.timestamp),
                    label=f"{object_type} detections",
                    confidence=max(float(item.confidence or 0.0) for item in detections),
                    evidence=f"{label} xuất hiện từ giây {first.timestamp:.1f} đến {last.timestamp:.1f}.",
                )
            )

        return QueryResponse(
            answer=answer,
            debug=QueryDebug(
                intent="detection_qa",
                entities={
                    "object": object_type,
                    "unique_track_count": len(track_ids),
                    "detection_count": len(detections),
                    "count_method": "unique_track_id" if track_ids else "detections",
                },
            ),
            segments=segments,
            raw_facts=[answer],
            query=question,
            nlp_provider=provider,
            task_type="Q&A",
            parsed_query=parsed,
            status="ok",
            model="yolo_detection_tracking",
            competition_preview=[],
        )

    @staticmethod
    def _qa_detection_context(db: Session, video_id: str, frame_id: int, timestamp: float) -> dict[str, Any]:
        detections = (
            db.query(Detection)
            .filter(Detection.video_id == video_id, Detection.frame_index == frame_id)
            .order_by(Detection.confidence.desc())
            .all()
        )
        detection_scope = "frame"

        if not detections:
            window = max(float(settings.keyframe_interval_seconds) / 2.0, 0.5)
            detections = (
                db.query(Detection)
                .filter(
                    Detection.video_id == video_id,
                    Detection.timestamp >= max(float(timestamp) - window, 0.0),
                    Detection.timestamp <= float(timestamp) + window,
                )
                .order_by(Detection.confidence.desc())
                .all()
            )
            detection_scope = f"timestamp_window_{window:.2f}s"

        video_detections = (
            db.query(Detection)
            .filter(Detection.video_id == video_id)
            .order_by(Detection.timestamp.asc())
            .all()
        )

        return {
            "detections": detections,
            "video_detections": video_detections,
            "detections_used": len(detections),
            "video_detections_used": len(video_detections),
            "detection_scope": detection_scope,
            "frame_id": frame_id,
            "timestamp": timestamp,
            "video_id": video_id,
        }

    def _route_trake(
        self,
        db: Session,
        video_id: str | None,
        question: str,
        provider: str,
        parsed: StructuredQuery,
    ) -> QueryResponse:
        from app.services.trake import trake_service

        events = parsed.events
        if not events and parsed.actions:
            # Reconstruct basic events if Gemini provided actions instead of events
            events = [
                StructuredEvent(
                    order=idx + 1,
                    description_vi=act,
                    semantic_query_en=act,
                    objects=parsed.objects,
                    actions=[act],
                    relations=parsed.relations,
                )
                for idx, act in enumerate(parsed.actions)
            ]

        trake_res = trake_service.align_events(
            db=db,
            events=events,
            video_id=video_id,
            query=question,
        )

        comp_preview = [trake_res.competition_output] if trake_res.competition_output else []

        return QueryResponse(
            answer=trake_res.message or f"Aligned {len(trake_res.frame_ids)} frames for {len(events)} events.",
            debug=QueryDebug(
                intent="trake",
                entities={
                    "video_id": trake_res.video_id,
                    "video_name": trake_res.video_name,
                    "frame_ids": trake_res.frame_ids,
                    "sequence_score": trake_res.sequence_score,
                    "alignments": [a.model_dump() for a in trake_res.alignments],
                    "status": trake_res.status,
                },
            ),
            segments=[],
            raw_facts=[a.event_description for a in trake_res.alignments],
            query=question,
            nlp_provider=provider,
            task_type="TRAKE",
            parsed_query=parsed,
            status=trake_res.status,
            model="dynamic_programming_viterbi",
            competition_preview=comp_preview,
        )

    @staticmethod
    def _not_implemented(question: str, provider: str, parsed: StructuredQuery, engine: str) -> QueryResponse:
        label = "Visual Q&A" if engine == "visual_qa" else "TRAKE"
        return QueryResponse(
            answer=f"{label} request recognized, but this engine is not implemented in Phase 3A.",
            debug=QueryDebug(
                intent=engine,
                entities={
                    "question": parsed.question,
                    "events": [item.model_dump() for item in parsed.events],
                },
            ),
            segments=[],
            raw_facts=[],
            query=question,
            nlp_provider=provider,
            task_type=parsed.task_type,
            parsed_query=parsed,
            status="engine_not_implemented",
        )

    @staticmethod
    def _translate_vi_to_en(text: str) -> str:
        t = text.lower().strip()
        t = re.sub(r"^(tìm cảnh|cảnh|video|đoạn video)\s*", "", t)
        replacements = [
            (r"\bxe ô tô\b", "a car"),
            (r"\bô tô\b", "a car"),
            (r"\bxe hơi\b", "a car"),
            (r"\bxe máy\b", "a motorcycle"),
            (r"\bxe buýt\b", "a bus"),
            (r"\bxe tải\b", "a truck"),
            (r"\bngười đi bộ\b", "a pedestrian"),
            (r"\bngười\b", "a person"),
            (r"\bdi chuyển\b", "moving on the road"),
            (r"\bdừng lại\b", "stopping or stopped"),
            (r"\bđi qua\b", "passing by"),
            (r"\brẽ\b", "turning"),
            (r"\bxuất hiện\b", "appearing"),
            (r"\bvào\b", "entering"),
            (r"\bra\b", "exiting"),
            (r"\btrực thăng\b", "a helicopter"),
            (r"\btàu ngầm\b", "a submarine"),
            (r"\btên lửa\b", "a rocket"),
            (r"\bnhảy dù\b", "parachuting"),
        ]
        for pat, repl in replacements:
            t = re.sub(pat, repl, t)
        return t.strip() or text

    @staticmethod
    def _normalize_for_match(text: str) -> str:
        import unicodedata

        normalized = unicodedata.normalize("NFD", text.lower())
        normalized = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
        return f" {' '.join(normalized.split())} "

    @staticmethod
    def _local_structured_query(query: str) -> StructuredQuery:
        understanding: QueryUnderstanding = query_parser.parse(query)
        objects: list[StructuredObject] = []
        object_name = understanding.entities.get("object")
        if object_name:
            attributes: dict[str, Any] = {}
            if understanding.entities.get("color"):
                attributes["color"] = understanding.entities["color"]
            if understanding.entities.get("vehicle_type"):
                attributes["vehicle_type"] = understanding.entities["vehicle_type"]
            objects.append(StructuredObject(type=str(object_name), attributes=attributes))

        actions: list[str] = []
        if understanding.entities.get("action"):
            actions.append(str(understanding.entities["action"]))

        q_lower = query.lower()

        # Check for sequential event conjunctions for TRAKE
        seq_pattern = r"\b(rồi|sau đó|tiếp theo|sau đấy|kế tiếp|then|after that|followed by)\b"
        seq_matches = list(re.finditer(seq_pattern, q_lower, flags=re.IGNORECASE))
        if seq_matches:
            raw_parts = re.split(seq_pattern, query, flags=re.IGNORECASE)
            # Filter out the delimiters themselves
            parts = [p.strip() for p in raw_parts if p.strip() and not re.fullmatch(seq_pattern, p.strip(), flags=re.IGNORECASE)]
            if len(parts) >= 2:
                events = []
                for idx, part in enumerate(parts):
                    p_lower = part.lower()
                    ev_objects = []
                    if any(k in p_lower for k in ["xe ô tô", "ô tô", "xe hơi", "car"]):
                        ev_objects.append(StructuredObject(type="car"))
                    if any(k in p_lower for k in ["xe máy", "mô tô", "motorcycle"]):
                        ev_objects.append(StructuredObject(type="motorcycle"))
                    if any(k in p_lower for k in ["người", "đi bộ", "person", "pedestrian"]):
                        ev_objects.append(StructuredObject(type="person"))
                    if any(k in p_lower for k in ["xe buýt", "bus"]):
                        ev_objects.append(StructuredObject(type="bus"))
                    if any(k in p_lower for k in ["xe tải", "truck"]):
                        ev_objects.append(StructuredObject(type="truck"))
                    if any(k in p_lower for k in ["trực thăng", "helicopter"]):
                        ev_objects.append(StructuredObject(type="helicopter"))
                    if any(k in p_lower for k in ["tàu ngầm", "submarine"]):
                        ev_objects.append(StructuredObject(type="submarine"))
                    if any(k in p_lower for k in ["tên lửa", "rocket"]):
                        ev_objects.append(StructuredObject(type="rocket"))

                    en_text = QueryRoutingService._translate_vi_to_en(part)
                    events.append(
                        StructuredEvent(
                            order=idx + 1,
                            description_vi=part,
                            semantic_query_en=en_text,
                            objects=ev_objects,
                            actions=[],
                            relations=[],
                        )
                    )

                return StructuredQuery(
                    task_type="TRAKE",
                    legacy_intent="trake",
                    normalized_query_vi=query,
                    semantic_query_en=None,
                    objects=objects,
                    actions=actions,
                    relations=[],
                    spatial=[],
                    count=None,
                    question=None,
                    events=events,
                    confidence=0.8,
                )

        is_qa = "?" in query or any(
            kw in q_lower
            for kw in ["gì", "nào", "mấy", "bao nhiêu", "ai", "đâu", "sao", "thế nào", "màu gì", "là gì"]
        )
        task_type = "Q&A" if is_qa else "LEGACY"

        return StructuredQuery(
            task_type=task_type,
            legacy_intent=understanding.intent,
            normalized_query_vi=query,
            semantic_query_en=None,
            objects=objects,
            actions=actions,
            relations=[],
            spatial=[],
            count=None,
            question=query if is_qa else None,
            events=[],
            confidence=0.5 if is_qa else 0.0,
        )


query_routing_service = QueryRoutingService()
