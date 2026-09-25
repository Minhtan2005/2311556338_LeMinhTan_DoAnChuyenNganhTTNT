from __future__ import annotations

import logging
import tempfile
import time
import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import Detection, SemanticEmbedding, Video, VideoKeyframe
from app.schemas import GroundingDetection, KISResult, StructuredQuery
from app.services.nlp import normalize_text
from app.services.semantic_index import semantic_index_service
from app.services.video_names import competition_video_name

logger = logging.getLogger(__name__)


SUPPORTED_DETECTION_OBJECTS = {
    "person",
    "bicycle",
    "motorcycle",
    "car",
    "bus",
    "truck",
    "traffic light",
}

VEHICLE_OBJECTS = {"bicycle", "motorcycle", "car", "bus", "truck"}
KNOWN_COLORS = {"black", "white", "gray", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown"}
UNKNOWN_COLOR = "unknown"
ATTRIBUTE_VERIFICATION_COLORS = ("red", "blue", "green", "white", "black", "yellow", "gray", "orange", "brown")
ATTRIBUTE_VERIFY_MIN_MARGIN = 0.01

OBJECT_ALIASES = {
    "person": ["person", "people", "pedestrian", "man", "woman", "nguoi"],
    "car": ["car", "auto", "automobile", "o to", "oto", "xe hoi", "xe con"],
    "motorcycle": ["motorcycle", "motorbike", "bike", "scooter", "xe may", "mo to"],
    "truck": ["truck", "xe tai"],
    "bus": ["bus", "xe bus", "xe buyt"],
    "bicycle": ["bicycle", "bike", "xe dap"],
    "traffic light": ["traffic light", "den giao thong"],
    "backpack": ["backpack", "bag", "ba lo", "balo", "tui"],
}

COLOR_ALIASES = {
    "black": ["black", "mau den", "den"],
    "white": ["white", "mau trang", "trang"],
    "gray": ["gray", "grey", "mau xam", "xam", "ghi"],
    "red": ["red", "mau do", "ao do", "do"],
    "orange": ["orange", "mau cam", "cam"],
    "yellow": ["yellow", "mau vang", "vang"],
    "green": ["green", "mau xanh la", "xanh la", "mau xanh la cay", "xanh la cay"],
    "blue": ["blue", "mau xanh duong", "xanh duong", "mau xanh", "xanh"],
    "purple": ["purple", "mau tim"],
    "pink": ["pink", "mau hong", "hong"],
    "brown": ["brown", "mau nau", "nau"],
}

RELATION_ALIASES = {
    "near": ["near", "close to", "gan", "o gan"],
    "next_to": ["next to", "beside", "standing next to", "ben canh", "dung canh", "o canh"],
}


@dataclass(frozen=True)
class ColorConstraint:
    object_type: str
    color: str
    part: str


@dataclass(frozen=True)
class RelationConstraint:
    subject: str
    relation: str
    object: str


@dataclass(frozen=True)
class QuerySignals:
    required_objects: set[str]
    soft_objects: set[str]
    color_constraints: tuple[ColorConstraint, ...]
    relation_constraints: tuple[RelationConstraint, ...]
    wants_near: bool


@dataclass(frozen=True)
class ColorEvidence:
    color: str
    confidence: float


@dataclass(frozen=True)
class ConstraintScore:
    object_match: float
    color_match: float
    relation_match: float
    crop_semantic: float
    constraint_status: str
    rerank_score: float


@dataclass(frozen=True)
class AttributeVerification:
    requested_color: str
    grounded_bbox: GroundingDetection
    attribute_scores: dict[str, float]
    predicted_color: str | None
    attribute_verified: bool
    final_rerank_score: float
    debug_crop_path: str | None = None
    debug_crop_url: str | None = None


@dataclass(frozen=True)
class RankedEmbedding:
    embedding: SemanticEmbedding
    keyframe: VideoKeyframe
    video: Video
    semantic_score: float
    constraints: ConstraintScore
    signals: QuerySignals

    @property
    def rerank_score(self) -> float:
        return self.constraints.rerank_score

    @property
    def score(self) -> float:
        return settings.semantic_weight * self.semantic_score + self.constraints.rerank_score


class KISService:
    def __init__(self) -> None:
        self._detections_cache: dict[tuple[str, int, int], list[Detection]] = {}
        self._color_cache: dict[tuple[str, int, int, int, str], ColorEvidence] = {}
        self._crop_vector_cache: dict[tuple[str, int, int, int], np.ndarray] = {}

    @property
    def model_name(self) -> str:
        return semantic_index_service.model_name

    @property
    def model_version(self) -> str:
        return semantic_index_service.model_version

    def search(
        self,
        db: Session,
        query: str,
        top_k: int = 20,
        structured_query: StructuredQuery | None = None,
        enable_grounding_rerank: bool | None = None,
        grounding_top_n: int | None = None,
    ) -> list[KISResult]:
        results, _ = self.search_with_meta(
            db=db,
            query=query,
            top_k=top_k,
            structured_query=structured_query,
            enable_grounding_rerank=enable_grounding_rerank,
            grounding_top_n=grounding_top_n,
        )
        return results

    def search_with_meta(
        self,
        db: Session,
        query: str,
        top_k: int = 20,
        structured_query: StructuredQuery | None = None,
        enable_grounding_rerank: bool | None = None,
        grounding_top_n: int | None = None,
    ) -> tuple[list[KISResult], dict[str, Any]]:
        top_k = max(1, min(int(top_k), 100))
        allow_attribute_verification = bool(
            enable_grounding_rerank
            if enable_grounding_rerank is not None
            else settings.kis_grounding_rerank_enabled
        ) or structured_query is not None
        unsupported = unsupported_query(
            query,
            structured_query=structured_query,
            allow_attribute_verification=allow_attribute_verification,
        )
        if unsupported is not None:
            return [], unsupported

        query_vector = semantic_index_service.encode_text(query)
        if float(np.linalg.norm(query_vector)) <= 1e-9:
            return [], {"status": "empty_query_vector"}

        signals = extract_query_signals(query, structured_query=structured_query)
        rows = (
            db.query(SemanticEmbedding, VideoKeyframe, Video)
            .join(VideoKeyframe, SemanticEmbedding.keyframe_id == VideoKeyframe.id)
            .join(Video, SemanticEmbedding.video_id == Video.id)
            .filter(
                SemanticEmbedding.model_name == semantic_index_service.model_name,
                SemanticEmbedding.model_version == semantic_index_service.model_version,
            )
            .all()
        )
        if not rows:
            return [], {"status": "no_embeddings"}

        semantic_ranked: list[tuple[float, SemanticEmbedding, VideoKeyframe, Video]] = []
        for embedding, keyframe, video in rows:
            vector = semantic_index_service.load_vector(embedding)
            if vector.shape != query_vector.shape:
                continue
            semantic_score = float(np.dot(query_vector, vector))
            semantic_ranked.append((semantic_score, embedding, keyframe, video))

        semantic_ranked.sort(key=lambda item: item[0], reverse=True)
        candidate_pool_size = min(len(semantic_ranked), max(top_k * 12, 120))

        ranked: list[RankedEmbedding] = []
        rejected_unknown_fallback: list[RankedEmbedding] = []
        rejected_constraint_kf_ids: set[int] = set()
        for semantic_score, embedding, keyframe, video in semantic_ranked[:candidate_pool_size]:
            constraints = self._constraint_score(db, keyframe, signals, query_vector)
            if constraints.constraint_status == "reject":
                rejected_constraint_kf_ids.add(keyframe.id)
                continue
            item = RankedEmbedding(
                embedding=embedding,
                keyframe=keyframe,
                video=video,
                semantic_score=semantic_score,
                constraints=constraints,
                signals=signals,
            )
            if constraints.constraint_status == "fallback_unknown_color":
                rejected_unknown_fallback.append(item)
            else:
                ranked.append(item)

        ranked.sort(key=lambda item: item.score, reverse=True)
        rejected_unknown_fallback.sort(key=lambda item: item.score, reverse=True)
        selected = [item for item in ranked + rejected_unknown_fallback if item.score > 1e-9]

        # Determine if grounding rerank is requested and enabled
        do_grounding = (
            enable_grounding_rerank
            if enable_grounding_rerank is not None
            else settings.kis_grounding_rerank_enabled
        )

        # When grounding rerank is enabled, ensure Top-N candidates are populated from CLIP semantic ranking
        target_top_n = grounding_top_n or settings.kis_grounding_top_n
        if do_grounding and len(selected) < target_top_n:
            seen_kf_ids = {item.keyframe.id for item in selected} | rejected_constraint_kf_ids
            for s_score, emb, kf, vid in semantic_ranked:
                if kf.id not in seen_kf_ids:
                    c = self._constraint_score(db, kf, signals, query_vector)
                    if c.constraint_status == "reject":
                        rejected_constraint_kf_ids.add(kf.id)
                        continue
                    selected.append(
                        RankedEmbedding(
                            embedding=emb,
                            keyframe=kf,
                            video=vid,
                            semantic_score=s_score,
                            constraints=c,
                            signals=signals,
                        )
                    )
                    seen_kf_ids.add(kf.id)
                if len(selected) >= target_top_n:
                    break

        grounding_meta: dict[str, Any] = {
            "enabled": do_grounding,
            "execution_mode": "disabled",
            "prompt": None,
            "top_n_evaluated": 0,
            "latency_ms": 0.0,
        }

        if not do_grounding or not selected:
            return [self._to_result(item) for item in selected[:top_k]], grounding_meta

        grounding_prompt = derive_grounding_prompt(query, structured_query=structured_query, signals=signals)
        grounding_meta["prompt"] = grounding_prompt
        if not grounding_prompt:
            grounding_meta["status"] = "skipped_no_prompt"
            return [self._to_result(item) for item in selected[:top_k]], grounding_meta

        # LocateAnything Grounding Service Verification
        try:
            from app.services.grounding import grounding_service

            if not grounding_service.is_loaded:
                grounding_service._ensure_loaded()

            grounding_meta["execution_mode"] = grounding_service.execution_mode
            if grounding_service.execution_mode != "real_model":
                logger.warning(
                    "LocateAnything real model unavailable (%s). Preserving original CLIP ranking.",
                    grounding_service.execution_mode,
                )
                grounding_meta["status"] = "skipped_real_model_unavailable"
                return [self._to_result(item) for item in selected[:top_k]], grounding_meta

            t_start = time.perf_counter()
            top_n = min(len(selected), grounding_top_n or settings.kis_grounding_top_n)
            grounding_meta["top_n_evaluated"] = top_n

            evaluated_results: list[tuple[float, KISResult]] = []
            color_constraint = signals.color_constraints[0] if signals.color_constraints else None

            # 1. Evaluate top_n candidates with LocateAnything
            for item in selected[:top_n]:
                failed = False
                try:
                    detections = grounding_service.locate(
                        image_path=item.keyframe.image_path,
                        prompt=grounding_prompt,
                        top_k=5,
                        confidence_threshold=settings.kis_grounding_confidence_threshold,
                    )
                except Exception as exc:
                    logger.warning("LocateAnything inference failed on keyframe %s: %s. Preserving base score.", item.keyframe.id, exc)
                    detections = []
                    failed = True

                if failed:
                    # On inference failure, preserve the original CLIP score and do not demote
                    res = self._to_result(
                        item=item,
                        grounding_score=None,
                        grounding_status="error",
                        grounding_detections=[],
                        final_score=item.score,
                    )
                    evaluated_results.append((item.score, res))
                elif detections:
                    best_conf = max(d.confidence for d in detections)
                    bonus = settings.kis_grounding_weight * (0.8 + 0.2 * min(len(detections), 3) / 3.0)
                    new_score = item.score + bonus
                    attribute_verification = None
                    if color_constraint is not None:
                        attribute_verification = verify_grounded_color_attribute(
                            image_path=item.keyframe.image_path,
                            detections=detections,
                            constraint=color_constraint,
                            video_id=item.video.id,
                            frame_id=int(item.keyframe.frame_id),
                        )
                        if attribute_verification is not None:
                            requested_score = attribute_verification.attribute_scores.get(color_constraint.color)
                            if attribute_verification.attribute_verified and requested_score is not None:
                                new_score += settings.kis_grounding_weight * max(requested_score, 0.0)
                            elif (
                                attribute_verification.predicted_color != color_constraint.color
                                and not attribute_verification.attribute_verified
                                and attribute_score_margin(attribute_verification.attribute_scores) >= ATTRIBUTE_VERIFY_MIN_MARGIN
                            ):
                                new_score = max(new_score - settings.kis_grounding_demote_penalty * 0.5, 0.001)
                            attribute_verification = AttributeVerification(
                                requested_color=attribute_verification.requested_color,
                                grounded_bbox=attribute_verification.grounded_bbox,
                                attribute_scores=attribute_verification.attribute_scores,
                                predicted_color=attribute_verification.predicted_color,
                                attribute_verified=attribute_verification.attribute_verified,
                                final_rerank_score=float(new_score),
                                debug_crop_path=attribute_verification.debug_crop_path,
                                debug_crop_url=attribute_verification.debug_crop_url,
                            )
                    res = self._to_result(
                        item=item,
                        grounding_score=round(best_conf, 4),
                        grounding_status="verified",
                        grounding_detections=detections,
                        final_score=new_score,
                        attribute_verification=attribute_verification,
                    )
                    evaluated_results.append((new_score, res))
                else:
                    if color_constraint is not None:
                        res = self._to_result(
                            item=item,
                            grounding_score=None,
                            grounding_status="no_grounded_bbox",
                            grounding_detections=[],
                            final_score=item.score,
                        )
                        evaluated_results.append((item.score, res))
                        continue
                    penalty = settings.kis_grounding_demote_penalty
                    new_score = max(item.score * 0.1, round(item.score - penalty, 6), 0.001)
                    res = self._to_result(
                        item=item,
                        grounding_score=0.0,
                        grounding_status="demoted",
                        grounding_detections=[],
                        final_score=new_score,
                    )
                    evaluated_results.append((new_score, res))

            # 2. Unevaluated candidates retain base score
            for item in selected[top_n:]:
                res = self._to_result(
                    item=item,
                    grounding_score=None,
                    grounding_status="not_evaluated",
                    grounding_detections=[],
                    final_score=item.score,
                )
                evaluated_results.append((item.score, res))

            # 3. Re-sort candidates by priority and score
            def sort_key(entry: tuple[float, KISResult]) -> tuple[int, float]:
                score, result = entry
                status_priority = 2 if result.grounding_status == "verified" else (1 if result.grounding_status in {"not_evaluated", "error"} else 0)
                return (status_priority, score)

            evaluated_results.sort(key=sort_key, reverse=True)
            grounding_meta["latency_ms"] = round((time.perf_counter() - t_start) * 1000.0, 2)
            grounding_meta["status"] = "success"

            final_results = [res for _, res in evaluated_results[:top_k]]
            return final_results, grounding_meta

        except Exception as exc:
            logger.error("LocateAnything grounding rerank encountered error: %s. Preserving CLIP ranking.", exc)
            grounding_meta["status"] = f"error: {exc}"
            return [self._to_result(item) for item in selected[:top_k]], grounding_meta

    def _constraint_score(
        self,
        db: Session,
        keyframe: VideoKeyframe,
        signals: QuerySignals,
        query_vector: np.ndarray,
    ) -> ConstraintScore:
        detections = self._detections_for_keyframe(db, keyframe)
        detections_by_class = group_detections_by_class(detections)

        object_match = self._object_match(signals, detections_by_class)
        if signals.required_objects and object_match <= 0.0:
            return ConstraintScore(0.0, 0.0, 0.0, 0.0, "reject", -1.0)

        color_match, color_status = self._color_match(keyframe, signals, detections_by_class)
        if color_status == "wrong":
            return ConstraintScore(object_match, color_match, 0.0, 0.0, "reject", -1.0)

        relation_match = self._relation_match(signals, detections_by_class)
        if relation_match <= 0.0 and (signals.relation_constraints or signals.wants_near):
            return ConstraintScore(object_match, color_match, 0.0, 0.0, "reject", -1.0)

        crop_semantic = self._object_crop_semantic_score(keyframe, signals, detections_by_class, query_vector)
        rerank_score = (
            settings.object_weight * object_match
            + settings.attribute_weight * color_match
            + settings.relation_weight * relation_match
            + settings.kis_object_crop_semantic_weight * crop_semantic
        )
        status = "fallback_unknown_color" if color_status == "unknown" else "ok"
        return ConstraintScore(object_match, color_match, relation_match, crop_semantic, status, float(rerank_score))

    def _detections_for_keyframe(self, db: Session, keyframe: VideoKeyframe) -> list[Detection]:
        cache_key = (keyframe.video_id, keyframe.id, keyframe.frame_id)
        cached = self._detections_cache.get(cache_key)
        if cached is not None:
            return cached
        window = max(0.35, min(settings.keyframe_interval_seconds, 1.0))
        detections = (
            db.query(Detection)
            .filter(
                Detection.video_id == keyframe.video_id,
                Detection.timestamp >= keyframe.timestamp_sec - window,
                Detection.timestamp <= keyframe.timestamp_sec + window,
            )
            .limit(300)
            .all()
        )
        self._detections_cache[cache_key] = detections
        return detections

    @staticmethod
    def _object_match(signals: QuerySignals, detections_by_class: dict[str, list[Detection]]) -> float:
        if not signals.required_objects:
            return 0.0
        matched = sum(1 for object_type in signals.required_objects if detections_by_class.get(object_type))
        if matched < len(signals.required_objects):
            return 0.0
        return matched / max(len(signals.required_objects), 1)

    def _color_match(
        self,
        keyframe: VideoKeyframe,
        signals: QuerySignals,
        detections_by_class: dict[str, list[Detection]],
    ) -> tuple[float, str]:
        if not signals.color_constraints:
            return 0.0, "not_required"

        scores: list[float] = []
        statuses: list[str] = []
        for constraint in signals.color_constraints:
            detections = detections_by_class.get(constraint.object_type, [])
            if not detections:
                return 0.0, "wrong"

            evidence = [self._color_for_detection(keyframe, detection, constraint.part) for detection in detections[:20]]
            if any(item.color == constraint.color for item in evidence):
                best = max(item.confidence for item in evidence if item.color == constraint.color)
                scores.append(max(0.75, best))
                statuses.append("ok")
            elif any(item.color == UNKNOWN_COLOR for item in evidence):
                scores.append(0.35)
                statuses.append("unknown")
            else:
                scores.append(0.0)
                statuses.append("wrong")

        if "wrong" in statuses and "ok" not in statuses:
            return 0.0, "wrong"
        if "unknown" in statuses and "ok" not in statuses:
            return min(scores), "unknown"
        return float(sum(scores) / max(len(scores), 1)), "ok"

    @staticmethod
    def _relation_match(signals: QuerySignals, detections_by_class: dict[str, list[Detection]]) -> float:
        relation_constraints = signals.relation_constraints
        if not relation_constraints and signals.wants_near:
            relation_constraints = (
                RelationConstraint(subject="person", relation="near", object="motorcycle"),
                RelationConstraint(subject="person", relation="near", object="car"),
                RelationConstraint(subject="person", relation="near", object="bicycle"),
                RelationConstraint(subject="person", relation="near", object="bus"),
                RelationConstraint(subject="person", relation="near", object="truck"),
            )

        if not relation_constraints:
            return 0.0

        if signals.wants_near and not signals.relation_constraints:
            return 1.0 if any_relation_matches(relation_constraints, detections_by_class) else 0.0

        matched = sum(1 for relation in relation_constraints if relation_matches(relation, detections_by_class))
        return matched / max(len(relation_constraints), 1)

    def _color_for_detection(self, keyframe: VideoKeyframe, detection: Detection, part: str) -> ColorEvidence:
        cache_key = (keyframe.video_id, keyframe.frame_id, detection.id, int(detection.frame_index), part)
        cached = self._color_cache.get(cache_key)
        if cached is not None:
            return cached

        evidence = color_from_keyframe_crop(keyframe.image_path, detection, part)
        self._color_cache[cache_key] = evidence
        return evidence

    def _object_crop_semantic_score(
        self,
        keyframe: VideoKeyframe,
        signals: QuerySignals,
        detections_by_class: dict[str, list[Detection]],
        query_vector: np.ndarray,
    ) -> float:
        if settings.kis_object_crop_semantic_weight <= 0:
            return 0.0
        object_types = sorted(signals.required_objects or signals.soft_objects)
        if not object_types:
            return 0.0

        scores: list[float] = []
        for object_type in object_types:
            for detection in detections_by_class.get(object_type, [])[:4]:
                vector = self._crop_vector(keyframe, detection)
                if vector is None or vector.shape != query_vector.shape:
                    continue
                scores.append(float(np.dot(query_vector, vector)))
        return max(scores) if scores else 0.0

    def _crop_vector(self, keyframe: VideoKeyframe, detection: Detection) -> np.ndarray | None:
        cache_key = (keyframe.video_id, keyframe.frame_id, detection.id, int(detection.frame_index))
        cached = self._crop_vector_cache.get(cache_key)
        if cached is not None:
            return cached
        crop_path = crop_detection_to_temp_file(keyframe.image_path, detection)
        if crop_path is None:
            return None
        try:
            vector = semantic_index_service.provider.encode_image(crop_path)
        except Exception:
            logger.warning("Skipping crop semantic score for unreadable crop %s", crop_path, exc_info=True)
            return None
        finally:
            try:
                crop_path.unlink(missing_ok=True)
            except OSError:
                pass
        self._crop_vector_cache[cache_key] = vector
        return vector

    def _to_result(
        self,
        item: RankedEmbedding,
        grounding_score: float | None = None,
        grounding_status: str | None = None,
        grounding_detections: list[GroundingDetection] | None = None,
        final_score: float | None = None,
        attribute_verification: AttributeVerification | None = None,
    ) -> KISResult:
        video_name = competition_video_name(item.video)
        image_path = item.keyframe.image_path
        score = final_score if final_score is not None else item.score
        primary_detection = self._primary_detection(item)
        return KISResult(
            video_id=item.video.id,
            video_name=video_name,
            original_filename=item.video.original_filename,
            media_url=f"/media/{Path(item.video.stored_path).name}",
            frame_id=int(item.keyframe.frame_id),
            fps=item.video.fps,
            timestamp=item.keyframe.timestamp_sec,
            semantic_score=round(item.semantic_score, 6),
            rerank_score=round(item.rerank_score, 6),
            score=round(score, 6),
            image_url=self._media_url(image_path),
            image_path=image_path,
            object_name=primary_detection.class_name if primary_detection is not None else None,
            confidence=round(float(primary_detection.confidence), 6) if primary_detection is not None else None,
            track_id=int(primary_detection.track_id) if primary_detection is not None and primary_detection.track_id is not None else None,
            competition_output=f"{video_name},{item.keyframe.frame_id}",
            grounding_score=grounding_score,
            grounding_status=grounding_status,
            grounding_detections=grounding_detections or [],
            requested_color=attribute_verification.requested_color if attribute_verification else None,
            grounded_bbox=attribute_verification.grounded_bbox.bbox if attribute_verification else None,
            attribute_scores=attribute_verification.attribute_scores if attribute_verification else {},
            predicted_color=attribute_verification.predicted_color if attribute_verification else None,
            attribute_verified=attribute_verification.attribute_verified if attribute_verification else None,
            final_rerank_score=round(attribute_verification.final_rerank_score, 6) if attribute_verification else (round(score, 6) if final_score is not None else None),
            debug_crop_path=attribute_verification.debug_crop_path if attribute_verification else None,
            debug_crop_url=attribute_verification.debug_crop_url if attribute_verification else None,
        )

    def _primary_detection(self, item: RankedEmbedding) -> Detection | None:
        cache_key = (item.keyframe.video_id, item.keyframe.id, item.keyframe.frame_id)
        detections = self._detections_cache.get(cache_key, [])
        if not detections:
            return None
        candidates = [
            detection
            for detection in detections
            if detection.class_name in item.signals.required_objects
        ] or detections
        return max(candidates, key=lambda detection: float(detection.confidence or 0.0), default=None)

    @staticmethod
    def _media_url(image_path: str) -> str:
        path = Path(image_path)
        try:
            relative = path.resolve().relative_to(Path(settings.upload_dir).resolve())
            return "/media/" + relative.as_posix()
        except ValueError:
            return "/media/" + path.name


def unsupported_query(
    query: str,
    structured_query: StructuredQuery | None = None,
    *,
    allow_attribute_verification: bool = False,
) -> dict[str, Any] | None:
    signals = extract_query_signals(query, structured_query=structured_query)
    normalized = normalize_for_matching(query)

    has_generic_vehicle_color = bool(colors_from_text(query)) and any(
        phrase_in_text(alias, normalized)
        for alias in ["xe", "phuong tien", "vehicle"]
    )
    if (signals.color_constraints or has_generic_vehicle_color) and not allow_attribute_verification:
        return {
            "status": "unsupported_attribute",
            "message": "Chưa đủ dữ liệu để xác minh thuộc tính màu.",
            "reason": "color_attribute_requires_reliable_visual_verification",
        }

    relationship_phrases = [
        "di xe",
        "cuoi xe",
        "ngoi tren xe",
        "dang di xe",
        "riding",
        "person on motorcycle",
        "person riding motorcycle",
    ]
    has_relationship_phrase = any(phrase in normalized for phrase in relationship_phrases)
    has_person_and_vehicle = "person" in signals.required_objects and bool(signals.required_objects.intersection(VEHICLE_OBJECTS))
    if has_relationship_phrase and has_person_and_vehicle:
        return {
            "status": "unsupported_relationship",
            "message": "Truy vấn quan hệ phức tạp chưa được hỗ trợ đầy đủ.",
            "reason": "relationship_reasoning_not_enabled",
        }

    return None


def derive_grounding_prompt(
    query: str,
    structured_query: StructuredQuery | None = None,
    signals: QuerySignals | None = None,
) -> str | None:
    """
    Derive a concise, high-precision visual grounding prompt suitable for LocateAnything-3B.
    Targets queries such as:
      - 'blue car'
      - 'red car'
      - 'person wearing red shirt'
      - 'motorcycle'
      - 'person next to car'
    """
    if signals is None:
        signals = extract_query_signals(query, structured_query=structured_query)

    # 1. Relation constraints (e.g. person next to car, person next to motorcycle)
    if signals.relation_constraints:
        rel = signals.relation_constraints[0]
        rel_phrase = "next to" if rel.relation in {"next_to", "beside"} else "near"
        return f"{rel.subject} {rel_phrase} {rel.object}"

    # 2. Color / attribute constraints (e.g. blue car, person wearing red shirt)
    if signals.color_constraints:
        c = signals.color_constraints[0]
        if c.object_type == "person":
            if c.part == "upper":
                return f"person wearing {c.color} shirt"
            elif c.part == "lower":
                return f"person wearing {c.color} pants"
            else:
                return f"person wearing {c.color} clothes"
        else:
            return f"{c.color} {c.object_type}"

    # 3. Specific required objects (e.g. motorcycle, car)
    if signals.required_objects:
        objs = sorted(list(signals.required_objects))
        return " and ".join(objs) if len(objs) > 1 else objs[0]

    # 4. Use structured query semantic_query_en if available
    if structured_query and structured_query.semantic_query_en:
        sq = structured_query.semantic_query_en.strip()
        for prefix in ["a ", "an ", "the "]:
            if sq.lower().startswith(prefix):
                sq = sq[len(prefix):].strip()
        if sq:
            return sq

    # 5. Clean text fallback
    clean = normalize_for_matching(query)
    if clean:
        return clean

    return None


def extract_query_signals(query: str, structured_query: StructuredQuery | None = None) -> QuerySignals:
    required_objects: set[str] = set()
    soft_objects: set[str] = set()
    color_constraints: list[ColorConstraint] = []
    relation_constraints: list[RelationConstraint] = []

    if structured_query is not None:
        for item in structured_query.objects:
            object_type = normalize_object_name(item.type)
            if object_type is None:
                continue
            if object_type in SUPPORTED_DETECTION_OBJECTS:
                required_objects.add(object_type)
            else:
                soft_objects.add(object_type)

            color = color_from_attributes(item.attributes)
            if color and object_type in SUPPORTED_DETECTION_OBJECTS:
                color_constraints.append(ColorConstraint(object_type=object_type, color=color, part=color_part_for_object(object_type, item.attributes)))

        for item in structured_query.relations:
            relation = normalize_relation_name(item.relation)
            subject = normalize_object_name(item.subject or "")
            obj = normalize_object_name(item.object or "")
            if relation in {"near", "next_to"} and subject in SUPPORTED_DETECTION_OBJECTS and obj in SUPPORTED_DETECTION_OBJECTS:
                required_objects.update({subject, obj})
                relation_constraints.append(RelationConstraint(subject=subject, relation=relation, object=obj))

    parsed_objects = objects_from_text(query)
    parsed_colors = colors_from_text(query)
    parsed_relations = relations_from_text(query)
    required_objects.update(object_type for object_type in parsed_objects if object_type in SUPPORTED_DETECTION_OBJECTS)
    soft_objects.update(object_type for object_type in parsed_objects if object_type not in SUPPORTED_DETECTION_OBJECTS)

    if parsed_colors and required_objects:
        target_objects = color_target_objects(query, required_objects)
        for object_type in target_objects:
            for color in parsed_colors:
                color_constraints.append(ColorConstraint(object_type=object_type, color=color, part=color_part_from_text(query, object_type)))

    for relation in parsed_relations:
        if "person" in required_objects:
            for object_type in sorted(required_objects.intersection(VEHICLE_OBJECTS)):
                relation_constraints.append(RelationConstraint(subject="person", relation=relation, object=object_type))

    wants_near = bool(parsed_relations or relation_constraints)
    return QuerySignals(
        required_objects=required_objects,
        soft_objects=soft_objects,
        color_constraints=tuple(dedupe_color_constraints(color_constraints)),
        relation_constraints=tuple(dedupe_relation_constraints(relation_constraints)),
        wants_near=wants_near,
    )


def normalize_object_name(value: str) -> str | None:
    normalized = normalize_for_matching(value)
    for canonical, aliases in OBJECT_ALIASES.items():
        if normalized == canonical or normalized in aliases:
            if canonical == "bike":
                return "bicycle"
            return canonical
    if normalized in SUPPORTED_DETECTION_OBJECTS:
        return normalized
    if normalized == "vehicle":
        return None
    return normalized if normalized else None


def normalize_relation_name(value: str) -> str | None:
    normalized = normalize_for_matching(value)
    if normalized in {"next_to", "beside"}:
        return "next_to"
    if normalized in {"near", "close_to"}:
        return "near"
    for canonical, aliases in RELATION_ALIASES.items():
        if normalized == canonical or normalized in aliases:
            return canonical
    return None


def color_from_attributes(attributes: dict[str, Any]) -> str | None:
    for key in ("upper_color", "lower_color", "vehicle_color", "color"):
        value = attributes.get(key)
        if value:
            color = normalize_color_name(str(value))
            if color:
                return color
    return None


def color_part_for_object(object_type: str, attributes: dict[str, Any]) -> str:
    if object_type != "person":
        return "full"
    if attributes.get("lower_color"):
        return "lower"
    return "upper"


def color_part_from_text(query: str, object_type: str) -> str:
    normalized = normalize_for_matching(query)
    if object_type == "person" and any(phrase in normalized for phrase in ["quan", "pants", "bottom", "lower"]):
        return "lower"
    if object_type == "person":
        return "upper"
    return "full"


def objects_from_text(query: str) -> set[str]:
    normalized = normalize_for_matching(query)
    objects = set()
    for canonical, aliases in OBJECT_ALIASES.items():
        if any(phrase_in_text(alias, normalized) for alias in aliases + [canonical]):
            objects.add(canonical)
    return objects


def colors_from_text(query: str) -> set[str]:
    normalized = normalize_for_matching(query)
    colors = set()
    for canonical, aliases in COLOR_ALIASES.items():
        if any(phrase_in_text(alias, normalized) for alias in aliases + [canonical]):
            colors.add(canonical)
    return colors


def relations_from_text(query: str) -> set[str]:
    normalized = normalize_for_matching(query)
    relations = set()
    for canonical, aliases in RELATION_ALIASES.items():
        if any(phrase_in_text(alias, normalized) for alias in aliases + [canonical]):
            relations.add(canonical)
    return relations


def color_target_objects(query: str, required_objects: set[str]) -> set[str]:
    normalized = normalize_for_matching(query)
    if any(phrase in normalized for phrase in ["ao", "shirt", "top", "quan", "pants"]):
        return {"person"} if "person" in required_objects else set()
    vehicles = required_objects.intersection(VEHICLE_OBJECTS)
    if vehicles:
        return vehicles
    return required_objects


def normalize_color_name(value: str) -> str | None:
    normalized = normalize_for_matching(value)
    for canonical, aliases in COLOR_ALIASES.items():
        if normalized == canonical or normalized in aliases:
            return canonical
    return normalized if normalized in KNOWN_COLORS else None


def normalize_for_matching(text: str) -> str:
    lowered = normalize_text(text)
    decomposed = unicodedata.normalize("NFD", lowered)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    stripped = stripped.replace("đ", "d").replace("_", " ")
    return " ".join(stripped.split())


def phrase_in_text(phrase: str, text: str) -> bool:
    phrase = normalize_for_matching(phrase)
    if not phrase:
        return False
    padded = f" {text} "
    return f" {phrase} " in padded


def dedupe_color_constraints(items: list[ColorConstraint]) -> list[ColorConstraint]:
    seen = set()
    result = []
    for item in items:
        key = (item.object_type, item.color, item.part)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def dedupe_relation_constraints(items: list[RelationConstraint]) -> list[RelationConstraint]:
    seen = set()
    result = []
    for item in items:
        key = (item.subject, item.relation, item.object)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def group_detections_by_class(detections: list[Detection]) -> dict[str, list[Detection]]:
    grouped: dict[str, list[Detection]] = {}
    for detection in detections:
        grouped.setdefault(detection.class_name, []).append(detection)
    return grouped


def any_relation_matches(relations: tuple[RelationConstraint, ...], detections_by_class: dict[str, list[Detection]]) -> bool:
    return any(relation_matches(relation, detections_by_class) for relation in relations)


def relation_matches(relation: RelationConstraint, detections_by_class: dict[str, list[Detection]]) -> bool:
    subjects = detections_by_class.get(relation.subject, [])
    objects = detections_by_class.get(relation.object, [])
    if not subjects or not objects:
        return False
    threshold = 2.2 if relation.relation == "near" else 1.4
    return any(bboxes_close(left, right, threshold) for left in subjects[:30] for right in objects[:30])


def bboxes_close(left: Detection, right: Detection, scale: float) -> bool:
    lc = ((left.x1 + left.x2) / 2, (left.y1 + left.y2) / 2)
    rc = ((right.x1 + right.x2) / 2, (right.y1 + right.y2) / 2)
    left_size = max(left.x2 - left.x1, left.y2 - left.y1, 1.0)
    right_size = max(right.x2 - right.x1, right.y2 - right.y1, 1.0)
    return float(np.hypot(lc[0] - rc[0], lc[1] - rc[1])) <= max(left_size, right_size) * scale


def color_from_keyframe_crop(image_path: str, detection: Detection, part: str) -> ColorEvidence:
    try:
        import cv2
    except ImportError:
        return ColorEvidence(UNKNOWN_COLOR, 0.0)

    image = cv2.imread(str(image_path))
    if image is None:
        return ColorEvidence(UNKNOWN_COLOR, 0.0)

    crop = crop_detection(image, detection, part)
    if crop is None or crop.size == 0:
        return ColorEvidence(UNKNOWN_COLOR, 0.0)

    color, confidence = dominant_crop_color(crop)
    if confidence < settings.kis_color_confidence_threshold:
        return ColorEvidence(UNKNOWN_COLOR, confidence)
    return ColorEvidence(color, confidence)


def crop_detection(image, detection: Detection, part: str):
    height, width = image.shape[:2]
    x1 = max(0, min(width - 1, int(round(detection.x1))))
    y1 = max(0, min(height - 1, int(round(detection.y1))))
    x2 = max(0, min(width, int(round(detection.x2))))
    y2 = max(0, min(height, int(round(detection.y2))))
    if x2 <= x1 or y2 <= y1:
        return None

    box_h = y2 - y1
    box_w = x2 - x1
    if part == "upper":
        y2 = y1 + max(1, int(box_h * 0.55))
    elif part == "lower":
        y1 = y1 + max(0, int(box_h * 0.45))

    pad_x = max(0, int(box_w * 0.08))
    pad_y = max(0, int((y2 - y1) * 0.08))
    x1 = min(max(0, x1 + pad_x), width - 1)
    x2 = max(min(width, x2 - pad_x), x1 + 1)
    y1 = min(max(0, y1 + pad_y), height - 1)
    y2 = max(min(height, y2 - pad_y), y1 + 1)
    return image[y1:y2, x1:x2]


def dominant_crop_color(crop) -> tuple[str, float]:
    import cv2

    resized = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    labels: list[str] = []
    for h, s, v in hsv.reshape(-1, 3):
        labels.append(pixel_color(int(h), int(s), int(v)))
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return UNKNOWN_COLOR, 0.0
    color, count = max(counts.items(), key=lambda item: item[1])
    confidence = count / max(len(labels), 1)
    return color, float(confidence)


def pixel_color(h: int, s: int, v: int) -> str:
    if v < 45:
        return "black"
    if s < 28 and v > 205:
        return "white"
    if s < 35:
        return "gray"
    if 10 <= h < 24 and v < 165:
        return "brown"
    if h < 10 or h >= 170:
        return "red"
    if h < 22:
        return "orange"
    if h < 35:
        return "yellow"
    if h < 85:
        return "green"
    if h < 130:
        return "blue"
    if h < 155:
        return "purple"
    if h < 170:
        return "pink"
    return "gray"


def crop_detection_to_temp_file(image_path: str, detection: Detection) -> Path | None:
    try:
        import cv2
    except ImportError:
        return None
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    crop = crop_detection(image, detection, "full")
    if crop is None or crop.size == 0:
        return None
    output = Path(tempfile.gettempdir()) / f"kis_crop_{detection.id}.jpg"
    if not cv2.imwrite(str(output), crop):
        return None
    return output


def verify_grounded_color_attribute(
    image_path: str,
    detections: list[GroundingDetection],
    constraint: ColorConstraint,
    video_id: str,
    frame_id: int,
) -> AttributeVerification | None:
    if constraint.color not in ATTRIBUTE_VERIFICATION_COLORS or not detections:
        return None

    grounded = max(detections, key=lambda item: item.confidence)
    crop_path = crop_grounded_bbox_to_debug_file(
        image_path=image_path,
        detection=grounded,
        part=constraint.part,
        video_id=video_id,
        frame_id=frame_id,
        requested_color=constraint.color,
    )
    if crop_path is None:
        return None

    prompts = [attribute_prompt(constraint.object_type, color, constraint.part) for color in ATTRIBUTE_VERIFICATION_COLORS]
    try:
        image_vector = semantic_index_service.provider.encode_image(crop_path)
        if hasattr(semantic_index_service.provider, "encode_text_batch"):
            text_vectors = semantic_index_service.provider.encode_text_batch(prompts)
        else:
            text_vectors = [semantic_index_service.provider.encode_text(prompt) for prompt in prompts]
    except Exception as exc:
        logger.warning("CLIP attribute verification failed for crop %s: %s", crop_path, exc)
        return None

    scores: dict[str, float] = {}
    for color, text_vector in zip(ATTRIBUTE_VERIFICATION_COLORS, text_vectors):
        if image_vector.shape != text_vector.shape:
            continue
        scores[color] = float(np.dot(image_vector, text_vector))
    if not scores:
        return None

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_color, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else best_score
    margin = best_score - second_score
    attribute_verified = best_color == constraint.color and margin >= ATTRIBUTE_VERIFY_MIN_MARGIN
    predicted_color = best_color if margin >= ATTRIBUTE_VERIFY_MIN_MARGIN else None

    return AttributeVerification(
        requested_color=constraint.color,
        grounded_bbox=deepcopy(grounded),
        attribute_scores={color: round(score, 6) for color, score in scores.items()},
        predicted_color=predicted_color,
        attribute_verified=attribute_verified,
        final_rerank_score=0.0,
        debug_crop_path=str(crop_path),
        debug_crop_url=debug_crop_url(crop_path),
    )


def attribute_prompt(object_type: str, color: str, part: str) -> str:
    if object_type == "person":
        if part == "lower":
            return f"a person wearing {color} pants"
        if part == "upper":
            return f"a person wearing a {color} shirt"
        return f"a person wearing {color} clothes"
    label = "traffic light" if object_type == "traffic light" else object_type
    return f"a {color} {label}"


def crop_grounded_bbox_to_debug_file(
    image_path: str,
    detection: GroundingDetection,
    part: str,
    video_id: str,
    frame_id: int,
    requested_color: str,
) -> Path | None:
    try:
        import cv2
    except ImportError:
        return None

    image = cv2.imread(str(image_path))
    if image is None:
        return None

    crop = crop_grounded_bbox(image, detection, part)
    if crop is None or crop.size == 0:
        return None

    output_dir = Path(settings.upload_dir) / "kis_debug_crops"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_video = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in video_id)[:64]
    output = output_dir / f"{safe_video}_frame_{frame_id}_{requested_color}_{int(time.time() * 1000)}.jpg"
    if not cv2.imwrite(str(output), crop):
        return None
    return output


def crop_grounded_bbox(image, detection: GroundingDetection, part: str):
    height, width = image.shape[:2]
    bbox = detection.bbox
    x1 = max(0, min(width - 1, int(round(bbox.x1 * width))))
    y1 = max(0, min(height - 1, int(round(bbox.y1 * height))))
    x2 = max(0, min(width, int(round(bbox.x2 * width))))
    y2 = max(0, min(height, int(round(bbox.y2 * height))))
    if x2 <= x1 or y2 <= y1:
        return None

    box_h = y2 - y1
    if part == "upper":
        y2 = y1 + max(1, int(box_h * 0.55))
    elif part == "lower":
        y1 = y1 + max(0, int(box_h * 0.45))

    return image[y1:y2, x1:x2]


def debug_crop_url(crop_path: Path) -> str | None:
    try:
        relative = crop_path.resolve().relative_to(Path(settings.upload_dir).resolve())
    except ValueError:
        return None
    return "/media/" + relative.as_posix()


def attribute_score_margin(scores: dict[str, float]) -> float:
    if len(scores) < 2:
        return 0.0
    ranked = sorted(scores.values(), reverse=True)
    return float(ranked[0] - ranked[1])


kis_service = KISService()
