from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import Detection, SemanticEmbedding, Video, VideoKeyframe
from app.schemas import (
    StructuredEvent,
    TRAKEEventAlignment,
    TRAKEResponse,
)
from app.services.export import CompetitionExportService
from app.services.semantic_index import semantic_index_service

logger = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE_MESSAGE = "Không đủ dữ liệu thị giác để xác định chuỗi sự kiện"


@dataclass
class EventCandidate:
    keyframe: VideoKeyframe
    video: Video
    semantic_score: float
    total_score: float
    detection_match: bool = False
    grounding_score: float | None = None


@dataclass
class VideoSequenceAlignment:
    video: Video
    sequence_score: float
    aligned_keyframes: list[VideoKeyframe]
    aligned_scores: list[float]
    event_alignments: list[TRAKEEventAlignment]


class TRAKEService:
    def __init__(self) -> None:
        self.min_event_similarity: float = 0.18
        self.min_sequence_score: float = 0.20

    def align_events(
        self,
        db: Session,
        events: list[StructuredEvent],
        video_id: str | None = None,
        top_k_videos: int = 5,
        query: str = "",
    ) -> TRAKEResponse:
        """
        Retrieves and aligns an ordered sequence of events across video keyframes.
        Enforces strict chronological monotonicity: frame_1 < frame_2 < ... < frame_N.
        Guarantees all frames originate from the single best matching video.
        """
        t0 = time.perf_counter()
        m_events = len(events)

        if m_events < 2:
            return TRAKEResponse(
                query=query,
                status="invalid_query",
                message="TRAKE requires at least 2 sequential events.",
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        # 1. Fetch candidate videos
        video_query = db.query(Video).filter(Video.status == "completed")
        if video_id:
            video_query = video_query.filter(Video.id == video_id)
        candidate_videos: list[Video] = video_query.all()

        if not candidate_videos:
            return TRAKEResponse(
                query=query,
                status="no_videos",
                message="Không tìm thấy video nào đã xử lý trong hệ thống.",
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        # 2. Encode event semantic queries
        event_embeddings: list[np.ndarray] = []
        for ev in events:
            text = ev.semantic_query_en or ev.description_vi or ""
            vec = semantic_index_service.encode_text(text)
            event_embeddings.append(vec)

        # 3. Align sequence per candidate video using Dynamic Programming
        video_alignments: list[VideoSequenceAlignment] = []

        for video in candidate_videos:
            # Fetch all keyframes and embeddings for this video sorted by frame_id
            rows = (
                db.query(VideoKeyframe, SemanticEmbedding)
                .join(SemanticEmbedding, SemanticEmbedding.keyframe_id == VideoKeyframe.id)
                .filter(VideoKeyframe.video_id == video.id)
                .order_by(VideoKeyframe.frame_id.asc())
                .all()
            )

            if len(rows) < m_events:
                # Video doesn't have enough keyframes to satisfy M distinct events
                continue

            keyframes: list[VideoKeyframe] = [r[0] for r in rows]
            embeddings: list[np.ndarray] = [
                semantic_index_service.load_vector(r[1]) for r in rows
            ]
            t_frames = len(keyframes)

            # Compute S[k][j] = similarity of event k to keyframe j
            score_matrix = np.zeros((m_events, t_frames), dtype=np.float32)

            for k in range(m_events):
                q_vec = event_embeddings[k]
                ev = events[k]
                target_objects = {obj.type.lower() for obj in ev.objects if obj.type}
                video_has_target_obj = True
                if target_objects:
                    video_has_target_obj = (
                        db.query(Detection)
                        .filter(
                            Detection.video_id == video.id,
                            Detection.class_name.in_(target_objects),
                        )
                        .first()
                        is not None
                    )

                for j in range(t_frames):
                    kf_vec = embeddings[j]
                    if q_vec.shape == kf_vec.shape:
                        sem_score = float(np.dot(q_vec, kf_vec))
                    else:
                        sem_score = 0.0

                    # Check object detection bonus / penalty from DB
                    det_adjustment = 0.0
                    if target_objects:
                        if not video_has_target_obj:
                            det_adjustment = -0.15
                        else:
                            kf = keyframes[j]
                            has_obj = (
                                db.query(Detection)
                                .filter(
                                    Detection.video_id == video.id,
                                    Detection.frame_index == kf.frame_id,
                                    Detection.class_name.in_(target_objects),
                                )
                                .first()
                                is not None
                            )
                            det_adjustment = 0.05 if has_obj else -0.03

                    score_matrix[k][j] = sem_score + det_adjustment

            # Optional selective LocateAnything verification on top-2 candidate frames
            # only if event specifies fine-grained attribute queries and grounding is enabled
            if settings.locate_anything_enabled:
                self._apply_selective_grounding(score_matrix, events, keyframes)

            # Dynamic Programming Viterbi alignment
            # dp[k][j] = max cumulative score aligning events 0..k ending at keyframe j
            dp = np.full((m_events, t_frames), -np.inf, dtype=np.float32)
            parent = np.full((m_events, t_frames), -1, dtype=np.int32)

            # Base case: event 0
            for j in range(t_frames - m_events + 1):
                if score_matrix[0][j] >= self.min_event_similarity:
                    dp[0][j] = score_matrix[0][j]

            # Recurrence: events 1..m_events-1
            for k in range(1, m_events):
                for j in range(k, t_frames - m_events + k + 1):
                    if score_matrix[k][j] < self.min_event_similarity:
                        continue

                    # Search best predecessor p < j
                    best_prev = -np.inf
                    best_p = -1
                    for p in range(k - 1, j):
                        if dp[k - 1][p] > -np.inf:
                            # Monotonic spacing consistency
                            spacing_bonus = 0.01 * min((j - p), 10)
                            cand = dp[k - 1][p] + score_matrix[k][j] + spacing_bonus
                            if cand > best_prev:
                                best_prev = cand
                                best_p = p

                    if best_p != -1:
                        dp[k][j] = best_prev
                        parent[k][j] = best_p

            # Find best sequence ending at any valid j for last event
            best_final_j = -1
            best_final_score = -np.inf
            for j in range(m_events - 1, t_frames):
                if dp[m_events - 1][j] > best_final_score:
                    best_final_score = dp[m_events - 1][j]
                    best_final_j = j

            if best_final_j != -1 and best_final_score > -np.inf:
                # Backtrack optimal sequence
                aligned_indices: list[int] = [best_final_j]
                curr = best_final_j
                for k in range(m_events - 1, 0, -1):
                    curr = parent[k][curr]
                    aligned_indices.append(curr)
                aligned_indices.reverse()

                # Verify strict monotonicity: j_0 < j_1 < ... < j_{M-1}
                is_monotonic = all(
                    aligned_indices[i] < aligned_indices[i + 1]
                    for i in range(len(aligned_indices) - 1)
                )
                if not is_monotonic:
                    continue

                aligned_kfs = [keyframes[idx] for idx in aligned_indices]
                aligned_scores = [float(score_matrix[k][aligned_indices[k]]) for k in range(m_events)]
                seq_score = float(best_final_score / m_events)

                event_alignments: list[TRAKEEventAlignment] = []
                for k in range(m_events):
                    kf = aligned_kfs[k]
                    ev = events[k]
                    event_alignments.append(
                        TRAKEEventAlignment(
                            order=ev.order,
                            event_description=ev.description_vi or ev.semantic_query_en,
                            frame_id=kf.frame_id,
                            timestamp=kf.timestamp_sec,
                            similarity_score=round(aligned_scores[k], 4),
                            image_url=f"/media/keyframes/{video.id}/frame_{kf.frame_id:08d}.jpg",
                            image_path=kf.image_path,
                        )
                    )

                video_alignments.append(
                    VideoSequenceAlignment(
                        video=video,
                        sequence_score=seq_score,
                        aligned_keyframes=aligned_kfs,
                        aligned_scores=aligned_scores,
                        event_alignments=event_alignments,
                    )
                )

        # 4. Rank candidate videos by sequence score
        video_alignments.sort(key=lambda a: a.sequence_score, reverse=True)

        if not video_alignments or video_alignments[0].sequence_score < self.min_sequence_score:
            logger.info("TRAKE sequence alignment found insufficient evidence across candidate videos.")
            return TRAKEResponse(
                query=query,
                status="insufficient_evidence",
                message=INSUFFICIENT_EVIDENCE_MESSAGE,
                frame_ids=[],
                alignments=[],
                sequence_score=0.0,
                competition_output=None,
                parsed_events=[ev.model_dump() for ev in events],
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        best_alignment = video_alignments[0]
        selected_frames = [kf.frame_id for kf in best_alignment.aligned_keyframes]

        # Format strictly compliant competition row: <video_name>,<f1>,<f2>,...,<fn>
        video_name = CompetitionExportService.sanitize_video_name(best_alignment.video.original_filename)
        comp_output = CompetitionExportService.format_trake_row(video_name, selected_frames)

        return TRAKEResponse(
            query=query,
            video_id=best_alignment.video.id,
            video_name=video_name,
            frame_ids=selected_frames,
            alignments=best_alignment.event_alignments,
            sequence_score=round(best_alignment.sequence_score, 4),
            competition_output=comp_output,
            status="ok",
            message=f"Đã xác định thành công chuỗi {m_events} sự kiện theo thứ tự thời gian.",
            parsed_events=[ev.model_dump() for ev in events],
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )

    def _apply_selective_grounding(
        self,
        score_matrix: np.ndarray,
        events: list[StructuredEvent],
        keyframes: list[VideoKeyframe],
    ) -> None:
        """Run LocateAnything only on top-2 candidate keyframes for events requiring attribute verification."""
        try:
            from app.services.grounding import grounding_service

            if not grounding_service.is_loaded:
                grounding_service._ensure_loaded()
            if grounding_service.execution_mode != "real_model":
                return

            for k, ev in enumerate(events):
                # Only check grounding if event has fine-grained attributes or objects
                has_attr = any(bool(obj.attributes) for obj in ev.objects)
                if not has_attr and not ev.relations:
                    continue

                prompt = ev.semantic_query_en or ev.description_vi
                # Select top-2 candidate frames for this event
                top_indices = np.argsort(score_matrix[k])[-2:]
                for idx in top_indices:
                    kf = keyframes[idx]
                    dets = grounding_service.locate(
                        image_path=kf.image_path,
                        prompt=prompt,
                        top_k=3,
                        confidence_threshold=0.25,
                    )
                    if dets:
                        best_c = max(d.confidence for d in dets)
                        score_matrix[k][idx] += 0.10 * best_c
                    else:
                        score_matrix[k][idx] -= 0.05
        except Exception as exc:
            logger.debug("Selective grounding in TRAKE skipped: %s", exc)


trake_service = TRAKEService()
