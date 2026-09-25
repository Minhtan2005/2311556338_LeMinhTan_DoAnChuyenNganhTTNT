from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from app.services.kis import kis_service


@dataclass(frozen=True)
class KISBenchmarkMetrics:
    labels_count: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    average_latency_ms: float


def run_kis_benchmark(db: Session, fixture_path: str | Path) -> KISBenchmarkMetrics:
    labels = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    if not labels:
        return KISBenchmarkMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []

    for label in labels:
        started = time.perf_counter()
        results = kis_service.search(db, str(label["query"]), top_k=10)
        latencies.append((time.perf_counter() - started) * 1000)

        rank = None
        for index, result in enumerate(results, start=1):
            if _matches_label(result, label):
                rank = index
                break
        if rank is not None:
            if rank <= 1:
                hits_at_1 += 1
            if rank <= 5:
                hits_at_5 += 1
            if rank <= 10:
                hits_at_10 += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    total = len(labels)
    return KISBenchmarkMetrics(
        labels_count=total,
        recall_at_1=hits_at_1 / total,
        recall_at_5=hits_at_5 / total,
        recall_at_10=hits_at_10 / total,
        mrr=sum(reciprocal_ranks) / total,
        average_latency_ms=sum(latencies) / total,
    )


def _matches_label(result, label: dict) -> bool:
    expected_video = str(label["expected_video"])
    start = int(label["acceptable_frame_start"])
    end = int(label["acceptable_frame_end"])
    return result.video_name == expected_video and start <= int(result.frame_id) <= end
