from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

from sqlalchemy import func

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.session import SessionLocal
from app.models import Detection, Video
from app.services.kis_benchmark import run_kis_benchmark


DEFAULT_FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "kis_benchmark.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print read-only experiment/evaluation numbers from the current SQLite database.",
    )
    parser.add_argument(
        "--kis-fixture",
        type=Path,
        default=DEFAULT_FIXTURE,
        help="Path to KIS benchmark labels. Defaults to backend/tests/fixtures/kis_benchmark.json.",
    )
    args = parser.parse_args()

    command = "python scripts/evaluate_experiment.py"
    if args.kis_fixture != DEFAULT_FIXTURE:
        command += f" --kis-fixture {args.kis_fixture}"

    started = time.perf_counter()
    db = SessionLocal()
    try:
        processed_videos = db.query(func.count(Video.id)).filter(Video.status == "completed").scalar() or 0
        total_detections = db.query(func.count(Detection.id)).scalar() or 0
        average_confidence = db.query(func.avg(Detection.confidence)).scalar()
        class_counts = (
            db.query(Detection.class_name, func.count(Detection.id))
            .group_by(Detection.class_name)
            .order_by(func.count(Detection.id).desc(), Detection.class_name.asc())
            .all()
        )
        unique_tracks = count_unique_tracks(db)

        print("AI Video Assistant Experiment Report")
        print("====================================")
        print(f"Command: {command}")
        print(f"Processed videos: {processed_videos}")
        print(f"Total detections: {total_detections}")
        print(f"Unique tracks: {unique_tracks}")
        print(f"Average detection confidence: {format_float(average_confidence)}")
        print("Detections by class:")
        if class_counts:
            for class_name, count in class_counts:
                print(f"  - {class_name}: {count}")
        else:
            print("  - none")

        print("KIS benchmark:")
        labels_count = benchmark_label_count(args.kis_fixture)
        if labels_count <= 0:
            print(f"  Fixture: {args.kis_fixture}")
            print("  Labels: 0")
            print("  Recall@1: n/a")
            print("  Recall@5: n/a")
            print("  Recall@10: n/a")
            print("  Average KIS query latency: n/a")
        else:
            metrics = run_kis_benchmark(db, args.kis_fixture)
            print(f"  Fixture: {args.kis_fixture}")
            print(f"  Labels: {metrics.labels_count}")
            print(f"  Recall@1: {metrics.recall_at_1:.4f}")
            print(f"  Recall@5: {metrics.recall_at_5:.4f}")
            print(f"  Recall@10: {metrics.recall_at_10:.4f}")
            print(f"  Average KIS query latency: {metrics.average_latency_ms:.2f} ms")

        elapsed_ms = (time.perf_counter() - started) * 1000
        print(f"Report generation latency: {elapsed_ms:.2f} ms")
    finally:
        db.close()


def count_unique_tracks(db) -> int:
    rows = (
        db.query(Detection.video_id, Detection.class_name, Detection.track_id)
        .filter(Detection.track_id.isnot(None))
        .distinct()
        .all()
    )
    return len(rows)


def benchmark_label_count(path: Path) -> int:
    if not path.exists():
        return 0
    labels = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(labels, list):
        raise ValueError(f"KIS benchmark fixture must be a list: {path}")
    return len(labels)


def format_float(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
