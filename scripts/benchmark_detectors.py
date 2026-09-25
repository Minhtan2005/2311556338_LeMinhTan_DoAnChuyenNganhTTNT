from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.services.open_vocab_query import sanitize_prompt  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLOv8n and YOLO-World on the same image folder.")
    parser.add_argument("--images", required=True, help="Folder containing input images.")
    parser.add_argument("--output", default="detector_benchmark.csv", help="CSV output path.")
    parser.add_argument("--yolov8-model", default=str(BACKEND / "models" / "best.pt"), help="YOLOv8 model path.")
    parser.add_argument("--yolo-world-model", default="yolov8s-worldv2.pt", help="YOLO-World checkpoint.")
    parser.add_argument("--classes", default="car,dog,ambulance", help="Comma-separated query classes.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--detectors", default="yolov8,yolo_world", help="Comma-separated: yolov8,yolo_world.")
    return parser.parse_args()


def iter_images(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def benchmark_yolov8(model, image_path: Path, query_class: str, conf: float) -> tuple[int, float | None, float]:
    started = time.perf_counter()
    result = model.predict(str(image_path), conf=conf, verbose=False)[0]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return 0, None, elapsed_ms
    names = getattr(result, "names", {}) or {}
    cls_values = boxes.cls.cpu().numpy().astype(int).tolist() if boxes.cls is not None else []
    confs = boxes.conf.cpu().numpy().tolist() if boxes.conf is not None else []
    selected = [
        float(confs[index])
        for index, class_id in enumerate(cls_values)
        if names.get(int(class_id)) == query_class and len(confs) > index
    ]
    return len(selected), max(selected) if selected else None, elapsed_ms


def benchmark_yolo_world(model, image_path: Path, query_class: str, conf: float) -> tuple[int, float | None, float]:
    model.set_classes([query_class])
    started = time.perf_counter()
    result = model.predict(str(image_path), conf=conf, verbose=False)[0]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return 0, None, elapsed_ms
    confs = boxes.conf.cpu().numpy().tolist() if boxes.conf is not None else []
    return len(confs), max((float(item) for item in confs), default=None), elapsed_ms


def main() -> int:
    args = parse_args()
    image_folder = Path(args.images)
    images = iter_images(image_folder)
    if not images:
        raise SystemExit(f"No images found in {image_folder}")

    classes = [sanitize_prompt(item) for item in args.classes.split(",") if sanitize_prompt(item)]
    detectors = {item.strip() for item in args.detectors.split(",") if item.strip()}

    from ultralytics import YOLO

    yolo_model = YOLO(args.yolov8_model) if "yolov8" in detectors else None
    yolo_world_model = None
    if "yolo_world" in detectors:
        from ultralytics import YOLOWorld

        yolo_world_model = YOLOWorld(args.yolo_world_model)

    rows: list[dict[str, str | int | float]] = []
    for image_path in images:
        for query_class in classes:
            if yolo_model is not None:
                count, max_conf, ms = benchmark_yolov8(yolo_model, image_path, query_class, args.conf)
                rows.append(
                    {
                        "image": str(image_path),
                        "query_class": query_class,
                        "detector": "yolov8",
                        "detections": count,
                        "confidence": "" if max_conf is None else round(max_conf, 6),
                        "inference_ms": round(ms, 3),
                    }
                )
            if yolo_world_model is not None:
                count, max_conf, ms = benchmark_yolo_world(yolo_world_model, image_path, query_class, args.conf)
                rows.append(
                    {
                        "image": str(image_path),
                        "query_class": query_class,
                        "detector": "yolo_world",
                        "detections": count,
                        "confidence": "" if max_conf is None else round(max_conf, 6),
                        "inference_ms": round(ms, 3),
                    }
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image", "query_class", "detector", "detections", "confidence", "inference_ms"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} benchmark rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
