from __future__ import annotations

import argparse
import sys
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.services.yolo_world_detector import yolo_world_detector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional YOLO-World smoke test on one image/frame.")
    parser.add_argument("--image", required=True, help="Path to a test image.")
    parser.add_argument("--classes", default="car,dog,ambulance", help="Comma-separated classes.")
    parser.add_argument("--conf", type=float, default=None, help="Override confidence threshold.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is not installed.") from exc

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise SystemExit(f"Cannot read image: {image_path}")

    classes = [item.strip() for item in args.classes.split(",") if item.strip()]
    detections = yolo_world_detector.detect_frame(frame, classes, conf=args.conf)
    for item in detections:
        print(
            {
                "class_name": item.class_name,
                "confidence": round(item.confidence, 4),
                "bbox": [round(value, 2) for value in item.bbox],
                "frame_id": item.frame_id,
                "timestamp": item.timestamp,
                "detector": item.detector,
            }
        )
    print(f"YOLO-World smoke test completed with {len(detections)} detections.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
