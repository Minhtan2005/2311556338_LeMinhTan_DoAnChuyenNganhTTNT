# Detector Benchmark

This script measures real inference output for YOLOv8n `best.pt` and YOLO-World on the same image folder. It does not invent precision, recall, mAP, or FPS values.

Run from the project root:

```powershell
cd D:\doanv2
backend\.venv\Scripts\python.exe scripts\benchmark_detectors.py --images backend\uploads\keyframes --output backend\data\detector_benchmark.csv --classes car,dog,ambulance
```

CSV columns:

- `image`
- `query_class`
- `detector`
- `detections`
- `confidence`
- `inference_ms`

If a YOLO-format ground-truth set is available, use Ultralytics validation tools for mAP rather than filling metric values by hand.
