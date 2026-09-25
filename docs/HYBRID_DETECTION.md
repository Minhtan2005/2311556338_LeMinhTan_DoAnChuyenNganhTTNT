# Hybrid Object Detection

## Summary

The application now keeps the existing YOLOv8n fine-tuned detector as the primary detector and adds YOLO-World as an optional, on-demand open-vocabulary detector.

Normal video analysis is unchanged:

```text
Video -> YOLOv8n best.pt -> ByteTrack -> detections/events -> SQLite -> keyframes/search/Q&A/frontend
```

Open-vocabulary search is separate:

```text
User query -> Gemini/local NLP -> object router -> YOLO-World on demand -> normalized segments -> timestamp search results
```

## Primary Detector

YOLOv8n fine-tuned remains primary.

Model path:

```text
backend/models/best.pt
```

Primary classes:

```text
0 person
1 car
2 motorcycle
3 bicycle
4 bus
5 truck
```

The normal analysis path still uses `VideoAnalyzer` and Ultralytics `model.track(..., tracker="bytetrack.yaml")`, so ByteTrack and SQLite detection persistence remain the stable path for traffic-class detections.

## YOLO-World Detector

Implementation:

```text
backend/app/services/yolo_world_detector.py
```

The service uses the existing `ultralytics` dependency and imports `YOLOWorld` lazily. The default checkpoint is:

```text
yolov8s-worldv2.pt
```

YOLO-World is loaded only for an open-vocabulary query. The model instance is cached in memory and reused for later queries.

Official references:

- Ultralytics documents YOLO-World usage with `YOLOWorld("yolov8s-worldv2.pt")` and dynamic prompts via `set_classes(...)`.
- The official YOLO-World repository metadata declares Apache License 2.0 for the original project.
- This project uses the Ultralytics Python package already present in the environment; Ultralytics licensing should be reviewed before any proprietary/commercial deployment.

Sources:

- https://docs.ultralytics.com/models/yolo-world
- https://github.com/AILab-CVC/YOLO-World/blob/master/pyproject.toml
- https://www.ultralytics.com/license

## Query Routing

Gemini is the primary text query parser when configured. Local NLP remains the automatic fallback. Gemini only parses text into `StructuredQuery`; it does not inspect video content or produce detections.

Fixed vocabulary queries use existing metadata:

```text
"Tìm ô tô"      -> car        -> YOLOv8 metadata
"Tìm xe máy"   -> motorcycle -> YOLOv8 metadata
"Tìm người"    -> person     -> YOLOv8 metadata
```

Open-vocabulary queries call YOLO-World on demand:

```text
"Tìm con chó"          -> dog          -> YOLO-World
"Tìm xe cứu thương"   -> ambulance    -> YOLO-World
"Tìm traffic cone"    -> traffic cone -> YOLO-World
"Tìm cái ô"           -> umbrella     -> YOLO-World
```

The router is implemented in:

```text
backend/app/services/open_vocab_query.py
backend/app/services/query_router.py
backend/app/services/gemini_nlp.py
```

## Performance

YOLO-World is not run during normal analysis. On-demand search samples video frames using:

```text
YOLO_WORLD_SAMPLE_FPS=2
```

For a 60-second video, this scans about 120 frames instead of every frame. Nearby detections are grouped into temporal segments so the UI does not show many duplicate cards for the same object appearance.

## Cache

Cache is file-based JSON under:

```text
backend/data/yolo_world_cache
```

The cache key includes:

```text
video_id
normalized_object_query
model
confidence
sample_fps
```

Corrupt cache files are ignored and regenerated.

## Configuration

```env
YOLO_WORLD_ENABLED=true
YOLO_WORLD_MODEL=yolov8s-worldv2.pt
YOLO_WORLD_CONF=0.25
YOLO_WORLD_SAMPLE_FPS=2
YOLO_WORLD_CACHE=true
YOLO_WORLD_CACHE_DIR=./data/yolo_world_cache
YOLO_WORLD_CLIP_CACHE_HOME=./data
YOLO_WORLD_ATTRIBUTE_PROMPTS=false
YOLO_WORLD_FALLBACK_WHEN_PRIMARY_EMPTY=false
YOLO_WORLD_MERGE_GAP_SECONDS=1.5
```

If `YOLO_WORLD_ENABLED=false`, the app remains usable with the existing YOLOv8/ByteTrack/database/search/Q&A pipeline. Open-vocabulary queries return a clear disabled message instead of crashing.

## Attribute And Relationship Limits

Color attributes remain disabled by default:

```text
"xe ô tô màu vàng" -> unsupported_attribute
```

Relationship/action queries are not claimed as solved by YOLO-World:

```text
"người đang đi xe máy" -> unsupported_relationship
```

YOLO-World is used to extend object vocabulary, not to guarantee color verification, relationship reasoning, or action recognition.

## Smoke Test

Optional real model smoke test:

```powershell
cd D:\doanv2\backend
.\.venv\Scripts\python.exe scripts\smoke_yolo_world.py --image uploads\keyframes\<video_id>\frame_00000000.jpg --classes car,dog,ambulance
```

The smoke test passes if the model loads, inference completes, and any returned detections have valid fields. It is acceptable for a test image to return zero detections.

## Known Limitations

- Open-vocabulary detection is subject to model capability and confidence.
- Sampling can miss objects that appear only briefly.
- Performance depends on CPU/GPU, model size, and video length.
- YOLO-World failures do not stop primary YOLOv8 analysis.
- No fake benchmark numbers are recorded; use `scripts/benchmark_detectors.py` to measure real timings.
