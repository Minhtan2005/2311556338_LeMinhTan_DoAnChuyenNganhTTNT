# AI Video Analysis V2

Final V2 project for traffic video analysis. The app supports real video upload, YOLOv8n fine-tuned detection, ByteTrack tracking, SQLite persistence, keyframes, object search, Vietnamese Q&A, and click-to-seek from results.

## Actual Architecture

- Backend: FastAPI, SQLAlchemy, SQLite.
- Frontend: React + Vite.
- Detection: Ultralytics YOLOv8n fine-tuned `backend/models/best.pt`.
- Tracking: ByteTrack through `model.track(..., tracker="bytetrack.yaml")`, with centroid fallback only when no track id is returned.
- Database: `backend/data/v2_app.db`.
- Media storage: `backend/uploads`, keyframes in `backend/uploads/keyframes`.
- Search index: `local-semantic-v1`, a lightweight metadata feature vector built from YOLO/keyframe metadata. It is not CLIP and is not claimed as CLIP.
- Q&A: local NLP/detection fallback for required counting and presence questions. Gemini is disabled by default.

## Model

`backend/models/best.pt` loads successfully with 6 classes:

```text
0 person
1 car
2 motorcycle
3 bicycle
4 bus
5 truck
```

The frontend shows the runtime model source as `fine-tuned best.pt`.

## Hybrid Detection Upgrade

The V2 pipeline now supports optional hybrid object detection:

- Gemini can act as the primary text-only NLP query parser when configured; local NLP remains the automatic fallback.
- YOLOv8n fine-tuned `backend/models/best.pt` remains the primary detector for normal analysis.
- YOLO-World is added as an on-demand open-vocabulary detector for object queries outside the 6 primary classes.
- Fixed-class queries such as `Tìm ô tô`, `Tìm xe máy`, and `Tìm người` use existing YOLOv8/SQLite metadata.
- Open-vocabulary queries such as `Tìm con chó`, `Tìm xe cứu thương`, and `Tìm traffic cone` route to YOLO-World only when needed.
- Color attributes and relationship/action queries remain conservative unless explicitly supported.

Configuration:

```env
GEMINI_ENABLED=false
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TIMEOUT_SECONDS=15
GEMINI_VISION_ENABLED=false
YOLO_WORLD_ENABLED=true
YOLO_WORLD_MODEL=yolov8s-worldv2.pt
YOLO_WORLD_CONF=0.25
YOLO_WORLD_SAMPLE_FPS=2
YOLO_WORLD_CACHE=true
YOLO_WORLD_ATTRIBUTE_PROMPTS=false
```

If `YOLO_WORLD_ENABLED=false`, the existing YOLOv8/ByteTrack/database/search/Q&A path continues to run without YOLO-World.
If Gemini is disabled, missing, timed out, invalid, or quota-limited, the existing local NLP fallback continues to run.

Detailed docs:

- `docs/GEMINI_NLP.md`
- `docs/HYBRID_DETECTION.md`
- `docs/THESIS_HYBRID_DETECTION_NOTES.md`
- `scripts/README_BENCHMARK.md`

## What Was Reused

- From `D:\doan`: FastAPI backend, React/Vite frontend, YOLO/ByteTrack analyzer, SQLite models, keyframe/index modules, Q&A/local NLP pieces, and tests.

## Safety Behavior

- Object counting uses unique `track_id` when available, not raw detection count.
- Event timeline deduplicates repeated appear/leave events by object class and timestamp bucket.
- Color query without reliable runtime verification returns `unsupported_attribute` with message: `Chua du du lieu de xac minh thuoc tinh mau.`
- Complex riding/relationship query such as `tim doan co nguoi dang di xe may` returns `unsupported_relationship` instead of returning all person results.

## Commands

Backend:

```bat
cd /d D:\doanv2
run_backend.bat
```

Frontend:

```bat
cd /d D:\doanv2
run_frontend.bat
```

If drive C is full, run frontend build with temp/cache on D:

```powershell
cd D:\doanv2\frontend
$env:TEMP='D:\doanv2\.tmp'
$env:TMP='D:\doanv2\.tmp'
$env:npm_config_cache='D:\doanv2\.npm-cache'
npm run build --cache D:\doanv2\.npm-cache
```

URLs:

- Backend: http://127.0.0.1:8000
- API docs: http://127.0.0.1:8000/docs
- Frontend: http://127.0.0.1:5173

## Verified Final Test

Sample video: `integration_bestpt_6da0fbd1.mp4`, uploaded through the API as a real upload.

- Backend tests: `74 passed, 1 skipped`.
- Frontend build: passed with npm temp/cache on D because drive C was full.
- Upload: passed.
- Analysis: completed.
- YOLO: fine-tuned `best.pt`, 6 target classes.
- Tracking: ByteTrack returned track ids.
- Detection/tracking on sample: 8 detections, 1 unique track.
- Event dedup on sample: 2 events (`vehicle_appears`, `vehicle_leaves`).
- Q&A `Co bao nhieu o to?`: `Co 1 o to trong video.`
- Q&A `Co bao nhieu nguoi?`: `Co 0 nguoi trong video.`
- Q&A `Co bao nhieu xe may?`: `Co 0 xe may trong video.`
- Q&A `Co xe tai khong?`: `Khong thay xe tai trong video.`
- Q&A `Co nguoi trong video khong?`: `Khong thay nguoi trong video.`
- Search `tim xe may`: returns motorcycle results from indexed videos with object, frame, timestamp, confidence, and track id.
- Search `xe o to mau vang`: returns `unsupported_attribute`.
- Search `tim doan co nguoi dang di xe may`: returns `unsupported_relationship`.
- Search results include timestamp and the frontend click handler seeks with `video.currentTime = timestamp`.

## Known Limitations

- `local-semantic-v1` is metadata-based. It is stable and light for object-driven demo search, but not rich visual semantic retrieval.
- Gemini is disabled by default; runtime Q&A does not depend on Gemini.
- CLIP/SigLIP is not the default runtime path.
- LocateAnything and TRAKE are not enabled for final P0/P1 demo stability.
- Browser click-to-seek was verified by code path and available result timestamps; no Playwright browser automation is installed in this project.
