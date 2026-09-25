# Final Submission Status

## Architecture Thuc Te

- Backend: FastAPI.
- Frontend: React/Vite.
- Database: SQLite via SQLAlchemy.
- Video processing: OpenCV frame reader.
- Detection: YOLOv8n fine-tuned `backend/models/best.pt`.
- Tracking: ByteTrack through Ultralytics `model.track`, with centroid fallback only if no track id exists.
- Keyframes: extracted after analysis and stored under `backend/uploads/keyframes`.
- Search: `local-semantic-v1`, a lightweight metadata feature vector from YOLO/keyframe metadata plus object reranking.
- Q&A: local Vietnamese NLP fallback using detection/tracking facts. Gemini is disabled by default.

## Chuc Nang Hoan Thanh

- Upload video that is stored by backend.
- Video library and video playback.
- Analyze video with fine-tuned YOLO model.
- Detection records with class, confidence, bbox, frame index, timestamp.
- Tracking records with `track_id`.
- Q&A counting uses unique `track_id` first.
- Keyframe extraction and index creation.
- Object search with frame/timestamp/thumbnail/result metadata.
- Result click-to-seek path in frontend uses `video.currentTime = timestamp`.
- Event timeline deduplicates repeated appear/leave events by object class and timestamp bucket.
- Unsupported color/relationship queries do not return misleading results.

## Model Su Dung

Model file:

```text
backend/models/best.pt
```

Classes:

```text
0 person
1 car
2 motorcycle
3 bicycle
4 bus
5 truck
```

Runtime display:

```text
YOLO: fine-tuned best.pt
Semantic/index: local-semantic-v1 metadata features
```

## Test Results

Backend:

```text
74 passed, 1 skipped
```

Frontend:

```text
npm run build passed
```

Note: drive C had 0 bytes free, so frontend build was run with `TEMP`, `TMP`, and npm cache on drive D.

Final API smoke test video:

```text
integration_bestpt_6da0fbd1.mp4
```

Results:

```text
UPLOAD: passed
ANALYSIS: completed
YOLO: fine-tuned best.pt
TRACKING: 1 unique track on sample
DETECTIONS: 8
EVENT DEDUP: 2 events, vehicle_appears + vehicle_leaves
```

Q&A:

```text
Co bao nhieu o to? -> Co 1 o to trong video.
Co bao nhieu nguoi? -> Co 0 nguoi trong video.
Co bao nhieu xe may? -> Co 0 xe may trong video.
Co xe tai khong? -> Khong thay xe tai trong video.
Co nguoi trong video khong? -> Khong thay nguoi trong video.
```

Search:

```text
tim xe may -> returns motorcycle results from indexed videos with object/frame/timestamp/confidence/track_id
tim o to -> returns car results with object/frame/timestamp/confidence/track_id
xe o to mau vang -> unsupported_attribute
tim doan co nguoi dang di xe may -> unsupported_relationship
```

## Demo Workflow

1. Start backend.
2. Start frontend.
3. Open http://127.0.0.1:5173.
4. Upload MP4.
5. Click Analyze.
6. Wait for completed status.
7. Ask required Q&A questions.
8. Search object query such as `tim o to` or `tim xe may`.
9. Click a result to seek video to result timestamp.
10. Try unsupported queries and confirm clear message instead of misleading result.

## Known Limitations

- `local-semantic-v1` is metadata-based, not CLIP/SigLIP.
- Color verification is not trusted in default runtime, so color attributes return `unsupported_attribute`.
- Complex relationship/riding query is not fully supported and returns `unsupported_relationship`.
- Gemini is configured off by default; runtime does not depend on Gemini.
- LocateAnything and TRAKE remain disabled for final stability.
- Browser click-to-seek was verified through code path and API timestamps; no Playwright automation is installed.

## Exact Demo Commands

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

Frontend build when drive C is full:

```powershell
cd D:\doanv2\frontend
$env:TEMP='D:\doanv2\.tmp'
$env:TMP='D:\doanv2\.tmp'
$env:npm_config_cache='D:\doanv2\.npm-cache'
npm run build --cache D:\doanv2\.npm-cache
```
