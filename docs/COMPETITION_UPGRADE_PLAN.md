# Competition Upgrade Plan

## Scope

This document is Phase 1 only: audit and architecture planning for evolving the existing AI Security Video Assistant into a competition-oriented video retrieval system. It does not propose rewriting the project, removing working features, retraining YOLO, downloading large models, or implementing KIS/Q&A/TRAKE yet.

## Existing Project Structure

```text
D:\doan
|-- backend
|   |-- app
|   |   |-- api
|   |   |   |-- query.py
|   |   |   `-- videos.py
|   |   |-- core
|   |   |   `-- config.py
|   |   |-- db
|   |   |   `-- session.py
|   |   |-- ml
|   |   |   |-- data\vietnamese_intents.csv
|   |   |   |-- train_nlp.py
|   |   |   `-- evaluate_nlp.py
|   |   |-- services
|   |   |   |-- analyzer.py
|   |   |   |-- events.py
|   |   |   |-- nlp.py
|   |   |   |-- openai_response.py
|   |   |   |-- retrieval.py
|   |   |   |-- speech.py
|   |   |   `-- storage.py
|   |   |-- main.py
|   |   |-- models.py
|   |   `-- schemas.py
|   |-- data\app.db
|   |-- tests
|   |   |-- test_events.py
|   |   `-- test_nlp.py
|   |-- uploads
|   |-- requirements-base.txt
|   |-- requirements-ai.txt
|   |-- requirements-optional.txt
|   `-- yolov8n.pt
|-- frontend
|   |-- src
|   |   |-- App.tsx
|   |   |-- api.ts
|   |   |-- main.tsx
|   |   |-- styles.css
|   |   `-- types.ts
|   |-- package.json
|   `-- tsconfig.json
|-- scripts
|   `-- smoke_test.py
|-- run_backend.bat
`-- run_frontend.bat
```

## Existing Architecture

```text
React/Vite UI
  -> FastAPI
     -> upload storage
     -> SQLite metadata
     -> YOLOv8n + ByteTrack video analyzer
        -> Detection rows
        -> Event rows
     -> Vietnamese NLP parser
     -> Retrieval service
     -> deterministic or optional OpenAI answer wording
  -> query results with timestamp seeking
```

Current backend entrypoint is `backend/app/main.py`. It creates tables with `Base.metadata.create_all`, mounts `/media`, and exposes `/api/videos` and `/api/query`.

## Current Capability Table

| Feature | Current implementation | File(s) | Status | Limitations | Reusable for competition? |
| --- | --- | --- | --- | --- | --- |
| Frontend shell | React/Vite single-page app with sidebar, video player, upload, analyze button, query panel, event timeline | `frontend/src/App.tsx`, `frontend/src/api.ts`, `frontend/src/types.ts` | PARTIAL | Text encoding appears mojibake in source/display strings; build currently fails with TypeScript module resolution setting | Yes, as operator UI foundation |
| Upload/storage | Validates video extension and saves UUID filename under upload dir | `backend/app/services/storage.py`, `backend/app/api/videos.py` | READY | No dataset-style video name normalization such as `L00_V000`; no duplicate hash/index management | Yes |
| Video metadata | Stores fps, frame count, duration, status/progress/error | `backend/app/models.py`, `backend/app/services/analyzer.py` | READY | No width/height columns; width/height only local during analysis | Yes |
| YOLO detection | Uses Ultralytics YOLO with configured `yolov8n.pt`; target classes are person, bicycle, car, motorcycle, bus, truck, traffic light | `backend/app/services/analyzer.py`, `backend/requirements-ai.txt` | READY foundation | Pretrained COCO only, no fine-tuning; object-only, not semantic scene understanding | Yes |
| ByteTrack tracking | Calls `model.track(..., tracker="bytetrack.yaml", persist=True)` and maps returned ids to detections | `backend/app/services/analyzer.py` | PARTIAL | No explicit dependency/runtime check for ByteTrack extras; fallback only assigns centroid ids when YOLO returns no id, not when `model.track` raises | Yes, but harden first |
| Attribute extraction | Current code records object class, bbox, confidence, track id, timestamp | `backend/app/models.py`, `backend/app/services/analyzer.py` | PARTIAL | No upper/lower clothing color, vehicle color, holding object, pose, scene attributes in this checked `D:\doan` tree | Detection metadata is reusable; attributes need schema/services |
| Event extraction | Rule-based events: appears/leaves, wrong_way, stopped, congestion, restricted area | `backend/app/services/events.py` | PARTIAL | Rules are traffic-oriented; no human action/event sequence model; restricted area requires config | Useful foundation for TRAKE candidates |
| Speech/transcript | Optional FFmpeg + Whisper transcription into `transcript_segments` | `backend/app/services/speech.py`, `requirements-optional.txt` | PARTIAL | Disabled by default; requires FFmpeg and Whisper; no transcript data currently | Optional later |
| Vietnamese NLP | Underthesea tokenization with TF-IDF + LinearSVC intent classifier; alias entity extraction | `backend/app/services/nlp.py`, `backend/app/ml/train_nlp.py`, `backend/app/ml/data/vietnamese_intents.csv` | PARTIAL | Actual dataset has 240 rows but trainer requires at least 250; no confidence score; no competition task router | Reusable for initial Vietnamese query parsing |
| Retrieval | Intent-specific DB queries plus TF-IDF semantic search over event/transcript text; optional FAISS wrapper for TF-IDF vectors | `backend/app/services/retrieval.py` | PARTIAL | Not visual-semantic embeddings; not cross-video by default; no ranking calibration; output is time segment, not CSV frame id | Reusable as object/event retrieval baseline |
| Natural answer | Deterministic Vietnamese template; optional OpenAI for wording only | `backend/app/services/openai_response.py` | PARTIAL | No visual Q&A answer generation; optional API disabled by default | Reusable for evidence-bound answers |
| API endpoints | Video CRUD-ish workflow, detections/tracks/events, query | `backend/app/api/videos.py`, `backend/app/api/query.py`, `backend/app/schemas.py` | READY foundation | No `/api/runtime`; no KIS/Q&A/TRAKE endpoints; no CSV export | Yes |
| Database | SQLite with videos, detections, events, transcript_segments | `backend/app/models.py`, `backend/data/app.db` | READY foundation | No migrations; no keyframes, embeddings, captions, event candidates, result/export tables | Yes, needs additive evolution |
| Tests | Unit tests for NLP parser and event detector; smoke test for API flow | `backend/tests`, `scripts/smoke_test.py` | PARTIAL | Backend venv currently lacks pytest; smoke depends on running backend and CV deps | Expand for competition tasks |
| Model files | `backend/yolov8n.pt` exists, about 6.5 MB | `backend/yolov8n.pt` | READY | Pretrained small model only | Keep |

## Runtime Verification Notes

Static audit was completed against the current `D:\doan` project.

Light checks performed:

- SQLite database exists at `backend/data/app.db`, size about 421 KB.
- Current DB tables: `videos`, `detections`, `events`, `transcript_segments`.
- Current DB counts: 4 videos, 1256 detections, 271 events, 0 transcript segments.
- Detected classes currently present in DB: person, car, motorcycle, bus, bicycle, truck.
- Current uploaded/analyzed videos include two `completed` videos and two `uploaded` videos.

Runtime blockers found:

- Backend test command `backend\.venv\Scripts\python.exe -m pytest` fails because `pytest` is not installed in the current `D:\doan\backend\.venv`.
- Frontend `npm run build` fails because TypeScript reports `moduleResolution=node10` has been removed. `frontend/tsconfig.json` currently uses `"moduleResolution": "Node"`, which current TypeScript normalizes to the removed `node10` mode.
- A previous dependency install was cancelled while installing backend requirements, so the venv may be incomplete.
- `backend/app/ml/models/intent_classifier.joblib` does not exist in this tree.

These blockers should be fixed before Phase 2 implementation, but they are not fixed in Phase 1.

## Chatbot Audit

The frontend query panel is connected to real backend logic:

- `frontend/src/App.tsx` calls `queryVideo(selectedVideo.id, question)`.
- `frontend/src/api.ts` sends `POST /api/query` with `{ video_id, question }`.
- `backend/app/api/query.py` validates that the selected video is `completed`.
- The query API calls `query_parser.parse`, then `retrieval_service.retrieve`, then `natural_answer_service.generate`.
- Results include `answer`, `debug.intent`, `debug.entities`, `segments`, and `raw_facts`.
- The UI renders returned segments and allows seeking the HTML video player to `segment.start_time`.

Current chatbot limitations:

- It is a single-turn query form, not a conversational memory system.
- It does not maintain conversation context, references like "that person", or follow-up state.
- It has no explicit router for KIS/Q&A/TRAKE.
- It only queries one selected video at a time.
- It cannot answer visual questions requiring inspection of a retrieved frame beyond stored detections/events.
- It cannot export competition rows.

Conclusion: the chatbot is not merely decorative, but it is currently a single-turn object/event retrieval UI, not yet a competition task assistant.

## NLP Audit

Actual intents:

- `search_object`
- `search_action`
- `find_timestamp`
- `count_object`
- `summarize_video`

Dataset:

- File: `backend/app/ml/data/vietnamese_intents.csv`
- Actual rows: 240
- Intent counts: `search_object` 65, `search_action` 55, `find_timestamp` 40, `count_object` 40, `summarize_video` 40

Training pipeline:

- `backend/app/ml/train_nlp.py`
- Uses Underthesea tokenization, `TfidfVectorizer`, `LinearSVC(class_weight="balanced")`
- Splits train/test with stratification
- Saves to `backend/app/ml/models/intent_classifier.joblib`

Important issue:

- `load_dataset()` requires at least 250 rows, but the dataset currently has 240. Training will fail until either the dataset is expanded or the threshold is adjusted with justification.
- Runtime parser can still train from CSV in `services/nlp.py` because that fallback does not enforce the 250-row check, but no persisted model artifact exists.

Entity extraction:

- Rule-based aliases for object, action, color, direction, and time ranges.
- Useful for baseline object/event search.
- It does not parse rich scene semantics, multi-event TRAKE steps, Q&A answer type, spatial relations, or contextual references.
- It does not expose classifier confidence; `LinearSVC` does not naturally return probabilities unless calibrated or distance margins are added.

## Computer Vision Pipeline Audit

Model:

- `settings.yolo_model` defaults to `yolov8n.pt`.
- `backend/yolov8n.pt` is present.
- Current implementation is pretrained COCO, not fine-tuned.

Detection classes:

- Hard-coded target class names:
  - person
  - bicycle
  - car
  - motorcycle
  - bus
  - truck
  - traffic light

Frame sampling:

- `FRAME_STRIDE=5` by default.
- Analyzer only processes every fifth frame.
- Timestamp is `frame_index / fps`.
- Stored `frame_index` is the original video frame index, which is useful for competition CSV frame output.

Tracking:

- Uses Ultralytics `model.track` with `bytetrack.yaml`, `persist=True`.
- If `boxes.id` is missing, `LocalCentroidTracker` assigns local ids beginning at 100000.
- There is no explicit fallback if `model.track` raises due to missing tracking dependencies.

Attributes:

- Current `D:\doan` code stores bbox geometry and class only.
- It does not store clothing color, lower clothing color, vehicle color, bag/holding object as dedicated attributes.
- Backpack can be detected only if added to target classes; current target set does not include backpack.

Events:

- `EventDetector` observes tracks and emits rule-based events.
- It can create:
  - `vehicle_appears`
  - `object_appears`
  - `vehicle_leaves`
  - `object_leaves`
  - `wrong_way`
  - `vehicle_stopped`
  - `traffic_congestion`
  - `enter_restricted_area`
- These are useful event candidates, but not enough for semantic event retrieval such as "person exits vehicle" or "person returns carrying bag".

## Database Schema Audit

Current tables:

### `videos`

Columns:

- `id`
- `original_filename`
- `stored_filename`
- `stored_path`
- `content_type`
- `status`
- `duration_seconds`
- `fps`
- `total_frames`
- `progress_percent`
- `error_message`
- `created_at`
- `processed_at`

### `detections`

Columns:

- `id`
- `video_id`
- `frame_index`
- `timestamp`
- `track_id`
- `class_name`
- `confidence`
- `x1`
- `y1`
- `x2`
- `y2`

### `events`

Columns:

- `id`
- `video_id`
- `event_type`
- `object_type`
- `track_id`
- `timestamp_start`
- `timestamp_end`
- `confidence`
- `description`
- `metadata_json`
- `created_at`

### `transcript_segments`

Columns:

- `id`
- `video_id`
- `start_time`
- `end_time`
- `text`

Schema support assessment:

| Requirement | Current support | Notes |
| --- | --- | --- |
| `video_id` | READY | Present in detection/event/transcript tables |
| `frame_id` | READY foundation | `detections.frame_index` exists; events use timestamp but not representative frame |
| timestamp | READY | Detection timestamp and event start/end exist |
| detections | READY | Dedicated `detections` table |
| tracks | PARTIAL | Track ids are embedded in detections/events; no dedicated track table |
| attributes | PARTIAL | No dedicated attribute table in current `D:\doan` tree |
| keyframes | MISSING | Needed for KIS/Q&A/TRAKE |
| scene embeddings | MISSING | Needed for semantic retrieval |
| captions | MISSING | Useful for semantic search and Q&A |
| event candidates | PARTIAL | `events` exists, but limited rule vocabulary |
| retrieval results | MISSING | No persisted competition result/export table |

Database evolution should be additive. Use migrations instead of destructive schema recreation.

## Competition Requirements Gap Analysis

### Textual KIS

| Capability | Status | Notes |
| --- | --- | --- |
| Object-based retrieval | READY | Can find object classes and timestamps through detections/tracks |
| Event-based retrieval | PARTIAL | Rule-based events only; traffic-oriented |
| Text query understanding | PARTIAL | Vietnamese object/action/time aliases; no semantic decomposition |
| Semantic visual retrieval | MISSING | No CLIP/SigLIP/image-text embeddings, keyframes, captions, or vector index |
| Cross-video retrieval | MISSING | Current API queries one selected video |
| Ranking | PARTIAL | Basic DB ordering and TF-IDF over event text |
| Required output `<video_name>,<frame_id>` | MISSING | No competition output formatter/export endpoint |

### Visual Question Answering

| Capability | Status | Notes |
| --- | --- | --- |
| Retrieve relevant frame/video | PARTIAL | Can retrieve object/event time segments within one video |
| Inspect visual content | PARTIAL | Stored detections/bboxes only; no frame crops/keyframe store |
| Count people/objects | PARTIAL | Can count tracks by class, but not frame-specific unless queried from detections manually |
| Answer color/holding/spatial questions | MISSING | Missing visual attributes, relation extraction, and VQA model/service |
| Required output `<video_name>,<frame_id>,<answer>` | MISSING | No Q&A route or CSV export |

### TRAKE

| Capability | Status | Notes |
| --- | --- | --- |
| Tracking foundation | PARTIAL | Track ids and timestamps exist |
| Event candidates | PARTIAL | Some rule-based vehicle events exist |
| Multi-event query decomposition | MISSING | NLP has no ordered event parser |
| Temporal alignment | MISSING | No sequence ranking/alignment over candidate events |
| Person-object/vehicle interaction | MISSING | No relation/event model for exits, enters, carries, returns |
| Required output exactly one frame per event | MISSING | No frame selection/export logic |

## Safest Upgrade Architecture

Preserve the current pipeline and add new modules around it:

```text
Existing video ingestion
  -> Existing YOLOv8n + ByteTrack analyzer
  -> Existing Detection/Event SQLite tables
  -> Additive post-processing:
     -> keyframe extraction
     -> track summaries
     -> attribute extraction
     -> relation candidates
     -> semantic captions
     -> visual/text embeddings
     -> vector index

User query/chatbot
  -> query understanding
  -> task router
     -> KIS retriever
     -> Q&A retriever + answerer
     -> TRAKE decomposer + temporal aligner
  -> ranker
  -> response formatter
  -> competition CSV export
```

Recommended module boundaries:

- Keep `services/analyzer.py` focused on detection/tracking and raw per-frame records.
- Add `services/keyframes.py` for extracting/storing representative frames.
- Add `services/semantic_index.py` for embeddings and vector index management.
- Add `services/query_router.py` for KIS/Q&A/TRAKE routing.
- Add `services/kis.py`, `services/vqa.py`, and `services/trake.py` for task-specific retrieval.
- Add `services/export.py` for competition CSV rows.
- Add new SQLAlchemy models for keyframes, embeddings metadata, captions, event candidates, and exports.
- Keep current `POST /api/query` backward-compatible; add task-specific endpoints rather than breaking existing UI.

## Proposed Database Evolution

Add migrations and do not destroy existing data.

Suggested new tables:

| Table | Purpose |
| --- | --- |
| `tracks` | One row per track with class, first/last frame, first/last timestamp, summary geometry |
| `object_attributes` | Per-track or per-keyframe attributes such as clothing color, vehicle color, bag/holding flags |
| `keyframes` | Representative frames with `video_id`, `frame_index`, `timestamp`, path, extraction reason |
| `frame_captions` | Textual captions/descriptions for keyframes or frame windows |
| `scene_embeddings` | Metadata for embedding vectors, model name, vector path/index id |
| `semantic_index_runs` | Version metadata for embedding/index builds |
| `event_candidates` | Richer event candidates with subject/object/action/time/window/confidence |
| `relations` | Person-object/person-vehicle spatial or temporal relations |
| `query_logs` | Optional record of task, query, selected results, debug info |
| `competition_exports` | Export jobs and generated CSV paths |

For vector storage, keep SQLite metadata but store actual vectors in a local index file under something like `backend/data/indexes/`, using FAISS where available or NumPy/SQLite fallback for Windows.

## Proposed API Evolution

Keep existing endpoints:

- `POST /api/videos/upload`
- `POST /api/videos/{video_id}/analyze`
- `GET /api/videos`
- `GET /api/videos/{video_id}`
- `GET /api/videos/{video_id}/detections`
- `GET /api/videos/{video_id}/tracks`
- `GET /api/videos/{video_id}/events`
- `POST /api/query`

Add later:

- `GET /api/runtime`: environment readiness, model paths, optional feature status
- `POST /api/videos/{video_id}/index`: create keyframes/captions/embeddings
- `GET /api/videos/{video_id}/keyframes`
- `POST /api/retrieve/kis`
- `POST /api/retrieve/vqa`
- `POST /api/retrieve/trake`
- `POST /api/query/chat`: backward-compatible chatbot router that can call KIS/Q&A/TRAKE
- `POST /api/exports/competition`
- `GET /api/exports/{export_id}`

## Implementation Roadmap

### Phase 1: Audit / Architecture

- Objective: Document current codebase and define safe upgrade path.
- Files to add: `docs/COMPETITION_UPGRADE_PLAN.md`.
- Files to modify: none beyond documentation.
- Database changes: none.
- APIs: none.
- Models/libraries needed: none.
- Risks: Audit can miss runtime issues if dependencies are incomplete.
- Test criteria: Static audit complete; current blockers recorded.
- Estimated difficulty: Low.

### Phase 2: Semantic KIS

Status: IMPLEMENTED and hardened in Phase 2.5 with a real pretrained CLIP embedding provider.

- Objective: Add keyframe extraction and semantic image/text retrieval while preserving object/event retrieval.
- Files to add:
  - `backend/app/services/keyframes.py`
  - `backend/app/services/semantic_index.py`
  - `backend/app/services/kis.py`
  - `backend/app/api/retrieval.py`
  - migration scripts for keyframes/embeddings/captions
  - tests for keyframe extraction and KIS ranking contract
- Files to modify:
  - `backend/app/main.py`
  - `backend/app/models.py`
  - `backend/app/schemas.py`
  - `backend/app/core/config.py`
  - `frontend/src/api.ts`
  - `frontend/src/types.ts`
  - minimally `frontend/src/App.tsx`
- Database changes:
  - Add `keyframes`, `frame_captions`, `scene_embeddings`, `semantic_index_runs`.
- APIs:
  - `POST /api/videos/{video_id}/index`
  - `POST /api/retrieve/kis`
  - `GET /api/videos/{video_id}/keyframes`
- Models/libraries needed:
  - Start with lightweight CLIP-compatible embeddings only after discussion.
  - Candidate: `sentence-transformers` CLIP/SigLIP small model or ONNX alternative.
  - Avoid downloading large VLMs in this phase without approval.
- Risks:
  - Windows dependency friction.
  - Embedding model size/performance.
  - Need clear cache/versioning of indexes.
- Test criteria:
  - Existing upload/analyze/query still works.
  - Keyframes stored with correct `video_id`, `frame_index`, `timestamp`.
  - KIS returns `<video_name>,<frame_id>` candidates.
  - Object-only queries still match current retrieval baseline.
- Estimated difficulty: Medium.

Implementation details now in place:

- `backend/app/services/keyframes.py`
- `backend/app/services/semantic_index.py`
- `backend/app/services/kis.py`
- `backend/app/api/semantic.py`
- `backend/app/api/retrieval.py`
- Additive migration for `video_keyframes` and `semantic_embeddings`
- Minimal frontend Semantic KIS panel
- Documentation: `docs/SEMANTIC_KIS.md`

Current default model: `clip-vit-base-patch32` using `openai/clip-vit-base-patch32` through `transformers`.

`local-semantic-v1` remains documented as the Phase 2 baseline/fallback, but it is no longer the default because it was not a learned image-text embedding space.

### Phase 3: Chatbot Query Router

- Status: Phase 3A implemented Gemini API NLP as the primary understanding layer; full Visual Q&A and TRAKE execution are still intentionally not implemented.
- Objective: Turn the current query form into a real router for legacy, KIS, Q&A, and TRAKE without losing current Vietnamese queries.
- Files to add:
  - `backend/app/services/query_router.py`
  - `backend/app/services/query_types.py`
  - router tests
- Files to modify:
  - `backend/app/api/query.py`
  - `backend/app/services/nlp.py`
  - `backend/app/schemas.py`
  - `frontend/src/App.tsx`
- Database changes:
  - Optional `query_logs`.
- APIs:
  - Keep `POST /api/query`.
  - Add `POST /api/query/chat` if the response contract becomes richer.
- Models/libraries needed:
  - Existing TF-IDF/LinearSVC can remain.
  - Add rule-based task classifier first; optional calibrated classifier later.
- Risks:
  - Misrouting semantic KIS vs Q&A vs TRAKE.
  - Vietnamese/English mixed competition prompts.
- Test criteria:
  - Legacy questions still return same intents.
  - KIS-like prompts route to KIS.
  - Direct questions route to Q&A.
  - Numbered/multi-step prompts route to TRAKE.
- Estimated difficulty: Medium.

Phase 3A implementation details:

- `backend/app/services/gemini_nlp.py` uses the official `google-genai` SDK to convert Vietnamese or English queries into validated `StructuredQuery` JSON.
- `backend/app/services/query_router.py` routes `KIS` to Semantic KIS, `LEGACY` to the existing local retrieval engine, and returns `engine_not_implemented` for `Q&A` and `TRAKE`.
- The existing Underthesea + TF-IDF + LinearSVC parser is preserved as `local_fallback` and is used automatically when Gemini is unavailable or returns invalid output.
- `semantic_query_en` is preferred for CLIP-based KIS retrieval because the current default model is `openai/clip-vit-base-patch32`.
- `POST /api/nlp/analyze` was added for NLP-only debugging without retrieval.
- `POST /api/query` remains backward-compatible and now includes optional NLP/debug fields.
- Documentation: `docs/GEMINI_NLP.md`.

### Phase 4: Visual Q&A

- Objective: Retrieve relevant frames and answer visual questions from stored metadata first, then optional visual model.
- Files to add:
  - `backend/app/services/vqa.py`
  - `backend/app/services/frame_inspector.py`
  - VQA tests with deterministic metadata answers
- Files to modify:
  - `backend/app/api/retrieval.py`
  - `backend/app/schemas.py`
  - frontend response rendering
- Database changes:
  - Add or use `object_attributes`, `relations`, `frame_captions`.
- APIs:
  - `POST /api/retrieve/vqa`
- Models/libraries needed:
  - Phase 4A: no large model; answer from detections/attributes/captions.
  - Phase 4B: discuss adding a small local VLM or API-backed VQA.
- Risks:
  - Hallucinated answers if evidence is weak.
  - Color/holding/spatial answers need reliable attribute extraction.
- Test criteria:
  - Count questions answer from exact frame window.
  - Color/spatial answers include confidence and fallback "unknown".
  - Output supports `<video_name>,<frame_id>,<answer>`.
- Estimated difficulty: Medium to High.

### Phase 5: TRAKE

- Objective: Support ordered multi-event temporal retrieval and alignment.
- Files to add:
  - `backend/app/services/trake.py`
  - `backend/app/services/event_decomposer.py`
  - `backend/app/services/temporal_aligner.py`
  - TRAKE tests
- Files to modify:
  - `backend/app/services/events.py`
  - `backend/app/models.py`
  - retrieval API/router
- Database changes:
  - Add `event_candidates`, `relations`, and possibly track summaries.
- APIs:
  - `POST /api/retrieve/trake`
- Models/libraries needed:
  - Start rule-based and metadata-based.
  - Later consider temporal/action recognition models only after evaluation.
- Risks:
  - Current events are too coarse for "exits vehicle", "enters building", "carrying bag".
  - Need robust sequence scoring and exact event count.
- Test criteria:
  - Ordered prompts produce exactly N frames for N events.
  - All returned frames belong to same selected video unless cross-video mode is requested.
  - Returned frames are temporally increasing.
- Estimated difficulty: High.

### Phase 6: Competition CSV Export

- Objective: Export task-specific rows in required formats.
- Files to add:
  - `backend/app/services/export.py`
  - `backend/app/api/exports.py`
  - export tests
- Files to modify:
  - `backend/app/main.py`
  - `backend/app/schemas.py`
  - frontend export controls
- Database changes:
  - Add `competition_exports`.
- APIs:
  - `POST /api/exports/competition`
  - `GET /api/exports/{export_id}`
- Models/libraries needed:
  - Standard CSV only.
- Risks:
  - Need map internal original filenames to official `Lxx_Vxxx` video ids.
  - Need choose one best frame vs segment start.
- Test criteria:
  - KIS rows: `<video_name>,<frame_id>`.
  - Q&A rows: `<video_name>,<frame_id>,<answer>`.
  - TRAKE rows: `<video_name>,<frame_event_1>,...,<frame_event_n>`.
- Estimated difficulty: Low to Medium.

### Phase 7: Evaluation / Benchmarking

- Objective: Add repeatable benchmarks for retrieval quality and runtime.
- Files to add:
  - `backend/app/eval/*`
  - `scripts/evaluate_competition.py`
  - sample ground-truth fixtures
- Files to modify:
  - docs and CI/test scripts
- Database changes:
  - Optional `evaluation_runs`.
- APIs:
  - Optional read-only evaluation summary endpoint.
- Models/libraries needed:
  - Metrics only: recall@k, mAP-like retrieval metrics, exact frame tolerance, answer accuracy.
- Risks:
  - Need labeled validation data.
  - Frame tolerance rules must match competition.
- Test criteria:
  - Benchmark can run locally.
  - Reports per-task metrics and latency.
  - Regression thresholds protect existing behavior.
- Estimated difficulty: Medium.

## Dependency Recommendations

Keep current dependencies for baseline:

- FastAPI, SQLAlchemy, SQLite
- OpenCV
- Ultralytics YOLO
- Torch
- Underthesea
- scikit-learn
- NumPy

Recommended later, after approval:

- Alembic for schema migrations.
- A lightweight image-text embedding stack for KIS, preferably with local caching and explicit model size.
- FAISS only where Windows installation is stable; otherwise NumPy or hnswlib fallback.
- Optional VQA/VLM only after the metadata-first Q&A baseline is working.

Do not add paid APIs, OpenAI APIs, huge VLMs, COCO/BDD100K downloads, or long training jobs without explicit approval.

## Testing Strategy

Immediate stabilization tests before Phase 2:

- Fix backend venv and run `python -m pytest`.
- Fix frontend TypeScript config and run `npm run build`.
- Run smoke test against a real running backend.
- Add a runtime check endpoint or script to report OpenCV, Ultralytics, Torch, YOLO model, DB path, upload path, and NLP model status.

Competition tests:

- KIS unit tests for query parsing and output format.
- KIS integration tests with small fixture videos/keyframes.
- Q&A tests for count/color/object-near questions with known expected answers.
- TRAKE tests for ordered event decomposition and temporal monotonicity.
- Export tests for exact CSV formatting.
- Regression tests to ensure old Vietnamese queries still work.

## Risks and Blockers

- Current `D:\doan` backend venv is incomplete; `pytest` is missing and a prior pip install was cancelled.
- Frontend build fails under current TypeScript until `moduleResolution` is updated.
- NLP training script threshold conflicts with actual dataset size.
- Source files contain mojibake Vietnamese text, likely from encoding corruption. This affects UI text, NLP aliases, tests, and generated answers.
- No migration framework exists; adding competition schema via `create_all` alone is risky for existing data.
- Current retrieval is single-video; competition usually needs dataset-wide retrieval.
- Current detection/tracking does not capture semantic scene context, object relations, or fine-grained actions.
- ByteTrack failure modes need a tested fallback path.
- Official video naming/output mapping is not implemented.

## Phase 2 Change List Preview

If Phase 2 is approved, expected changed files are:

- Add `backend/app/services/keyframes.py`
- Add `backend/app/services/semantic_index.py`
- Add `backend/app/services/kis.py`
- Add `backend/app/api/retrieval.py`
- Add migration support and migration file for `keyframes`, `frame_captions`, `scene_embeddings`, `semantic_index_runs`
- Modify `backend/app/main.py`
- Modify `backend/app/models.py`
- Modify `backend/app/schemas.py`
- Modify `backend/app/core/config.py`
- Modify `backend/app/services/retrieval.py` only for shared helpers/backward compatibility
- Modify `frontend/src/api.ts`
- Modify `frontend/src/types.ts`
- Minimally modify `frontend/src/App.tsx` to expose KIS results without redesigning the UI
- Add focused tests under `backend/tests`

No Phase 2 work should begin until this plan is approved.
