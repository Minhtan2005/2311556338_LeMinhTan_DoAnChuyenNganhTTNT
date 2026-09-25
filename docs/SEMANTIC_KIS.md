# Semantic KIS

## Purpose

Semantic KIS adds Textual Known Item Search on top of the existing video analysis pipeline. It lets a user search indexed videos with a free-form scene description and returns ranked frame candidates with:

- `video_name`
- `frame_id`
- `timestamp`
- `score`
- thumbnail preview
- competition preview row: `<video_name>,<frame_id>`

This phase does not implement Visual Q&A, TRAKE, conversation memory, paid APIs, or full CSV export.

## Architecture

```text
Existing analyzed video
  -> YOLO/ByteTrack detections and events
  -> keyframe extraction
  -> local semantic embedding
  -> SQLite-persisted vectors
  -> KIS text query
  -> cosine similarity ranking
  -> ranked frame results
```

## Baseline local-semantic-v1 Audit

Phase 2 originally used `local-semantic-v1`. It proved the indexing and retrieval structure worked, but it was not a genuine learned vision-language model.

How `encode_text()` worked:

- Lowercased and normalized the query string.
- Looked for hard-coded Vietnamese/English aliases such as `người`, `xe máy`, `ô tô`, `đỏ`, `cạnh`.
- Set values in a small manually defined feature vector.
- Normalized the vector.

How `encode_image()` worked:

- Loaded the keyframe image with OpenCV.
- Estimated a dominant color from a 1x1 resized image.
- Set only a color feature in the same manual vector.
- During Phase 2 indexing, YOLO metadata was also added to the keyframe vector: detected object classes, vehicle/person flags, and a simple person-vehicle proximity flag.

Were text and image vectors trained/aligned in the same semantic space?

- No. The space was hand-authored, not learned.
- Text features and image features shared dimensions only because the project manually assigned the same feature names.
- There was no pretrained image-text contrastive training.

Was cosine similarity meaningful?

- It was meaningful only as a lightweight structured matching score over manually chosen concepts.
- It was not semantic image-text retrieval in the CLIP/SigLIP sense.
- It could rank frames that shared detected objects with the query, but it could not infer rich visual similarity from image pixels.

Score sources in the baseline:

- Visual embedding: only dominant color from the keyframe image.
- YOLO metadata: object class features around the keyframe timestamp.
- Color metadata: dominant image color and color aliases from text.
- Structured reranking: effectively baked into the vector, not separated from semantic similarity.

Weakness:

- It could produce plausible-looking scores while not being a learned shared image/text representation. Phase 2.5 replaces the default with a real pretrained CLIP model and moves structured signals into a small, explicit rerank bonus.

## Phase 2.5 Model Comparison

| Candidate | Size / dim | CPU / RAM | Vietnamese suitability | Install complexity | License / notes | Assessment |
| --- | --- | --- | --- | --- | --- | --- |
| OpenAI CLIP via `transformers`, `openai/clip-vit-base-patch32` | About 600 MB cached weights, 512D | CPU works; first load is the slowest step | English-first; Vietnamese needs lightweight query normalization | One dependency: `transformers`; uses existing Torch/Pillow | OpenAI CLIP model on Hugging Face; widely used baseline | Selected for Phase 2.5 because it is stable, local, small enough, and easy to replace |
| OpenCLIP | Varies by checkpoint, often 512D+ | CPU works depending on checkpoint | Mostly English unless paired with multilingual text model | Adds `open_clip_torch` and checkpoint selection complexity | Depends on chosen checkpoint | Good later option, but more moving parts for this hardening pass |
| SigLIP / multilingual SigLIP-compatible | Often stronger retrieval, model-dependent dims | CPU works but can be larger/slower | Better multilingual options exist | Usually through `transformers`; checkpoint choice matters | Depends on selected checkpoint | Strong candidate for a future quality upgrade after benchmarking |
| Multilingual CLIP through `sentence-transformers` | Model-dependent, can be larger | CPU works; more packages | Better multilingual text alignment | Adds `sentence-transformers` and model-specific image/text behavior | Depends on model | Useful later if Vietnamese quality from English CLIP normalization is insufficient |

## Selected Embedding Model

Phase 2.5 uses `clip-vit-base-patch32`, backed by `openai/clip-vit-base-patch32` through `transformers`.

Reason:

- It is a real pretrained vision-language model with learned text/image alignment.
- It is local and CPU-compatible.
- It has a manageable 512D embedding and a moderate model size compared with larger SigLIP/VLM options.
- It only adds `transformers`; Torch and Pillow are already present through existing AI dependencies.
- It keeps a clean provider interface, so OpenCLIP, SigLIP, or multilingual CLIP can replace it later.

How it works:

- Keyframe images are encoded by CLIP's image encoder.
- Text queries are encoded by CLIP's text encoder.
- Vietnamese queries are preserved, and a lightweight deterministic Vietnamese-to-English phrase normalization is applied before CLIP encoding because this CLIP checkpoint is English-first.
- All embeddings are `float32`, 512D, normalized, and persisted with `model_name`, `model_version`, and `vector_dim`.
- Search uses global cosine similarity across all indexed videos.
- Structured metadata is applied only as a small configurable rerank bonus.

Limitations:

- `openai/clip-vit-base-patch32` is not multilingual; Vietnamese support depends on the local normalization rules.
- It is still weaker than a tuned multilingual CLIP/SigLIP model for Vietnamese competition prompts.
- It does not answer questions or solve TRAKE.

## Database Changes

Phase 2 adds two SQLite tables through additive startup migration:

### `video_keyframes`

- `id`
- `video_id`
- `frame_id`
- `timestamp_sec`
- `image_path`
- `created_at`

Unique key:

- `(video_id, frame_id)`

### `semantic_embeddings`

- `id`
- `keyframe_id`
- `video_id`
- `model_name`
- `model_version`
- `vector_dim`
- `vector_json`
- `descriptor_text`
- `created_at`

Unique key:

- `(keyframe_id, model_name)`

The actual vector is persisted in SQLite as compact JSON. This avoids regenerating embeddings at every startup and keeps the metadata/vector mapping explicit.

## Keyframe Extraction

Service:

- `backend/app/services/keyframes.py`

Configuration:

- `KEYFRAME_INTERVAL_SECONDS`
- `MAX_KEYFRAMES_PER_VIDEO`
- `KEYFRAME_IMAGE_QUALITY`

Behavior:

- Samples frames at a fixed interval.
- Includes event timestamps as candidate keyframes.
- Skips near-blank frames using frame standard deviation.
- Deduplicates visually similar frames with a small difference-hash distance threshold.
- Saves JPEG thumbnails under `UPLOAD_DIR/keyframes/{video_id}/frame_XXXXXXXX.jpg`.
- Stores keyframe metadata in `video_keyframes`.
- Skips already-existing keyframes when possible.

## Indexing

Automatic:

- After `POST /api/videos/{video_id}/analyze` completes, the analyzer attempts Semantic KIS indexing.
- Indexing errors are logged but do not fail the core YOLO/ByteTrack analysis.

Manual index one video:

```http
POST /api/semantic/index/{video_id}
```

Reindex one or all completed videos:

```http
POST /api/semantic/reindex
```

Request:

```json
{
  "video_id": null,
  "force": true,
  "recreate_keyframes": true
}
```

Response:

```json
{
  "video_id": "...",
  "video_name": "L00_V000.mp4",
  "keyframes_created": 32,
  "embeddings_created": 32,
  "total_keyframes": 32,
  "total_embeddings": 32,
  "model_name": "clip-vit-base-patch32",
  "model_version": "transformers-4.45.2",
  "embedding_dimension": 512
}
```

## KIS API

Endpoint:

```http
POST /api/retrieval/kis
```

Request:

```json
{
  "query": "Tìm cảnh một người đứng cạnh xe máy",
  "top_k": 20
}
```

Response:

```json
{
  "query": "Tìm cảnh một người đứng cạnh xe máy",
  "model": "clip-vit-base-patch32",
  "model_version": "transformers-4.45.2",
  "results": [
    {
      "video_id": "...",
      "video_name": "L00_V000",
      "original_filename": "L00_V000.mp4",
      "frame_id": 1234,
      "timestamp": 49.36,
      "semantic_score": 0.78,
      "rerank_score": 0.04,
      "score": 0.82,
      "image_url": "/media/keyframes/.../frame_00001234.jpg",
      "image_path": "D:\\doan\\backend\\uploads\\keyframes\\...\\frame_00001234.jpg",
      "competition_output": "L00_V000,1234"
    }
  ],
  "competition_preview": [
    "L00_V000,1234"
  ]
}
```

`top_k` is constrained to `1..100`.

If no videos are indexed, the endpoint returns an empty result list.

## Frontend Usage

The existing UI now has a small Semantic KIS section:

1. Upload and analyze a video.
2. Click `Index Semantic KIS` for an already-completed video if it was analyzed before Phase 2.5.
3. Enter a scene description.
4. Click `Tìm KIS`.
5. Inspect thumbnails, video name, frame id, timestamp, score, and competition preview rows.
6. Click a thumbnail to select the video and seek to the candidate timestamp.

## Testing

Added tests cover:

- keyframe extraction metadata
- embedding normalization
- real CLIP text/image embedding dimensions
- vector similarity ranking
- Semantic KIS API endpoint
- persisted embeddings reload
- model version persistence
- cross-video ranking
- competition video names without extension
- no indexed videos
- Vietnamese Unicode query handling
- Top-K constraints

Run:

```bat
cd /d D:\doan\backend
.venv\Scripts\python.exe -m pytest
```

Frontend build:

```bat
cd /d D:\doan\frontend
npm run build
```

## Performance Notes

- Keyframe extraction is linear in sampled frames, not full dense frame indexing.
- CLIP model load dominates the first query/index operation.
- Warm KIS search over the current 104 indexed frames is sub-second to about one second on CPU depending on query and system state.
- Current vectors are 512D and stored in SQLite JSON, so in-memory NumPy ranking is sufficient for MVP-sized datasets.
- For large competition datasets, Phase 2 can evolve to FAISS or another local vector index while keeping the same DB metadata mapping.

## Next Phase 3 Integration

Recommended next step:

- Add a query router that decides whether a request should use legacy object/event search or Semantic KIS.
- Keep `/api/query` backward-compatible.
- Add a richer response contract only after routing is stable.
