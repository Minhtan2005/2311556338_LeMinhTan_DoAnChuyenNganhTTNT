# Phase 3C — KIS Retrieval Hardening: Audit of Current Pipeline

Audit date: 2026-08-28. Scope: code under `backend/app/services`, `backend/app/api`, `backend/app/db`.

## 1. How keyframes are currently sampled

`KeyframeExtractor.extract_for_video` (`app/services/keyframes.py`):

- Candidate frames = every `KEYFRAME_INTERVAL_SECONDS` (default **2.0 s → 0.5 FPS**), plus frames at every `Event.timestamp_start`, plus the last frame.
- Filtering: near-blank frames skipped when `frame.std() < KEYFRAME_MIN_STDDEV` (4.0); visual dedup via a 64-bit difference hash, skip when Hamming distance to any previously kept frame ≤ `KEYFRAME_HASH_DISTANCE` (5).
- Candidate list is truncated to `MAX_KEYFRAMES_PER_VIDEO` (default **120**).
- Thumbnails saved as JPEG under `uploads/keyframes/{video_id}/`.

Conclusion: sampling is **sparse (0.5 FPS effective)**. Short visual moments between event timestamps can be missed.

## 2. Current number of indexed frames per video

- Hard cap: 120 keyframes per video (`max_keyframes_per_video`).
- Docs (`SEMANTIC_KIS.md`) report ~104 frames indexed across the current corpus.

## 3. Current CLIP embedding model and dimensions

- `openai/clip-vit-base-patch32` via `transformers` (`TransformersCLIPProvider` in `app/services/embedding_providers.py`).
- 512-dim, float32, L2-normalized, persisted as JSON in `semantic_embeddings` with `model_name` + `model_version`.
- Legacy `local-semantic-v1` provider retained as offline fallback.
- Vietnamese queries get a hard-coded phrase translation (`prepare_clip_text`) before encoding.

## 4. Current scoring formula

`KISService.search` (`app/services/kis.py`):

```text
final_score = SEMANTIC_WEIGHT (0.85) * cosine(CLIP_query, CLIP_frame) + rerank_bonus
candidates with final_score <= 0 are dropped
```

- Brute-force scan over **all** embeddings of the current model/version (no candidate prefilter).
- Rerank bonus is a small additive sum of object/attribute/near bonuses.

## 5. Current object score logic

- `extract_query_signals` finds objects by Vietnamese/English **substring aliases** on the raw query text (`OBJECT_ALIASES`).
- Bonus: `OBJECT_WEIGHT (0.10) * (matched_objects / requested_objects)`, where detections are matched by `class_name` within ±`keyframe_interval_seconds` of the keyframe timestamp.
- Missing requested objects simply reduce the bonus — **no rejection**.

## 6. Current attribute score logic

- `ATTRIBUTE_WEIGHT = 0.05` — a fixed tiny weight for explicit color constraints.
- Color match: raw query substring aliases (`COLOR_ALIASES`) vs. **whole-frame dominant color**.

## 7. Current relation handling

- `wants_near` keyword heuristic (`cạnh`, `bên cạnh`, `gần`, `next to`, `near`).
- `has_near_person_vehicle`: center-distance ≤ `max(person_size, vehicle_size) * 1.6` → bonus `0.5 * OBJECT_WEIGHT` (0.05).
- No subject/object pairing, no relation type, no configurable threshold, no riding/carrying/inside support.

## 8. Are colors extracted from object-local regions?

**No.** Colors come only from `dominant_color_name()` which resizes the **whole frame** to 1×1 and applies brightness/heuristic rules. No YOLO bbox crop, no person upper/lower body, no vehicle crop. A white car under a blue sky is evaluated as a blue frame.

## 9. Are explicit attributes (blue car, red shirt, backpack) hard filters?

**No.** They are only weak score bonuses (0.05–0.15), derived from substring matching on the raw query — never enforced.

## 10. Are query object types enforced?

**No.** Only CLIP semantic similarity plus the soft 0.10 object bonus. A query for "car" can still rank motorcycle-only frames highly.

## Additional critical findings

1. **Gemini structured output is discarded for KIS retrieval.** `QueryRoutingService._route_kis` parses a full `StructuredQuery` (objects, attributes incl. `color`/`vehicle_color`/`upper_color`/`backpack`, relations) but passes only `semantic_query_en` text into `kis_service.search`. All structured constraints are lost.
2. `prepare_clip_text` silently maps `màu xanh` → `blue`, losing the blue/green ambiguity of Vietnamese "xanh".
3. `COLOR_ALIASES` lacks silver, orange, pink, purple, brown, beige and bare `xanh`.
4. Detections already store `frame_index`, `track_id`, bboxes and confidences — reusable for object crops and spatial verification, but unused by KIS beyond class presence.
5. `semantic_embeddings` has no notion of embedding type (scene vs object crop).
6. Per-frame color/attribute features are computed **at query time** (whole-frame only) — should move to index time.

## Consequences motivating Phase 3C

- Sparse 0.5 FPS indexing misses moments.
- No object presence enforcement → wrong-object frames rank high.
- Whole-frame color → wrong-color frames rank high.
- Explicit constraints weigh ≤ 0.15 vs semantic weight 0.85 → CLIP dominates everything.
- Gemini parse is thrown away.
