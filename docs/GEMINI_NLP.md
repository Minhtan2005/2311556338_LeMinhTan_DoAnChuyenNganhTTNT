# Gemini NLP Integration

## Purpose

Gemini is used only as a text query parser. It converts Vietnamese user queries into a validated `StructuredQuery` for the existing video retrieval router.

Gemini does not analyze videos, frames, or images in this integration. It must not invent detections, counts, timestamps, frame IDs, or object existence. Those facts still come from YOLOv8, YOLO-World, ByteTrack, SQLite, and keyframe/search metadata.

## Architecture

```text
Vietnamese Query
       |
       v
Gemini NLP
       |
       v
Structured Query
       |
       v
Capability Validation
       |
       v
Query Router
    /       \
YOLOv8    YOLO-World
   |          |
   +----+-----+
        |
     Results
```

Fallback:

```text
Gemini failure
      |
      v
Local NLP
      |
      v
Query Router
```

## Runtime Rules

- `GEMINI_ENABLED=true` enables Gemini for text query parsing.
- `GEMINI_VISION_ENABLED=false` keeps Gemini Vision disabled. This task does not send video frames or images to Gemini.
- Missing key, timeout, invalid JSON, quota/429, or SDK errors all fall back to local NLP.
- Gemini output is still checked by local capability validation.
- Color attributes remain unsupported by default.
- Relationship/action reasoning remains unsupported unless handled by existing KIS/TRAKE paths.

## Environment

`.env.example` contains placeholders only:

```env
GEMINI_ENABLED=false
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TIMEOUT_SECONDS=15
GEMINI_CACHE_TTL_SECONDS=300
GEMINI_MAX_QUERY_LENGTH=1000
GEMINI_VISION_ENABLED=false
```

Local secrets may exist only in `backend/.env`. Do not copy secrets into documentation, tests, commits, or reports.

## Structured Query Schema

The project reuses the existing `StructuredQuery` Pydantic schema:

```text
task_type: KIS | Q&A | TRAKE | LEGACY
legacy_intent: optional existing local intent
normalized_query_vi: normalized Vietnamese query
semantic_query_en: optional English visual retrieval query
objects: list of {type, attributes}
actions: list of action strings
relations: list of {subject, relation, object}
spatial: list of spatial hints
question: optional Q&A text
events: ordered TRAKE events
confidence: parser confidence
```

For simple object queries, Gemini is instructed to return `task_type=LEGACY`, `legacy_intent=search_object` or `count_object`, and a concise English object label such as `car`, `motorcycle`, `dog`, or `ambulance`.

## Routing Examples

```text
"Tìm ô tô"
Gemini -> object=car -> YOLOv8 metadata

"Tìm con chó"
Gemini -> object=dog -> YOLO-World on demand

"Có bao nhiêu xe máy?"
Gemini -> count_object/motorcycle -> YOLOv8 + ByteTrack unique track count

"xe ô tô màu vàng"
Gemini parses color -> capability validation -> unsupported_attribute

"người đang đi xe máy"
Gemini parses relationship -> capability validation -> unsupported_relationship
```

## Services

```text
backend/app/services/gemini_nlp.py
backend/app/services/query_router.py
backend/app/services/open_vocab_query.py
```

`gemini_nlp.py` owns SDK calls, timeout, response schema validation, and cache. Endpoints and downstream services do not call Gemini directly for NLP.

`query_router.py` catches `GeminiNLPError` and unexpected Gemini failures, then automatically falls back to the local parser.

## Tests

Unit tests mock Gemini. They do not call the real API during normal pytest runs.

Covered cases:

- Gemini object `car` routes to YOLOv8 metadata.
- Gemini object `dog` routes to YOLO-World.
- Gemini count `motorcycle` uses primary detection/track metadata.
- Gemini color output stays `unsupported_attribute`.
- Gemini relationship output stays `unsupported_relationship`.
- timeout / 429 / invalid JSON / missing key fall back to local NLP.

Optional live smoke testing can be done manually with configured credentials. Do not print or log the API key.

## Thesis Note

Gemini được sử dụng ở tầng xử lý ngôn ngữ tự nhiên để phân tích và chuẩn hóa truy vấn của người dùng thành biểu diễn có cấu trúc. Việc xác minh nội dung video vẫn được thực hiện bởi các mô hình thị giác máy tính và dữ liệu metadata của hệ thống.

Gemini giúp:

- hiểu truy vấn tiếng Việt,
- xác định intent,
- trích xuất object,
- trích xuất attribute,
- trích xuất relationship/action.

YOLOv8n fine-tuned vẫn là detector chính cho 6 lớp giao thông. YOLO-World là detector open-vocabulary chạy theo yêu cầu cho object ngoài tập lớp chính. ByteTrack cung cấp tracking và unique object count cho đường YOLOv8 primary. Local NLP vẫn là fallback để hệ thống hoạt động khi Gemini lỗi, timeout, hết quota hoặc không có Internet.

## Known Limitations

- Gemini parsing quality depends on the configured model and prompt.
- Gemini understanding a relationship does not mean the CV pipeline can verify it.
- Gemini understanding a color does not mean the system has a reliable color classifier.
- Live API latency depends on network and quota state.
