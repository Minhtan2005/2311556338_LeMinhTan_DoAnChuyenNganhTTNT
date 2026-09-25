from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class VideoOut(BaseModel):
    id: str
    original_filename: str
    stored_filename: str
    status: str
    duration_seconds: float | None
    fps: float | None
    total_frames: int | None
    progress_percent: int = 0
    error_message: str | None
    created_at: datetime
    processed_at: datetime | None
    media_url: str
    detection_count: int = 0
    track_count: int = 0
    event_count: int = 0
    yolo_model: str
    yolo_model_source: str
    semantic_model: str

    model_config = {"from_attributes": True}


class EventOut(BaseModel):
    id: int
    event_type: str
    object_type: str | None
    track_id: int | None
    timestamp_start: float
    timestamp_end: float | None
    confidence: float
    description: str

    model_config = {"from_attributes": True}


class DetectionOut(BaseModel):
    id: int
    frame_index: int
    timestamp: float
    track_id: int | None
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    model_config = {"from_attributes": True}


class TrackOut(BaseModel):
    track_id: int
    class_name: str
    first_seen: float
    last_seen: float
    detection_count: int
    max_confidence: float


class UploadResponse(BaseModel):
    video: VideoOut
    message: str


class AnalyzeResponse(BaseModel):
    video: VideoOut
    message: str


class QueryRequest(BaseModel):
    video_id: str | None = None
    question: str = Field(min_length=1, max_length=1000)


class RetrievedSegment(BaseModel):
    start_time: float
    end_time: float
    label: str
    confidence: float = 1.0
    evidence: str
    frame_id: int | None = None
    timestamp: float | None = None
    detector: str | None = None
    class_name: str | None = None
    bbox: list[float] | None = None


class QueryDebug(BaseModel):
    intent: str
    entities: dict[str, Any]


class StructuredObject(BaseModel):
    type: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class StructuredRelation(BaseModel):
    subject: str | None = None
    relation: str
    object: str | None = None


class StructuredEvent(BaseModel):
    order: int
    description_vi: str
    semantic_query_en: str
    objects: list[StructuredObject] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    relations: list[StructuredRelation] = Field(default_factory=list)


class StructuredQuery(BaseModel):
    task_type: Literal["KIS", "Q&A", "TRAKE", "LEGACY"]
    legacy_intent: str | None = None
    normalized_query_vi: str
    semantic_query_en: str | None = None
    objects: list[StructuredObject] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    relations: list[StructuredRelation] = Field(default_factory=list)
    spatial: list[str] = Field(default_factory=list)
    count: int | None = None
    question: str | None = None
    events: list[StructuredEvent] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    answer: str
    debug: QueryDebug
    segments: list[RetrievedSegment] = Field(default_factory=list)
    raw_facts: list[str] = Field(default_factory=list)
    query: str | None = None
    nlp_provider: str | None = None
    task_type: str | None = None
    parsed_query: StructuredQuery | None = None
    kis_results: list[dict[str, Any]] = Field(default_factory=list)
    status: str | None = None
    model: str | None = None
    model_version: str | None = None
    competition_preview: list[str] = Field(default_factory=list)


class NLPAnalyzeRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)


class NLPAnalyzeResponse(BaseModel):
    provider: str
    parsed_query: StructuredQuery
    error: str | None = None
    latency_ms: float | None = None


class SemanticIndexResponse(BaseModel):
    video_id: str
    video_name: str
    keyframes_created: int
    embeddings_created: int
    total_keyframes: int
    total_embeddings: int
    model_name: str
    model_version: str
    embedding_dimension: int


class SemanticReindexRequest(BaseModel):
    video_id: str | None = None
    force: bool = True
    recreate_keyframes: bool = False


class SemanticReindexResponse(BaseModel):
    indexed_videos: list[SemanticIndexResponse]
    total_keyframes: int
    total_embeddings: int
    model_name: str
    model_version: str


class NormalizedBoundingBox(BaseModel):
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)


class GroundingDetection(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: NormalizedBoundingBox


class KISRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=20, ge=1, le=100)
    enable_grounding_rerank: bool | None = None
    grounding_top_n: int | None = None


class KISResult(BaseModel):
    video_id: str
    video_name: str
    original_filename: str
    media_url: str | None = None
    frame_id: int
    fps: float | None = None
    timestamp: float
    semantic_score: float
    rerank_score: float
    score: float
    image_url: str
    image_path: str
    object_name: str | None = None
    confidence: float | None = None
    track_id: int | None = None
    competition_output: str
    grounding_score: float | None = None
    grounding_status: str | None = None
    grounding_detections: list[GroundingDetection] = Field(default_factory=list)
    requested_color: str | None = None
    grounded_bbox: NormalizedBoundingBox | None = None
    attribute_scores: dict[str, float] = Field(default_factory=dict)
    predicted_color: str | None = None
    attribute_verified: bool | None = None
    final_rerank_score: float | None = None
    debug_crop_path: str | None = None
    debug_crop_url: str | None = None


class KISResponse(BaseModel):
    query: str
    model: str
    model_version: str
    results: list[KISResult]
    competition_preview: list[str]
    status: str = "ok"
    message: str | None = None
    grounding_prompt: str | None = None
    grounding_execution_mode: str | None = None
    latency_ms: float | None = None


class QARequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    video_id: str | None = None
    top_k: int = Field(default=1, ge=1, le=10)


class QAResponse(BaseModel):
    question: str
    video_name: str
    frame_id: int
    answer: str
    competition_output: str
    image_url: str | None = None
    confidence: float = 1.0
    model: str
    latency_ms: float
    is_insufficient_evidence: bool = False


# ============================================================================
# Competition CSV Export Schemas
# ============================================================================

class KISExportItem(BaseModel):
    video_name: str
    frame_id: int


class QAExportItem(BaseModel):
    video_name: str
    frame_id: int
    answer: str


class TRAKEExportItem(BaseModel):
    video_name: str
    frame_ids: list[int]


class CompetitionExportRequest(BaseModel):
    task_type: Literal["KIS", "QA", "TRAKE"]
    items: list[dict[str, Any]]
    save_file: bool = True
    submission_id: str | None = None


class CompetitionExportResponse(BaseModel):
    task_type: str
    row_count: int
    csv_content: str
    preview_rows: list[str]
    file_name: str | None = None
    download_url: str | None = None


# ============================================================================
# LocateAnything-3B Visual Grounding Schemas
# ============================================================================

class LocateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    video_id: str | None = None
    frame_id: int | None = None
    image_path: str | None = None
    top_k: int = Field(default=10, ge=1, le=50)
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class LocateResponse(BaseModel):
    prompt: str
    detections: list[GroundingDetection]
    model: str
    device: str
    execution_mode: str
    image_url: str | None = None
    total_found: int = 0


# ============================================================================
# TRAKE (Temporal Retrieval and Alignment of Key Events) Schemas
# ============================================================================

class TRAKEEventAlignment(BaseModel):
    order: int
    event_description: str
    frame_id: int
    timestamp: float | None = None
    similarity_score: float = 0.0
    image_url: str | None = None
    image_path: str | None = None


class TRAKERequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    video_id: str | None = None
    top_k_videos: int = Field(default=5, ge=1, le=50)


class TRAKEResponse(BaseModel):
    query: str
    video_id: str | None = None
    video_name: str | None = None
    frame_ids: list[int] = Field(default_factory=list)
    alignments: list[TRAKEEventAlignment] = Field(default_factory=list)
    sequence_score: float = 0.0
    competition_output: str | None = None
    status: str = "ok"
    message: str | None = None
    parsed_events: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0
