export type VideoStatus = "uploaded" | "processing" | "completed" | "failed";

export interface VideoItem {
  id: string;
  original_filename: string;
  stored_filename: string;
  status: VideoStatus;
  duration_seconds: number | null;
  fps: number | null;
  total_frames: number | null;
  progress_percent: number;
  error_message: string | null;
  created_at: string;
  processed_at: string | null;
  media_url: string;
  detection_count: number;
  track_count: number;
  event_count: number;
  yolo_model: string;
  yolo_model_source: string;
  semantic_model: string;
}

export interface EventItem {
  id: number;
  event_type: string;
  object_type: string | null;
  track_id: number | null;
  timestamp_start: number;
  timestamp_end: number | null;
  confidence: number;
  description: string;
}

export interface DetectionItem {
  id: number;
  frame_index: number;
  timestamp: number;
  track_id: number | null;
  class_name: string;
  confidence?: number | null;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface QueryResponse {
  answer: string;
  debug: {
    intent: string;
    entities: Record<string, unknown>;
  };
  segments: Array<{
    start_time: number;
    end_time: number;
    label: string;
    confidence: number;
    evidence: string;
    frame_id?: number | null;
    timestamp?: number | null;
    detector?: string | null;
    class_name?: string | null;
    bbox?: number[] | null;
  }>;
  raw_facts: string[];
  query?: string | null;
  nlp_provider?: string | null;
  task_type?: string | null;
  parsed_query?: Record<string, unknown> | null;
  kis_results?: KISResult[];
  status?: string | null;
  model?: string | null;
  model_version?: string | null;
  competition_preview?: string[];
}

export interface SemanticIndexResponse {
  video_id: string;
  video_name: string;
  keyframes_created: number;
  embeddings_created: number;
  total_keyframes: number;
  total_embeddings: number;
  model_name: string;
  model_version: string;
  embedding_dimension: number;
}

export interface KISResult {
  video_id: string;
  video_name: string;
  original_filename: string;
  media_url?: string | null;
  frame_id: number;
  fps?: number | null;
  timestamp: number;
  semantic_score: number;
  rerank_score: number;
  score: number;
  image_url: string;
  image_path: string;
  object_name?: string | null;
  confidence?: number | null;
  track_id?: number | null;
  competition_output: string;
}

export interface KISResponse {
  query: string;
  model: string;
  model_version: string;
  results: KISResult[];
  competition_preview: string[];
  status?: string;
  message?: string | null;
}
