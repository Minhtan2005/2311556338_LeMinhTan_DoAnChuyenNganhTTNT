from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BACKEND_DIR / ".env"


class Settings(BaseSettings):
    app_env: str = "development"
    database_url: str = "sqlite:///./data/app.db"
    upload_dir: str = "./uploads"
    yolo_model: str = "yolov8n.pt"
    frame_stride: int = 5
    confidence_threshold: float = 0.35
    expected_traffic_direction: str = "left_to_right"
    restricted_area_polygon: str = ""
    congestion_vehicle_threshold: int = 12
    stopped_min_seconds: float = 3.0
    stopped_max_movement_px: float = 18.0
    keyframe_interval_seconds: float = 2.0
    keyframe_target_fps: float = 2.0
    max_keyframes_per_video: int = 120
    keyframe_image_quality: int = 85
    semantic_model: str = "clip-vit-base-patch32"
    semantic_model_name_or_path: str = "openai/clip-vit-base-patch32"
    semantic_model_version: str = "transformers-4.45.2"
    semantic_model_cache_dir: str = "./data/huggingface"
    semantic_device: str = "auto"
    semantic_batch_size: int = 8
    semantic_weight: float = 0.85
    object_weight: float = 0.10
    attribute_weight: float = 0.05
    relation_weight: float = 0.10
    kis_object_crop_semantic_weight: float = 0.03
    kis_color_confidence_threshold: float = 0.22
    keyframe_min_stddev: float = 4.0
    keyframe_hash_distance: int = 5
    auto_analyze_on_upload: bool = False
    enable_faiss: bool = False
    enable_openai: bool = False
    enable_whisper: bool = False
    whisper_model: str = "base"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    gemini_enabled: bool = True
    gemini_timeout_seconds: float = 30.0
    gemini_cache_ttl_seconds: int = 300
    gemini_max_query_length: int = 1000
    gemini_vision_enabled: bool = False
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    export_dir: str = "./data/exports"
    locate_anything_model: str = "nvidia/LocateAnything-3B"
    locate_anything_enabled: bool = False
    locate_anything_device: str = "auto"
    locate_anything_confidence_threshold: float = 0.25
    locate_anything_timeout_seconds: float = 30.0
    kis_grounding_rerank_enabled: bool = False
    kis_grounding_top_n: int = 3
    kis_grounding_weight: float = 0.35
    kis_grounding_demote_penalty: float = 0.50
    kis_grounding_confidence_threshold: float = 0.25
    yolo_world_enabled: bool = True
    yolo_world_model: str = "yolov8s-worldv2.pt"
    yolo_world_conf: float = 0.25
    yolo_world_sample_fps: float = 2.0
    yolo_world_cache: bool = True
    yolo_world_cache_dir: str = "./data/yolo_world_cache"
    yolo_world_clip_cache_home: str = "./data"
    yolo_world_attribute_prompts: bool = False
    yolo_world_fallback_when_primary_empty: bool = False
    yolo_world_merge_gap_seconds: float = 1.5

    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8")

    def ensure_directories(self) -> None:
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.upload_dir, "keyframes").mkdir(parents=True, exist_ok=True)
        Path(self.semantic_model_cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.export_dir).mkdir(parents=True, exist_ok=True)
        Path(self.yolo_world_cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.yolo_world_clip_cache_home).mkdir(parents=True, exist_ok=True)
        if self.database_url.startswith("sqlite:///"):
            db_path = self.database_url.replace("sqlite:///", "", 1)
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
