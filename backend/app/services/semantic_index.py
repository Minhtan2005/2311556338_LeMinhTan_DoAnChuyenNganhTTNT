from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import Detection, SemanticEmbedding, Video, VideoKeyframe
from app.services.embedding_providers import SemanticEmbeddingProvider, get_embedding_provider, normalize_vector
from app.services.keyframes import keyframe_extractor


@dataclass(frozen=True)
class SemanticIndexResult:
    keyframes_created: int
    embeddings_created: int
    total_keyframes: int
    total_embeddings: int
    model_name: str
    model_version: str
    embedding_dimension: int


class SemanticIndexService:
    def __init__(self, provider: SemanticEmbeddingProvider | None = None) -> None:
        self.provider = provider or get_embedding_provider()

    @property
    def model_name(self) -> str:
        return self.provider.model_name

    @property
    def model_version(self) -> str:
        return self.provider.model_version

    @property
    def embedding_dimension(self) -> int:
        return self.provider.embedding_dimension

    def index_video(self, db: Session, video: Video, *, force: bool = False, recreate_keyframes: bool = False) -> SemanticIndexResult:
        if recreate_keyframes:
            self.delete_video_index(db, video.id, delete_keyframes=True)
        elif force:
            self.delete_video_index(db, video.id, delete_keyframes=False)

        keyframe_result = keyframe_extractor.extract_for_video(db, video)
        keyframes = db.query(VideoKeyframe).filter(VideoKeyframe.video_id == video.id).order_by(VideoKeyframe.frame_id).all()

        pending = []
        for keyframe in keyframes:
            existing = self._existing_embedding(db, keyframe.id)
            if existing is None or force:
                pending.append(keyframe)

        embeddings_created = 0
        if pending:
            batch_size = max(1, int(settings.semantic_batch_size))
            for offset in range(0, len(pending), batch_size):
                batch = pending[offset : offset + batch_size]
                vectors = self.provider.encode_image_batch([keyframe.image_path for keyframe in batch])
                for keyframe, vector in zip(batch, vectors):
                    existing = self._existing_embedding(db, keyframe.id)
                    descriptor = self._descriptor_text(db, keyframe)
                    if self.model_name == "local-semantic-v1":
                        vector = normalize_vector(vector + self.provider.encode_text(descriptor))
                    else:
                        vector = normalize_vector(vector)
                    vector_json = json.dumps(vector.astype(float).round(6).tolist(), separators=(",", ":"))
                    if existing is None:
                        db.add(
                            SemanticEmbedding(
                                keyframe_id=keyframe.id,
                                video_id=video.id,
                                model_name=self.model_name,
                                model_version=self.model_version,
                                vector_dim=len(vector),
                                vector_json=vector_json,
                                descriptor_text=descriptor,
                            )
                        )
                        embeddings_created += 1
                    else:
                        existing.model_version = self.model_version
                        existing.vector_dim = len(vector)
                        existing.vector_json = vector_json
                        existing.descriptor_text = descriptor
                db.commit()

        db.commit()
        total_embeddings = (
            db.query(SemanticEmbedding)
            .filter(
                SemanticEmbedding.video_id == video.id,
                SemanticEmbedding.model_name == self.model_name,
                SemanticEmbedding.model_version == self.model_version,
            )
            .count()
        )
        return SemanticIndexResult(
            keyframes_created=keyframe_result.created,
            embeddings_created=embeddings_created,
            total_keyframes=keyframe_result.total,
            total_embeddings=total_embeddings,
            model_name=self.model_name,
            model_version=self.model_version,
            embedding_dimension=self.embedding_dimension,
        )

    def delete_video_index(self, db: Session, video_id: str, *, delete_keyframes: bool = False) -> None:
        if delete_keyframes:
            db.query(SemanticEmbedding).filter(SemanticEmbedding.video_id == video_id).delete(synchronize_session=False)
            db.query(VideoKeyframe).filter(VideoKeyframe.video_id == video_id).delete(synchronize_session=False)
        else:
            db.query(SemanticEmbedding).filter(
                SemanticEmbedding.video_id == video_id,
                SemanticEmbedding.model_name == self.model_name,
                SemanticEmbedding.model_version == self.model_version,
            ).delete(synchronize_session=False)
        db.commit()

    def load_vector(self, embedding: SemanticEmbedding) -> np.ndarray:
        return np.asarray(json.loads(embedding.vector_json), dtype=np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        return self.provider.encode_text(text)

    def _existing_embedding(self, db: Session, keyframe_id: int) -> SemanticEmbedding | None:
        return (
            db.query(SemanticEmbedding)
            .filter(
                SemanticEmbedding.keyframe_id == keyframe_id,
                SemanticEmbedding.model_name == self.model_name,
            )
            .one_or_none()
        )

    @staticmethod
    def _descriptor_text(db: Session, keyframe: VideoKeyframe) -> str:
        detections = (
            db.query(Detection.class_name)
            .filter(
                Detection.video_id == keyframe.video_id,
                Detection.timestamp >= keyframe.timestamp_sec - 1.5,
                Detection.timestamp <= keyframe.timestamp_sec + 1.5,
            )
            .limit(100)
            .all()
        )
        classes = sorted({class_name for (class_name,) in detections})
        class_text = ", ".join(classes) if classes else "no detected target objects"
        return f"frame {keyframe.frame_id} at {keyframe.timestamp_sec:.2f}s contains {class_text}"


semantic_index_service = SemanticIndexService()
