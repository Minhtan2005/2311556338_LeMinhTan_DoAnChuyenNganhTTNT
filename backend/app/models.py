from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def new_uuid() -> str:
    return str(uuid4())


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    content_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="uploaded", index=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_frames: Mapped[int | None] = mapped_column(Integer, nullable=True)
    progress_percent: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    detections: Mapped[list["Detection"]] = relationship(back_populates="video", cascade="all, delete-orphan")
    events: Mapped[list["Event"]] = relationship(back_populates="video", cascade="all, delete-orphan")
    transcripts: Mapped[list["TranscriptSegment"]] = relationship(back_populates="video", cascade="all, delete-orphan")
    keyframes: Mapped[list["VideoKeyframe"]] = relationship(back_populates="video", cascade="all, delete-orphan")


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id"), index=True)
    frame_index: Mapped[int] = mapped_column(Integer, index=True)
    timestamp: Mapped[float] = mapped_column(Float, index=True)
    track_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    class_name: Mapped[str] = mapped_column(String(64), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)

    video: Mapped[Video] = relationship(back_populates="detections")


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id"), index=True)
    event_type: Mapped[str] = mapped_column(String(96), index=True)
    object_type: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    track_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    timestamp_start: Mapped[float] = mapped_column(Float, index=True)
    timestamp_end: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    description: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    video: Mapped[Video] = relationship(back_populates="events")


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id"), index=True)
    start_time: Mapped[float] = mapped_column(Float, index=True)
    end_time: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)

    video: Mapped[Video] = relationship(back_populates="transcripts")


class VideoKeyframe(Base):
    __tablename__ = "video_keyframes"
    __table_args__ = (UniqueConstraint("video_id", "frame_id", name="uq_video_keyframes_video_frame"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id"), index=True)
    frame_id: Mapped[int] = mapped_column(Integer, index=True)
    timestamp_sec: Mapped[float] = mapped_column(Float, index=True)
    image_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    video: Mapped[Video] = relationship(back_populates="keyframes")
    embeddings: Mapped[list["SemanticEmbedding"]] = relationship(
        back_populates="keyframe",
        cascade="all, delete-orphan",
    )


class SemanticEmbedding(Base):
    __tablename__ = "semantic_embeddings"
    __table_args__ = (
        UniqueConstraint("keyframe_id", "model_name", name="uq_semantic_embeddings_keyframe_model"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    keyframe_id: Mapped[int] = mapped_column(ForeignKey("video_keyframes.id"), index=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id"), index=True)
    model_name: Mapped[str] = mapped_column(String(128), index=True)
    model_version: Mapped[str] = mapped_column(String(128), default="", index=True)
    vector_dim: Mapped[int] = mapped_column(Integer)
    vector_json: Mapped[str] = mapped_column(Text, nullable=False)
    descriptor_text: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    keyframe: Mapped[VideoKeyframe] = relationship(back_populates="embeddings")
