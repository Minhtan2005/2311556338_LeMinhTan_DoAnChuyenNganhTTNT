from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy import distinct, func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.models import Detection, Event, SemanticEmbedding, TranscriptSegment, Video, VideoKeyframe
from app.schemas import AnalyzeResponse, DetectionOut, EventOut, TrackOut, UploadResponse, VideoOut
from app.services.analyzer import analyze_video_task
from app.services.storage import save_upload_file

router = APIRouter()


def model_source_label() -> str:
    model_path = Path(settings.yolo_model)
    if model_path.name.lower() == "best.pt":
        return "fine-tuned best.pt"
    return "pretrained/fallback"


def recover_processing_video_if_complete(video: Video, db: Session) -> bool:
    """Repair videos whose analysis artifacts were committed before final status."""
    if video.status != "processing":
        return False
    if int(video.progress_percent or 0) < 100:
        return False
    if not video.total_frames or not video.fps:
        return False

    video.status = "completed"
    video.error_message = None
    if video.processed_at is None:
        video.processed_at = datetime.utcnow()
    db.commit()
    db.refresh(video)
    return True


def to_video_out(video: Video, db: Session | None = None) -> VideoOut:
    detection_count = 0
    track_count = 0
    event_count = 0
    if db is not None:
        recover_processing_video_if_complete(video, db)
        detection_count = db.query(func.count(Detection.id)).filter(Detection.video_id == video.id).scalar() or 0
        event_count = db.query(func.count(Event.id)).filter(Event.video_id == video.id).scalar() or 0
        track_count = (
            db.query(func.count(distinct(Detection.track_id)))
            .filter(Detection.video_id == video.id, Detection.track_id.isnot(None))
            .scalar()
            or 0
        )
    return VideoOut(
        id=video.id,
        original_filename=video.original_filename,
        stored_filename=video.stored_filename,
        status=video.status,
        duration_seconds=video.duration_seconds,
        fps=video.fps,
        total_frames=video.total_frames,
        progress_percent=video.progress_percent,
        error_message=video.error_message,
        created_at=video.created_at,
        processed_at=video.processed_at,
        media_url=f"/media/{Path(video.stored_path).name}",
        detection_count=int(detection_count),
        track_count=int(track_count),
        event_count=int(event_count),
        yolo_model=str(settings.yolo_model),
        yolo_model_source=model_source_label(),
        semantic_model=str(settings.semantic_model),
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> UploadResponse:
    try:
        stored_filename, stored_path = await save_upload_file(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    video = Video(
        original_filename=file.filename or stored_filename,
        stored_filename=stored_filename,
        stored_path=stored_path,
        content_type=file.content_type,
        status="uploaded",
    )
    db.add(video)
    db.commit()
    db.refresh(video)
    if settings.auto_analyze_on_upload:
        background_tasks.add_task(analyze_video_task, video.id)
        message = "Video uploaded. Analysis has started."
    else:
        message = "Video uploaded. Click analyze to start processing."
    return UploadResponse(video=to_video_out(video, db), message=message)


@router.get("", response_model=list[VideoOut])
def list_videos(db: Session = Depends(get_db)) -> list[VideoOut]:
    videos = db.query(Video).order_by(Video.created_at.desc()).all()
    return [to_video_out(video, db) for video in videos]


@router.get("/{video_id}", response_model=VideoOut)
def get_video(video_id: str, db: Session = Depends(get_db)) -> VideoOut:
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return to_video_out(video, db)


@router.post("/{video_id}/analyze", response_model=AnalyzeResponse)
def analyze_video(
    video_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> AnalyzeResponse:
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.status == "processing":
        return AnalyzeResponse(video=to_video_out(video, db), message="Video analysis is already running.")

    db.query(Detection).filter(Detection.video_id == video_id).delete()
    db.query(Event).filter(Event.video_id == video_id).delete()
    db.query(TranscriptSegment).filter(TranscriptSegment.video_id == video_id).delete()
    db.query(SemanticEmbedding).filter(SemanticEmbedding.video_id == video_id).delete()
    db.query(VideoKeyframe).filter(VideoKeyframe.video_id == video_id).delete()
    video.status = "uploaded"
    video.error_message = None
    db.commit()
    db.refresh(video)
    background_tasks.add_task(analyze_video_task, video.id)
    return AnalyzeResponse(video=to_video_out(video, db), message="Video analysis has started.")


@router.get("/{video_id}/detections", response_model=list[DetectionOut])
def list_detections(video_id: str, db: Session = Depends(get_db), limit: int = 500) -> list[DetectionOut]:
    if db.get(Video, video_id) is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return (
        db.query(Detection)
        .filter(Detection.video_id == video_id)
        .order_by(Detection.timestamp.asc())
        .limit(min(max(limit, 1), 2000))
        .all()
    )


@router.get("/{video_id}/tracks", response_model=list[TrackOut])
def list_tracks(video_id: str, db: Session = Depends(get_db)) -> list[TrackOut]:
    if db.get(Video, video_id) is None:
        raise HTTPException(status_code=404, detail="Video not found")
    rows = (
        db.query(
            Detection.track_id,
            Detection.class_name,
            func.min(Detection.timestamp).label("first_seen"),
            func.max(Detection.timestamp).label("last_seen"),
            func.count(Detection.id).label("detection_count"),
            func.max(Detection.confidence).label("max_confidence"),
        )
        .filter(Detection.video_id == video_id, Detection.track_id.isnot(None))
        .group_by(Detection.track_id, Detection.class_name)
        .order_by(func.min(Detection.timestamp))
        .all()
    )
    return [
        TrackOut(
            track_id=int(row.track_id),
            class_name=row.class_name,
            first_seen=float(row.first_seen),
            last_seen=float(row.last_seen),
            detection_count=int(row.detection_count),
            max_confidence=float(row.max_confidence or 0),
        )
        for row in rows
    ]


@router.get("/{video_id}/events", response_model=list[EventOut])
def list_events(video_id: str, db: Session = Depends(get_db)) -> list[EventOut]:
    if db.get(Video, video_id) is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return (
        db.query(Event)
        .filter(Event.video_id == video_id)
        .order_by(Event.timestamp_start.asc())
        .limit(500)
        .all()
    )
