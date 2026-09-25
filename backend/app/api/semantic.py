from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import Video
from app.schemas import SemanticIndexResponse, SemanticReindexRequest, SemanticReindexResponse
from app.services.semantic_index import semantic_index_service
from app.services.video_names import competition_video_name


router = APIRouter()


@router.post("/index/{video_id}", response_model=SemanticIndexResponse)
def index_video(video_id: str, db: Session = Depends(get_db)) -> SemanticIndexResponse:
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.status != "completed":
        raise HTTPException(status_code=409, detail=f"Video is not completed. Current status: {video.status}")

    result = semantic_index_service.index_video(db, video)
    return SemanticIndexResponse(
        video_id=video.id,
        video_name=competition_video_name(video),
        keyframes_created=result.keyframes_created,
        embeddings_created=result.embeddings_created,
        total_keyframes=result.total_keyframes,
        total_embeddings=result.total_embeddings,
        model_name=result.model_name,
        model_version=result.model_version,
        embedding_dimension=result.embedding_dimension,
    )


@router.post("/reindex", response_model=SemanticReindexResponse)
def reindex(payload: SemanticReindexRequest, db: Session = Depends(get_db)) -> SemanticReindexResponse:
    if payload.video_id:
        video = db.get(Video, payload.video_id)
        if video is None:
            raise HTTPException(status_code=404, detail="Video not found")
        videos = [video]
    else:
        videos = db.query(Video).filter(Video.status == "completed").order_by(Video.created_at.asc()).all()

    responses: list[SemanticIndexResponse] = []
    for video in videos:
        if video.status != "completed":
            raise HTTPException(status_code=409, detail=f"Video {video.id} is not completed. Current status: {video.status}")
        result = semantic_index_service.index_video(
            db,
            video,
            force=payload.force,
            recreate_keyframes=payload.recreate_keyframes,
        )
        responses.append(
            SemanticIndexResponse(
                video_id=video.id,
                video_name=competition_video_name(video),
                keyframes_created=result.keyframes_created,
                embeddings_created=result.embeddings_created,
                total_keyframes=result.total_keyframes,
                total_embeddings=result.total_embeddings,
                model_name=result.model_name,
                model_version=result.model_version,
                embedding_dimension=result.embedding_dimension,
            )
        )

    return SemanticReindexResponse(
        indexed_videos=responses,
        total_keyframes=sum(item.total_keyframes for item in responses),
        total_embeddings=sum(item.total_embeddings for item in responses),
        model_name=semantic_index_service.model_name,
        model_version=semantic_index_service.model_version,
    )
