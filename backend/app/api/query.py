from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import Video
from app.schemas import QueryRequest, QueryResponse
from app.services.query_router import query_routing_service

router = APIRouter()


@router.post("", response_model=QueryResponse)
def query_video(payload: QueryRequest, db: Session = Depends(get_db)) -> QueryResponse:
    video = db.get(Video, payload.video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Video is not ready for querying. Current status: {video.status}. Please run analysis first.",
        )

    return query_routing_service.route_query(db, payload.video_id, payload.question)
