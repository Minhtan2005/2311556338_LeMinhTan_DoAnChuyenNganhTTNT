from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas import (
    KISRequest,
    KISResponse,
    LocateRequest,
    LocateResponse,
    QARequest,
    QAResponse,
    TRAKERequest,
    TRAKEResponse,
)
from app.services.grounding import grounding_service
from app.services.kis import kis_service
from app.services.trake import trake_service


router = APIRouter()


@router.post("/kis", response_model=KISResponse)
def semantic_kis(payload: KISRequest, db: Session = Depends(get_db)) -> KISResponse:
    results, meta = kis_service.search_with_meta(
        db=db,
        query=payload.query,
        top_k=payload.top_k,
        enable_grounding_rerank=payload.enable_grounding_rerank,
        grounding_top_n=payload.grounding_top_n,
    )
    return KISResponse(
        query=payload.query,
        model=kis_service.model_name,
        model_version=kis_service.model_version,
        results=results,
        competition_preview=[result.competition_output for result in results],
        status=meta.get("status") or "ok",
        message=meta.get("message"),
        grounding_prompt=meta.get("prompt"),
        grounding_execution_mode=meta.get("execution_mode"),
        latency_ms=meta.get("latency_ms"),
    )


@router.post("/locate", response_model=LocateResponse)
def locate_object(payload: LocateRequest, db: Session = Depends(get_db)) -> LocateResponse:
    image_url = None
    if payload.video_id is not None and payload.frame_id is not None:
        try:
            detections, image_url = grounding_service.locate_in_keyframe(
                db=db,
                video_id=payload.video_id,
                frame_id=payload.frame_id,
                prompt=payload.prompt,
                top_k=payload.top_k,
                confidence_threshold=payload.confidence_threshold,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    elif payload.image_path:
        try:
            detections = grounding_service.locate(
                image_path=payload.image_path,
                prompt=payload.prompt,
                top_k=payload.top_k,
                confidence_threshold=payload.confidence_threshold,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    else:
        raise HTTPException(
            status_code=400,
            detail="Must provide either (video_id and frame_id) or a valid image_path.",
        )

    return LocateResponse(
        prompt=payload.prompt,
        detections=detections,
        model=grounding_service.model_name,
        device=grounding_service.device,
        execution_mode=grounding_service.execution_mode,
        image_url=image_url,
        total_found=len(detections),
    )


@router.get("/locate/status")
def locate_status() -> dict:
    return grounding_service.get_status()


@router.post("/qa", response_model=QAResponse)
def visual_qa(payload: QARequest, db: Session = Depends(get_db)) -> QAResponse:
    from app.services.visual_qa import visual_qa_service

    results = kis_service.search(db, payload.question, top_k=5)
    target = None
    if payload.video_id:
        for r in results:
            if r.video_id == payload.video_id:
                target = r
                break
    if target is None and results:
        target = results[0]

    if target is None:
        raise HTTPException(status_code=404, detail="No matching keyframe found in database.")

    qa_result = visual_qa_service.answer_question(
        image_path=target.image_path,
        question=payload.question,
    )

    competition_row = f"{target.video_name},{target.frame_id},{qa_result.answer}"

    return QAResponse(
        question=payload.question,
        video_name=target.video_name,
        frame_id=target.frame_id,
        answer=qa_result.answer,
        competition_output=competition_row,
        image_url=target.image_url,
        confidence=qa_result.confidence,
        model=qa_result.model,
        latency_ms=qa_result.latency_ms,
        is_insufficient_evidence=qa_result.is_insufficient_evidence,
    )


@router.post("/trake", response_model=TRAKEResponse)
def temporal_trake(payload: TRAKERequest, db: Session = Depends(get_db)) -> TRAKEResponse:
    from app.services.query_router import query_routing_service

    provider, parsed, error, _ = query_routing_service.analyze_query_with_fallback(payload.query)
    events = parsed.events
    return trake_service.align_events(
        db=db,
        events=events,
        video_id=payload.video_id,
        top_k_videos=payload.top_k_videos,
        query=payload.query,
    )

