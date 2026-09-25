from fastapi import APIRouter

from app.schemas import NLPAnalyzeRequest, NLPAnalyzeResponse
from app.services.query_router import query_routing_service

router = APIRouter()


@router.post("/analyze", response_model=NLPAnalyzeResponse)
def analyze_nlp(payload: NLPAnalyzeRequest) -> NLPAnalyzeResponse:
    provider, parsed, error, latency_ms = query_routing_service.analyze_query_with_fallback(payload.query)
    return NLPAnalyzeResponse(
        provider=provider,
        parsed_query=parsed,
        error=error,
        latency_ms=latency_ms,
    )
