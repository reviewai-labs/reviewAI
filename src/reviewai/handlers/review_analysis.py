"""Review analysis endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from reviewai.schemas.api import (
    AnalyzeReviewRequest,
    AnalyzeReviewResponse,
    AnalyzeReviewsBatchRequest,
    AnalyzeReviewsBatchResponse,
)

router = APIRouter(tags=["analysis"])


@router.post("/analyze-review", response_model=AnalyzeReviewResponse)
async def analyze_review(
    payload: AnalyzeReviewRequest,
    request: Request,
) -> AnalyzeReviewResponse:
    service = request.app.state.analysis_service
    return service.analyze_review(payload)


@router.post("/analyze-reviews-batch", response_model=AnalyzeReviewsBatchResponse)
async def analyze_reviews_batch(
    payload: AnalyzeReviewsBatchRequest,
    request: Request,
) -> AnalyzeReviewsBatchResponse:
    service = request.app.state.analysis_service
    return service.analyze_reviews(payload)
