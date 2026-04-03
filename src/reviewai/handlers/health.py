"""Health endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

from reviewai.schemas.api import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    service = request.app.state.analysis_service
    return service.health()
