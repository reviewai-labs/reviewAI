"""Route registration for the reviewAI app."""

from __future__ import annotations

from fastapi import FastAPI

from reviewai.handlers.health import router as health_router
from reviewai.handlers.playground import router as playground_router
from reviewai.handlers.review_analysis import router as review_analysis_router


def register_routes(app: FastAPI) -> None:
    app.include_router(playground_router)
    app.include_router(health_router)
    app.include_router(review_analysis_router)
