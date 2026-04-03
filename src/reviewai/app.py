"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from reviewai.config import Settings, get_settings
from reviewai.errors import register_exception_handlers
from reviewai.logger import configure_logging
from reviewai.middleware import add_logging_middleware, add_request_context_middleware
from reviewai.routes import register_routes
from reviewai.services.analyzer import ReviewAnalysisService
from reviewai.services.model_adapter import WorkspaceModelAdapter


def create_app(
    settings: Settings | None = None,
    analysis_service: ReviewAnalysisService | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)

    app = FastAPI(
        title="reviewAI",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
    )
    app.state.settings = resolved_settings
    app.state.analysis_service = analysis_service or ReviewAnalysisService(
        settings=resolved_settings,
        adapter=WorkspaceModelAdapter(resolved_settings),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    add_request_context_middleware(app)
    add_logging_middleware(app)
    register_exception_handlers(app)
    register_routes(app)

    if resolved_settings.eager_load_model:
        app.state.analysis_service.adapter.analyze_review("The food was good.")

    return app


app = create_app()
