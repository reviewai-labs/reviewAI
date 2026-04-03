"""Request logging middleware."""

from __future__ import annotations

from time import perf_counter

from fastapi import FastAPI, Request

from reviewai.logger import get_logger

LOGGER = get_logger("reviewai.http")


def add_logging_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):
        started_at = perf_counter()
        response = await call_next(request)
        duration_ms = (perf_counter() - started_at) * 1000
        LOGGER.info(
            "%s %s -> %s %.2fms",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response
