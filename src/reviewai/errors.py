"""Application errors and exception handlers."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class ReviewAIError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, details: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or []


class ConfigurationError(ReviewAIError):
    status_code = 503
    code = "configuration_error"


class ModelUnavailableError(ReviewAIError):
    status_code = 503
    code = "model_unavailable"


def _error_payload(
    *,
    code: str,
    message: str,
    details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or [],
        }
    }


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ReviewAIError)
    async def handle_reviewai_error(_: Request, exc: ReviewAIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(
                code=exc.code,
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = []
        for error in exc.errors():
            details.append(
                {
                    "field": ".".join(str(part) for part in error["loc"] if part != "body"),
                    "message": error["msg"],
                    "type": error["type"],
                }
            )
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                code="validation_error",
                message="Invalid request body.",
                details=details,
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=_error_payload(
                code="internal_error",
                message="Unexpected server error.",
                details=[{"message": str(exc)}],
            ),
        )
