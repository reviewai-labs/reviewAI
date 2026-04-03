"""Server entrypoint for local development."""

from __future__ import annotations

import uvicorn

from reviewai.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "reviewai.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
