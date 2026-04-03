"""Tiny local demo UI."""

from __future__ import annotations

from importlib import resources

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter(include_in_schema=False)


@router.get("/")
async def root_redirect() -> RedirectResponse:
    return RedirectResponse(url="/playground", status_code=307)


@router.get("/playground", response_class=HTMLResponse)
async def playground() -> HTMLResponse:
    html = resources.files("reviewai").joinpath("static/playground.html").read_text(
        encoding="utf-8"
    )
    return HTMLResponse(content=html)
