"""API routes. M1 only exposes health + a stub for /generate."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/generate")
def generate() -> JSONResponse:
    # Placeholder until M2 (DEM -> terrain mesh -> 3MF/STL).
    return JSONResponse(
        status_code=501,
        content={"detail": "not implemented yet (M2: DEM -> terrain mesh)"},
    )
