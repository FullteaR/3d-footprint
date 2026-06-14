"""API routes."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from ..core.export import export_mesh
from ..core.gpx import expand_bbox, parse_gpx
from ..core.mesh import MeshParams, terrain_solid
from ..core.terrain import fetch_elevation_grid

router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/generate")
def generate(
    file: UploadFile = File(...),
    size_mm: float = Form(120.0),
    vertical_scale: float = Form(3.0),
    base_thickness_mm: float = Form(3.0),
    dem_zoom: int = Form(14),
    grid_max: int = Form(400),
    fmt: str = Form("stl"),
) -> Response:
    """GPX -> terrain solid -> printable file (M2: terrain only, single color)."""
    try:
        track = parse_gpx(file.file.read())
        bbox = expand_bbox(track.bbox)
        grid = fetch_elevation_grid(bbox, zoom=dem_zoom, grid_max=grid_max)
        mesh = terrain_solid(
            grid,
            MeshParams(
                size_mm=size_mm,
                vertical_scale=vertical_scale,
                base_thickness_mm=base_thickness_mm,
            ),
        )
        data, content_type, ext = export_mesh(mesh, fmt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return Response(
        content=data,
        media_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="footprint.{ext}"'},
    )
