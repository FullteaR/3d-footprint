"""API routes."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from ..core.export import Body, export_bodies
from ..core.gpx import expand_bbox, parse_gpx
from ..core.landuse import resolve_category_grid
from ..core.mesh import MeshParams, make_projection, terrain_solid
from ..core.terrain import fetch_elevation_grid
from ..core.track import track_ridge

router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/generate")
def generate(
    file: UploadFile = File(...),
    size_mm: float = Form(120.0),
    vertical_scale: float = Form(1.0),
    base_thickness_mm: float = Form(3.0),
    track_width_mm: float = Form(1.2),
    track_height_mm: float = Form(1.5),
    include_track: bool = Form(True),
    landuse: bool = Form(False),
    terrain_color: str = Form("#c2b280"),
    track_color: str = Form("#dc4628"),
    dem_zoom: int = Form(14),
    grid_max: int = Form(400),
    fmt: str = Form("stl"),
) -> Response:
    """GPX -> terrain solid (+ land-use color, + track ridge) -> printable file."""
    try:
        track = parse_gpx(file.file.read())
        bbox = expand_bbox(track.bbox)
        grid = fetch_elevation_grid(bbox, zoom=dem_zoom, grid_max=grid_max)
        proj = make_projection(
            grid,
            MeshParams(
                size_mm=size_mm,
                vertical_scale=vertical_scale,
                base_thickness_mm=base_thickness_mm,
            ),
        )

        cat_grid = resolve_category_grid(grid) if landuse else None
        terrain_mesh, terrain_labels = terrain_solid(proj, cat_grid)
        bodies: list[Body] = [Body(terrain_mesh, terrain_labels)]
        if include_track:
            bodies.append(Body(track_ridge(track, proj, track_width_mm, track_height_mm), "track"))

        # "terrain" label (land-use off) maps to the user's terrain color.
        colors = {"terrain": terrain_color, "track": track_color}
        data, content_type, ext = export_bodies(bodies, fmt, colors)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return Response(
        content=data,
        media_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="footprint.{ext}"'},
    )
