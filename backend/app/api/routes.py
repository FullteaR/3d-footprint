"""API routes."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from ..core.bridges import PlateauBridgeProvider
from ..core.buildings import PlateauBuildingProvider
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
    include_buildings: bool = Form(False),
    building_scale: float = Form(1.0),
    landuse: bool = Form(False),
    landuse_smooth_m: float = Form(60.0),
    terrain_color: str = Form("#c2b280"),
    track_color: str = Form("#dc4628"),
    building_color: str = Form("#b0b0b0"),
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

        cat_grid = None
        if landuse:
            # PLATEAU vs KSJ regime (affects only how much the categories are
            # smoothed) is decided inside the resolver; border straightening in
            # terrain_solid then keeps straight edges straight regardless.
            cat_grid, _ = resolve_category_grid(grid, landuse_smooth_m)
        # smoothness 0 => no naturalisation at all (raw grid-cell squares).
        bodies: list[Body] = terrain_solid(
            proj, cat_grid, naturalize=landuse_smooth_m > 0
        )
        if include_buildings:
            # Bridges/elevated structures share the buildings toggle and colour
            # layer; they differ only in placement (kept at their real elevation
            # above the relief rather than snapped onto the surface).
            building_body = PlateauBuildingProvider().building_body(proj, building_scale)
            if building_body is not None:
                bodies.append(building_body)
            bridge_body = PlateauBridgeProvider().bridge_body(proj)
            if bridge_body is not None:
                bodies.append(bridge_body)
        if include_track:
            bodies.append(Body(track_ridge(track, proj, track_width_mm, track_height_mm), "track"))

        # "terrain" label (land-use off) maps to the user's terrain color.
        colors = {
            "terrain": terrain_color, "track": track_color,
            "building": building_color,
        }
        data, content_type, ext = export_bodies(bodies, fmt, colors)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return Response(
        content=data,
        media_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="footprint.{ext}"'},
    )
