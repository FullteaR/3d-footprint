"""API routes."""
from __future__ import annotations

import shapely
import trimesh
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from shapely.geometry import box

from ..core.bridges import PlateauBridgeProvider
from ..core.buildings import PlateauBuildingProvider
from ..core.export import Body, export_bodies
from ..core.gpx import clip_track, expand_bbox, parse_bbox_param, parse_gpx
from ..core.coloring import category_grid
from ..core.mesh import MeshParams, make_projection, terrain_solid
from ..core.nameplate import nameplate_bodies
from ..core.region import (
    Region, clip_track_to_polygon, parse_rotation_param, parse_shape_param,
)
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
    min_feature_mm: float = Form(0.8),
    landuse: bool = Form(False),
    terrain_color: str = Form("#c2b280"),
    track_color: str = Form("#dc4628"),
    building_color: str = Form("#b0b0b0"),
    dem_zoom: int = Form(15),
    grid_max: int = Form(1000),
    bbox: str = Form(""),
    shape: str = Form("rect"),
    rotation_deg: float = Form(0.0),
    plate_svg: UploadFile | None = File(None),
    plate_depth_mm: float = Form(16.0),
    plate_relief_mm: float = Form(0.6),
    label_color: str = Form("#333333"),
    fmt: str = Form("stl"),
) -> Response:
    """GPX -> terrain solid (+ land-use color, + track ridge) -> printable file.

    `bbox` ("min_lon,min_lat,max_lon,max_lat") overrides the automatic
    track-plus-margin extent. `shape` (rect / square / hex) and `rotation_deg`
    (CCW) turn that bbox into the print outline: a square or regular hexagon is
    inscribed in the bbox, the outline is rotated about its centre on the map,
    everything is clipped to it, and the finished model is rotated back so the
    outline prints axis-aligned.

    A `plate_svg` upload adds a 銘板: a slab hanging off the model's front
    (south) edge with the SVG artwork raised on top (auto-fitted; text must
    be outlined to paths by the design tool).
    """
    try:
        track = parse_gpx(file.file.read())
        plate_data = plate_svg.file.read() if plate_svg is not None else b""
        area = parse_bbox_param(bbox) if bbox else expand_bbox(track.bbox)
        region = Region(
            bbox=area,
            shape=parse_shape_param(shape),
            rotation_deg=parse_rotation_param(rotation_deg),
        )
        grid = fetch_elevation_grid(region.fetch_bbox(), zoom=dem_zoom, grid_max=grid_max)
        proj = make_projection(
            grid,
            MeshParams(
                size_mm=size_mm,
                vertical_scale=vertical_scale,
                base_thickness_mm=base_thickness_mm,
            ),
            # size_mm applies to the requested outline itself (not the fetched
            # grid, which can snap a hair wider at DEM pixel edges) so the
            # frontend's live scale readout is exactly the printed scale.
            span_m=region.span_m,
        )
        clip_ll = clip_mm = None
        if not region.is_plain:
            clip_ll = region.polygon_lonlat()
            shapely.prepare(clip_ll)
            clip_mm = region.polygon_mm(proj)

        cat_grid = None
        if landuse:
            # PLATEAU luse painted as-is; JAXA HRLULC fills only the cells
            # PLATEAU doesn't classify; the rest stays the terrain colour.
            cat_grid = category_grid(grid)
        bodies: list[Body] = terrain_solid(proj, cat_grid, naturalize=False, clip=clip_mm)
        if include_buildings:
            # Bridges/elevated structures share the buildings toggle and colour
            # layer; both are massed into printable blocks (min_feature_mm sets
            # the minimum printable width), differing only in placement: buildings
            # sit on the surface, bridges keep their real deck elevation + pillars.
            building_body = PlateauBuildingProvider().building_body(
                proj, building_scale, min_feature_mm, clip=clip_ll
            )
            if building_body is not None:
                bodies.append(building_body)
            bridge_body = PlateauBridgeProvider().bridge_body(
                proj, min_feature_mm, clip=clip_ll
            )
            if bridge_body is not None:
                bodies.append(bridge_body)
        if include_track:
            # Clip to the terrain actually built (a custom outline may cut the
            # track) so the ridge never overhangs the base; each in-area piece
            # becomes its own sweep. Keep the ridge at least one nozzle wide
            # so it does not split.
            grid_extent = (
                float(grid.lons.min()), float(grid.lats.min()),
                float(grid.lons.max()), float(grid.lats.max()),
            )
            if region.is_plain:
                segments = clip_track(track, grid_extent)
            else:
                # Outline ∩ fetched grid: the outline can poke a sub-pixel past
                # the DEM crop at the corners that touch the fetch bbox.
                segments = clip_track_to_polygon(
                    track, clip_ll.intersection(box(*grid_extent))
                )
            ridges = []
            for seg in segments:
                try:
                    ridges.append(track_ridge(
                        seg, proj, max(track_width_mm, min_feature_mm), track_height_mm
                    ))
                except ValueError:
                    pass  # sliver that only grazes the border: nothing to sweep
            if ridges:
                bodies.append(Body(trimesh.util.concatenate(ridges), "track"))

        # Rotate the scene back so the outline prints axis-aligned at origin.
        region.to_print_frame(bodies, proj)

        if plate_data.strip():
            # In the print frame the model's front edge lies on y=0; the plate
            # spans it (a hexagon's flat bottom edge is its middle half).
            if region.is_plain:
                px0, px1 = 0.0, float(proj.x_of(grid.lons.max()))
            else:
                hw, _ = region.half_extents_m
                w = 2.0 * hw * proj.scale
                px0, px1 = (0.25 * w, 0.75 * w) if region.shape == "hex" else (0.0, w)
            bodies += nameplate_bodies(
                plate_data, px0, px1, 0.0,
                base_thickness_mm, plate_depth_mm, plate_relief_mm,
            )

        # "terrain" label (land-use off) maps to the user's terrain color;
        # the nameplate slab follows it so only the lettering stands out.
        colors = {
            "terrain": terrain_color, "track": track_color,
            "building": building_color,
            "plate": terrain_color, "label": label_color,
        }
        data, content_type, ext = export_bodies(bodies, fmt, colors)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return Response(
        content=data,
        media_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="footprint.{ext}"'},
    )
