"""API routes."""
from __future__ import annotations

from typing import NamedTuple

import trimesh
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from shapely.geometry import box
from shapely.ops import transform

from ..core.bridges import PlateauBridgeProvider
from ..core.buildings import PlateauBuildingProvider
from ..core.export import Body, export_bodies
from ..core.gpx import (
    clip_track, expand_bbox, parse_bbox_param, parse_gpx,
    parse_time_range_param, trim_track,
)
from ..core.coloring import category_grid, generalize
from ..core.mesh import MeshParams, make_projection, terrain_solid
from ..core.nameplate import (
    Levels, clamp_side, inset_ink, inset_plate_outline, nameplate_bodies,
    plate_base, plate_ink, plate_levels, plate_to_print, to_plate_frame,
)
from ..core.region import (
    Region, clip_track_to_polygon, parse_lonlat_param, parse_rotation_param,
    parse_shape_param,
)
from ..core.stamp import ascii_credit, engrave_credit, formal_credit
from ..core.terrain import fetch_elevation_grid
from ..core.track import track_ridge

router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


class PlatePlan(NamedTuple):
    """A 銘板 worked out ahead of the terrain it is let into."""
    ink: object       # artwork, plate frame
    tile: object      # the plate under it, plate frame
    at: tuple         # (x, y, deg) placing the plate frame in the print frame
    levels: Levels    # the z planes it occupies
    footprint: object # the tile in lon/lat — where the map gets routed out


def _plate_plan(
    svg: bytes, center: str, width_mm: float, depth_mm: float, rotation_deg: float,
    relief_mm: float, base_thickness_mm: float, region: Region, proj,
) -> PlatePlan | None:
    """Resolve the nameplate's geometry and the pocket it needs in the map.

    Runs before anything is built: the plaque is a plate let into the model,
    so the terrain has to know about it. The plate frame is the one thing that
    stays axis-aligned — the model outline is brought into it, and only the
    footprint goes back out (through the region's rotation) to reach the DEM.
    """
    if not svg.strip():
        return None
    if not center:
        raise ValueError("銘板の位置が指定されていません")
    plon, plat = parse_lonlat_param(center)
    cx, cy = region.print_xy(float(proj.x_of(plon)), float(proj.y_of(plat)), proj)
    at = (float(cx), float(cy), parse_rotation_param(rotation_deg))

    # The plain-rect path never rotates, so its model is the fetched grid
    # itself (which can snap a hair wider than the requested bbox).
    outline_mm = (
        box(0.0, 0.0, float(proj.x_of(proj.grid.lons.max())),
            float(proj.y_of(proj.grid.lats.max())))
        if region.is_plain else region.outline_print_mm(proj)
    )
    model = to_plate_frame(outline_mm.buffer(-0.2), at)
    ink = plate_ink(svg, inset_plate_outline(
        clamp_side(width_mm), clamp_side(depth_mm), model
    ))
    tile = plate_base(ink, model)   # the tile is the ink's box, so trim after
    ink = inset_ink(ink, tile)

    def to_lonlat(x, y):
        mx, my = region.map_xy(x, y, proj)
        return proj.lon_of(mx), proj.lat_of(my)

    footprint = transform(to_lonlat, plate_to_print(tile, at))
    z_lo, _ = proj.z_range_under(footprint)
    levels = plate_levels(
        z_lo, relief_mm,
        # The tile has to stay inside the base — a plaque poking out of the
        # underside would hold the print off the bed. Leave a little material
        # under it too, but only out of a base thick enough to spare it.
        z_floor=min(0.4, max(0.0, base_thickness_mm - 0.6)) - base_thickness_mm,
    )
    return PlatePlan(ink, tile, at, levels, footprint)


def _structure_clip(clip_mm, plate: PlatePlan | None, proj, grid):
    """The outline buildings and bridges are cut to, minus the 銘板.

    Nothing stands on the plaque. The map under it has just been routed out
    for the plate, so a block or a bridge deck left there would rise out of
    the pocket and bury the artwork — and the plate is the one thing on the
    model the reader has to be able to read. A user can place the plate clear
    of their own track; they cannot place it clear of the city, so it is the
    structures that give way. What straddles the edge is cut flush with it,
    the same as at the model's own outline.
    """
    if plate is None:
        return clip_mm
    if clip_mm is None:            # plain rect: the fetched grid is the outline
        clip_mm = box(
            float(proj.x_of(grid.lons.min())), float(proj.y_of(grid.lats.min())),
            float(proj.x_of(grid.lons.max())), float(proj.y_of(grid.lats.max())),
        )
    return clip_mm.difference(
        transform(lambda x, y: (proj.x_of(x), proj.y_of(y)), plate.footprint)
    )


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
    min_color_mm: float = Form(0.0),
    terrain_color: str = Form("#c2b280"),
    track_color: str = Form("#dc4628"),
    building_color: str = Form("#b0b0b0"),
    dem_zoom: int = Form(15),
    grid_max: int = Form(1000),
    bbox: str = Form(""),
    time_range: str = Form(""),
    shape: str = Form("rect"),
    rotation_deg: float = Form(0.0),
    plate_svg: UploadFile | None = File(None),
    plate_center: str = Form(""),
    plate_width_mm: float = Form(40.0),
    plate_depth_mm: float = Form(16.0),
    plate_rotation_deg: float = Form(0.0),
    plate_relief_mm: float = Form(0.6),
    fmt: str = Form("stl"),
) -> Response:
    """GPX -> terrain solid (+ land-use color, + track ridge) -> printable file.

    Land-use colouring is always applied where data covers the area; only its
    printed detail is tunable (`min_color_mm`, 0 = paint the raw cells).

    `time_range` ("start,end" in epoch seconds) trims the GPX to one leg of the
    outing before anything else: the automatic extent follows the trimmed track.

    `bbox` ("min_lon,min_lat,max_lon,max_lat") overrides the automatic
    track-plus-margin extent. `shape` (rect / square / hex) and `rotation_deg`
    (CCW) turn that bbox into the print outline: a square or regular hexagon is
    inscribed in the bbox, the outline is rotated about its centre on the map,
    everything is clipped to it, and the finished model is rotated back so the
    outline prints axis-aligned.

    A `plate_svg` upload adds a 銘板 with the SVG artwork raised on top
    (auto-fitted; text must be outlined to paths by the design tool): a flat
    pad on the map itself at `plate_center` ("lon,lat"), `plate_width_mm` x
    `plate_depth_mm` and turned by `plate_rotation_deg` (CCW *relative to the
    model outline*, so rotating the model carries the plate with it).
    """
    try:
        track = parse_gpx(file.file.read())
        if time_range:
            track = trim_track(track, *parse_time_range_param(time_range))
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
            clip_ll = region.polygon_lonlat()   # for the track (lon/lat)
            clip_mm = region.polygon_mm(proj)   # for terrain / buildings / bridges

        # 銘板: worked out before the terrain is built, because the map has to
        # be routed out under it — the plaque is a plate let into a pocket, not
        # something sitting on the surface. The bodies themselves come last
        # (they are built straight in the print frame, so they must not be
        # caught by the rotation below).
        plate = _plate_plan(
            plate_data, plate_center, plate_width_mm, plate_depth_mm,
            plate_rotation_deg, plate_relief_mm, base_thickness_mm, region, proj,
        )
        if plate is not None:
            proj.flatten_under(plate.footprint, plate.levels.pocket)
        structure_clip = _structure_clip(clip_mm, plate, proj, grid)

        # PLATEAU luse painted as-is; JAXA HRLULC fills only the cells PLATEAU
        # doesn't classify; the rest stays the terrain colour. None when neither
        # source covers the area — then the model is plain terrain-coloured and
        # the credit must not name land-use sources it never used.
        smooth_colors = False
        cat_grid = category_grid(grid)
        landuse = cat_grid is not None
        if landuse and min_color_mm > 0:
            # Colour detail is set in print mm, independently of the terrain's:
            # one DEM cell is a fraction of a mm, far under what a multi-material
            # printer resolves, so anything finer than `min_color_mm` is
            # dissolved before the mesh is cut.
            cell_mm = min(
                float(proj.x_of(grid.lons[1]) - proj.x_of(grid.lons[0])),
                float(proj.y_of(grid.lats[1]) - proj.y_of(grid.lats[0])),
            )
            if cell_mm > 0:
                cat_grid = generalize(cat_grid, min_color_mm / cell_mm)
                smooth_colors = True
        # Generalised colours get the contour cut too: dissolving fine features
        # is only half the job, the borders still have to leave the grid (cells
        # cut along a marching-squares polyline, then straightened) or the
        # colours read as a mosaic of squares. Painting as-is (the slider at 0)
        # keeps the raw cell edges, detail and all.
        bodies: list[Body] = terrain_solid(
            proj, cat_grid, naturalize=smooth_colors, clip=clip_mm
        )
        if include_buildings:
            # Bridges/elevated structures share the buildings toggle and colour
            # layer; both are massed into printable blocks (min_feature_mm sets
            # the minimum printable width), differing only in placement: buildings
            # sit on the surface, bridges keep their real deck elevation + pillars.
            building_body = PlateauBuildingProvider().building_body(
                proj, building_scale, min_feature_mm, clip=structure_clip
            )
            if building_body is not None:
                bodies.append(building_body)
            bridge_body = PlateauBridgeProvider().bridge_body(
                proj, min_feature_mm, clip=structure_clip
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

        # In the print frame the model's front edge lies on y=0; the plate and
        # the credit stamp span it (a hexagon's flat bottom edge is its middle
        # half).
        if region.is_plain:
            px0, px1 = 0.0, float(proj.x_of(grid.lons.max()))
        else:
            hw, _ = region.half_extents_m
            w = 2.0 * hw * proj.scale
            px0, px1 = (0.25 * w, 0.75 * w) if region.shape == "hex" else (0.0, w)

        # 出典刻印 (always on — the data licenses require attribution): the
        # formal 出典 sentence debossed on the underside travels with the
        # print itself; the same sentence rides in the file metadata.
        engrave_credit(
            bodies, landuse, include_buildings, px0, px1, base_thickness_mm,
        )

        if plate is not None:
            bodies += nameplate_bodies(plate.ink, plate.tile, plate.at, plate.levels)

        # "terrain" label (land-use off) maps to the user's terrain color; the
        # nameplate tile follows it so only the artwork stands out (in the
        # fixed label colour from DEFAULT_COLORS).
        colors = {
            "terrain": terrain_color, "track": track_color,
            "building": building_color, "plate": terrain_color,
        }
        data, content_type, ext = export_bodies(
            bodies, fmt, colors,
            credit_full=formal_credit(landuse, include_buildings),
            credit_ascii=ascii_credit(landuse, include_buildings),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return Response(
        content=data,
        media_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="footprint.{ext}"'},
    )
