"""PLATEAU LOD2/LOD1 buildings -> printable solids sitting on the terrain.

Source: PLATEAU CityGML `bldg` files (one per 3rd-level mesh, 8-digit code,
*per municipality* — a border mesh has one file per city, holding either just
that city's buildings or the whole mesh duplicated, dataset-dependent),
resolved from a bbox via the data-catalog API. All files are loaded and
identical content is rendered once.

Each building's best available LOD (LOD2 semantic surfaces
`bldg:boundedBy/{RoofSurface,WallSurface,GroundSurface}`, else the LOD1
`bldg:lod1Solid` prism) is parsed once and cached. For printing, fine roof/wall
detail is below the FDM nozzle and would collapse, so `building_body` does NOT
print the raw geometry. What it does instead depends on the scale, and the
scale alone (`massing.keeps_its_shape`).

Zoomed in, a building spans several nozzle widths and its own shape is worth
printing: the geometry is kept and thickened in three dimensions until nothing
is finer than `min_feature_mm` (voxel.py), so a podium stays a podium instead of
being extruded to the height of the tower on it, and Tokyo Tower comes out as
Tokyo Tower rather than as the 127 m column of its feet.

Zoomed out, it is not: each building is reduced to its **footprint** (the union
of its triangles in plan) and the footprints are merged into **city blocks** —
anything closer together than `min_feature_mm` becomes one flat-topped prism,
and anything further apart stays a gap, so the streets wide enough to print
survive as streets (massing.py).

Blocks, not individual buildings, because the scale demands it: a whole city on
a 120 mm plate is around 1:100,000, where one nozzle width is ~80 m of ground
and a typical Tokyo building (~60 m² footprint) is 1/100 of a printable speck.
Printing them separately is impossible — either they are dropped and the city
vanishes, or they are grown and overlap anyway. Merging is what an architect's
massing model of a city does, and it costs hundreds of prisms instead of tens
of thousands. Merging is confined to a height class (HEIGHT_CLASSES_M) so a
tower is not averaged away into the low-rise crust it stands in; a block's
height is the footprint-weighted mean of its members x a `height_scale` knob,
with a floor so short blocks still read.

All faces share a single "building" colour; a block's base is embedded so it
fuses to the terrain surface, and every block is trimmed to the model outline —
one that straddles the edge is cut flush with it rather than dropped, and none
overhangs the printed edge.

Polygons (lat/lon/height, EPSG 6697; height is 標高 T.P., same datum as the
GSI DEM) are triangulated once and cached per mesh as a compact npz in
geographic coordinates. At request time they are projected into the print's
millimetre space and massed onto the terrain surface, so the heavy parse runs
only on the first use of an area.
"""
from __future__ import annotations

import hashlib

import numpy as np
import requests
import shapely
import trimesh
from lxml import etree

from . import citygml, plateau
from .export import Body
from .net import atomic_savez
from .massing import (
    blocks_of, footprint_of, keeps_its_shape, outline_parts, polygon_parts, prism,
)
from .voxel import solid_from
from .mesh import Projection

EMBED_MM = 0.5            # how far building block bases sink into the terrain
MIN_H_MM = 0.6           # minimum block height so even short buildings read
# Touching footprints merge into one block only within a height class, so a
# tower keeps its own block instead of being averaged into the low-rise crust
# around it. The breaks are the usual Japanese massing classes: 低層 (~3階),
# 中層, the old 31 m absolute height limit, and 超高層 (60 m and 120 m).
HEIGHT_CLASSES_M = (12.0, 31.0, 60.0, 120.0)

_BLDG_NS = "http://www.opengis.net/citygml/building/2.0"
_BUILDING_TAG = f"{{{_BLDG_NS}}}Building"
_NS = {"bldg": _BLDG_NS}   # only the LOD1 lookup below needs a prefix
# Semantic surface -> label. Ground surfaces are kept (they cap the bottom so
# each building stays a closed solid) but labelled "wall" since they sit hidden
# below the terrain surface.
_SURFACE_LABEL = {"RoofSurface": "roof", "WallSurface": "wall",
                  "GroundSurface": "wall", "ClosureSurface": "wall",
                  "OuterCeilingSurface": "roof", "OuterFloorSurface": "wall"}
_LABELS = ("wall", "roof")  # ftype 0 = wall, 1 = roof


def _building_polygons(building: etree._Element):
    """Yield (label, exterior, holes) for one building (LOD2, else LOD1)."""
    surfaces = []  # (label, surface element)
    for bb in building.iter(f"{{{_BLDG_NS}}}boundedBy"):
        for child in bb:
            label = _SURFACE_LABEL.get(etree.QName(child).localname)
            if label is not None:
                surfaces.append((label, child))

    if surfaces:  # LOD2 semantic surfaces
        for label, surf in surfaces:
            for poly in surf.iter(citygml.POLYGON_TAG):
                yield (label, *citygml.rings(poly))
        return

    solid = building.find(".//bldg:lod1Solid", _NS)  # LOD1 fallback (flat prism)
    if solid is None:
        return
    polys = []
    for poly in solid.iter(citygml.POLYGON_TAG):
        ext, holes = citygml.rings(poly)
        if len(ext):
            polys.append((ext, holes))
    if not polys:
        return
    hmax = max(float(ext[:, 2].max()) for ext, _ in polys)
    for ext, holes in polys:
        flat = np.ptp(ext[:, 2]) < 0.1
        # Flat top -> roof; everything else (walls, the flat ground cap) -> wall.
        label = "roof" if flat and abs(ext[:, 2].mean() - hmax) < 0.1 else "wall"
        yield label, ext, holes


def _cache_path(mesh: str, url: str):
    return plateau.cache_file("buildings", mesh, url)


def _geometry(mesh: str, url: str):
    """Cached geographic geometry for one bldg GML.

    Returns (verts (N,3) lon/lat/h, faces (M,3), ftype (M,), vbid (N,)).
    Module-level so `plateau.distinct_files` can hand it to a parse worker
    by reference.
    """
    cache = _cache_path(mesh, url)
    if cache.is_file():
        d = np.load(cache)
        return d["verts"], d["faces"], d["ftype"], d["vbid"]

    lat_mid = citygml.mesh_lat_mid(mesh)
    all_v, all_f, all_t, all_b = [], [], [], []
    voff = bid = 0
    try:
        for b in citygml.stream_features(url, _BUILDING_TAG):
            started = voff
            for label, ext, holes in _building_polygons(b):
                if len(ext) < 3:
                    continue
                tri = citygml.triangulate(ext, holes, lat_mid)
                if tri is None:
                    continue
                pts, faces = tri
                all_v.append(pts)
                all_f.append(faces + voff)
                all_t.append(np.full(len(faces), _LABELS.index(label), np.uint8))
                voff += len(pts)
            if voff > started:
                all_b.append(np.full(voff - started, bid, np.int32))
                bid += 1
    except (requests.RequestException, OSError, ValueError):
        return None

    if not all_v:
        verts = np.empty((0, 3), np.float32)
        faces = np.empty((0, 3), np.int32)
        ftype = np.empty(0, np.uint8)
        vbid = np.empty(0, np.int32)
    else:
        verts = np.vstack(all_v).astype(np.float32)
        faces = np.vstack(all_f).astype(np.int32)
        ftype = np.concatenate(all_t)
        vbid = np.concatenate(all_b)
    atomic_savez(cache, verts=verts, faces=faces, ftype=ftype, vbid=vbid)
    return verts, faces, ftype, vbid


def _content_key(geo) -> bytes | None:
    """What makes one file's buildings distinct, or None when it has none.

    A border mesh's files are either city-partitioned (each city only its own
    buildings) or the identical mesh-wide content duplicated per city (2025
    pref datasets): keep every distinct file, render identical content once.
    """
    verts, faces = geo[0], geo[1]
    if len(verts) == 0:
        return None
    return hashlib.sha1(verts.tobytes() + faces.tobytes()).digest()


class PlateauBuildingProvider:
    """PLATEAU LOD2/LOD1 building provider. Covers PLATEAU cities only."""

    def building_body(
        self,
        proj: Projection,
        height_scale: float = 1.0,
        min_feature_mm: float = 0.8,
        clip: shapely.Polygon | None = None,
    ) -> Body | None:
        """One Body of every covered building, massed into city blocks.

        Each building is reduced to its footprint (the union of its triangles in
        plan); no building is ever lost for being small. Footprints closer than
        `min_feature_mm`, the minimum printable width, merge into one flat-topped
        block per height class, and any street wider than that stays a street
        rather than being paved over. `height_scale` exaggerates block height
        (1.0 = real-world proportion). Blocks are cut flush with the print
        outline, so the model edge reads as one clean slice through the city.
        `clip` (print mm) replaces the grid rectangle as that outline when the
        model is a rotated rect / hexagon.
        """
        grid = proj.grid
        urls = plateau.file_urls("bldg", plateau.mesh3_codes(grid.bbox))
        if not urls:
            return None

        verts, faces, vbid = [], [], []
        voff = boff = 0
        # roof/wall type unused: buildings are one colour.
        for v, f, _t, b in plateau.distinct_files(
            urls, cache_path=_cache_path, load=_geometry, key=_content_key
        ):
            verts.append(v)
            faces.append(f + voff)
            vbid.append(b + boff)
            voff += len(v)
            boff += int(b.max()) + 1 if len(b) else 0
        if not verts:
            return None

        verts = np.vstack(verts)
        faces = np.vstack(faces)
        vbid = np.concatenate(vbid)
        lon, lat, h = verts[:, 0], verts[:, 1], verts[:, 2]
        xy = np.column_stack([proj.x_of(lon), proj.y_of(lat)])  # print mm, for footprints

        # Per-building extents: ground/top elevation and centroid (for terrain snap).
        nb = int(vbid.max()) + 1
        ground = np.full(nb, np.inf)
        np.minimum.at(ground, vbid, h)
        top = np.full(nb, -np.inf)
        np.maximum.at(top, vbid, h)
        counts = np.bincount(vbid, minlength=nb)
        clon = np.bincount(vbid, lon, minlength=nb) / counts
        clat = np.bincount(vbid, lat, minlength=nb) / counts
        surface = proj.sample_z(clon, clat)  # terrain surface (mm) under each building

        # The print outline, in the same mm frame as the footprints: the model's
        # own (rotated rect / hexagon) or, by default, the fetched grid rectangle.
        if clip is None:
            clip = proj.grid_box()
        shapely.prepare(clip)

        # Keep every building that reaches into the print footprint at all: one
        # straddling the outline is cut flush with it by the massing trim below,
        # the same way a bridge crossing the boundary keeps its inside portion.
        inside = shapely.contains_xy(clip, xy[:, 0], xy[:, 1])
        keep_b = np.zeros(nb, bool)
        keep_b[vbid[inside]] = True

        # Zoomed in far enough that a building still has a shape of its own, it
        # is kept and thickened rather than flattened into the prism of its plan
        # (see massing.keeps_its_shape). Height is measured from the building's
        # own ground and re-based onto the terrain the model actually has, so it
        # sits on the relief rather than at whatever elevation PLATEAU recorded.
        if keeps_its_shape(proj, min_feature_mm):
            kept = faces[keep_b[vbid[faces[:, 0]]]]
            if len(kept) == 0:
                return None
            xyz = np.column_stack([
                xy[:, 0], xy[:, 1],
                surface[vbid] + (h - ground[vbid]) * proj.scale * height_scale,
            ])
            shaped = solid_from(xyz, kept, min_feature_mm, proj=proj,
                                embed=EMBED_MM, outline=clip)
            return None if shaped is None else Body(shaped, "building")

        # Group faces by building so each footprint is unioned independently.
        face_bid = vbid[faces[:, 0]]
        order = np.argsort(face_bid, kind="stable")
        faces_s = faces[order]
        bounds = np.searchsorted(face_bid[order], np.arange(nb + 1))

        # One footprint part per building, at its true size (a building modelled
        # as two detached wings contributes both, so each joins its own block).
        parts, cls, heights, surf = [], [], [], []
        for b in range(nb):
            if not keep_b[b]:
                continue
            fp = footprint_of(xy, faces_s[bounds[b]:bounds[b + 1]])
            outline = outline_parts(fp, min_feature_mm)
            if not outline:
                continue
            real_h = max(float(top[b] - ground[b]), 0.0)
            # Real height x exaggeration, floored so short blocks still read.
            # Only MIN_H_MM floors it: `min_feature_mm` is a *width* — the
            # nozzle — while height is resolved in layers, and flooring by it
            # would flatten the whole skyline to one slab at wide scales, which
            # is the one thing the block model is there to show.
            height = max(real_h * proj.scale * height_scale, MIN_H_MM)
            for poly in outline:
                parts.append(poly)
                cls.append(int(np.digitize(real_h, HEIGHT_CLASSES_M)))
                heights.append(height)
                surf.append(float(surface[b]))
        if not parts:
            return None

        meshes = []
        for poly, z_bottom, z_top in _blocks(
            parts, np.array(cls), np.array(heights), np.array(surf), min_feature_mm, clip
        ):
            block = prism(poly, z_bottom, z_top)
            if block is not None:
                meshes.append(block)
        if not meshes:
            return None

        mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
        return Body(mesh, "building")


def _blocks(parts, cls, heights, surf, min_feature_mm, clip):
    """Merge neighbouring footprints into city blocks; (polygon, z0, z1) in mm.

    `blocks_of` decides what merges: everything closer than one nozzle width,
    which erases alleys and lot lines but leaves any street wider than that as
    a real gap between blocks. Only footprints of the same height class merge,
    so a tower stands out of the crust rather than dissolving into it — a taller
    block simply overlaps the lower ones it neighbours, which prints as the step
    it should be.

    A block's height is its members' footprint-weighted mean: within one class
    that is the block's bulk rather than a compromise between a tower and a
    shed, and unlike a max it does not let one mis-modelled record raise a whole
    block. It spans from the lowest ground it covers (embedded) to the highest
    plus that height, so no part of it floats and none is buried in a slope.

    The clip is applied last, for the reason `printable` applies it last: a
    block that straddles the model edge comes back cut flush with it, and what
    survives the cut is held to the same noise floor, so the edge is not left
    with unprintable nubs hanging off it.
    """
    geoms = np.empty(len(parts), dtype=object)
    geoms[:] = parts
    areas = shapely.area(geoms)

    out = []
    for c in np.unique(cls):
        sel = np.flatnonzero(cls == c)
        merged = blocks_of(geoms[sel], min_feature_mm)
        if not merged:
            continue
        blocks = np.empty(len(merged), dtype=object)
        blocks[:] = merged
        shapely.prepare(blocks)  # the tested side of the predicate below

        # Which block each member landed in. A closing only ever adds area, so
        # every footprint still lies wholly inside exactly one of the blocks it
        # produced, and a single interior point places it.
        src, dst = shapely.STRtree(blocks).query(
            shapely.point_on_surface(geoms[sel]), predicate="intersects"
        )
        n = len(merged)
        weight = np.zeros(n)
        hsum = np.zeros(n)
        z_lo = np.full(n, np.inf)
        z_hi = np.full(n, -np.inf)
        w, h, s = areas[sel][src], heights[sel][src], surf[sel][src]
        np.add.at(weight, dst, w)
        np.add.at(hsum, dst, w * h)
        np.minimum.at(z_lo, dst, s)
        np.maximum.at(z_hi, dst, s)

        for k in range(n):
            if weight[k] <= 0:                        # no member placed: skip
                continue
            height = hsum[k] / weight[k]
            for piece in polygon_parts(merged[k].intersection(clip)):
                if piece.area >= min_feature_mm * min_feature_mm:
                    out.append((piece, z_lo[k] - EMBED_MM, z_hi[k] + height))
    return out
