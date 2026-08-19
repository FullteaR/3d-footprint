"""Printability "デフォルメ" (massing) helpers shared by buildings and bridges.

At print scale a PLATEAU LOD2 feature's fine detail (roofs, girders, railings)
falls below the FDM nozzle width and collapses. Rather than print the raw
geometry we reduce each feature to a clean, watertight *footprint prism* — its
outline extruded between two heights — while enforcing a minimum printable
feature width so nothing is left thinner than the nozzle can resolve.

This is the massing for a model zoomed far enough out that a building has no
printable shape of its own; `keeps_its_shape` decides that, and where a feature
does still have one, voxel.py keeps it instead.

Two paths, differing in what they do with a feature *smaller* than that width.
A bridge goes through `printable`, which drops it as noise — a stray speck of
deck is not a bridge. A building cannot: below about 1:20,000 nearly every real
building is smaller than one nozzle square, so dropping them would leave an
empty city. Buildings instead go through `outline_parts` and are made printable
only once `blocks_of` has merged them with their neighbours (see buildings.py).

The 2D footprint algebra (union / simplify / buffer / contains) uses shapely;
the prism is built with the same earcut + mirror-base + perimeter-wall pattern
the terrain and building pipelines already use (`mesh.terrain_solid`,
`buildings._triangulate`), so no extra triangulation backend is required.
"""
from __future__ import annotations

import mapbox_earcut as earcut
import numpy as np
import shapely
import trimesh
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient


# Ground metres one nozzle covers, past which a building is narrower than the
# line the printer can lay down and has no shape of its own left to print. A
# typical Japanese building is about ten metres across.
SHAPE_LIMIT_M = 10.0


def keeps_its_shape(proj, min_feature: float) -> bool:
    """Whether a feature is still big enough for its own shape to be worth printing.

    There are two massings and the scale picks between them, not taste. Zoomed
    in, a building spans several nozzle widths and everything about it that is
    not its footprint — a podium, a setback, a tower's splayed legs — can be
    laid down, so the geometry is kept and thickened in three dimensions
    (voxel.py). Zoomed out, one nozzle is tens of metres of ground: a Tokyo
    building is ten metres across and ten metres from its neighbour, so its
    outline, its height and the street beside it are all finer than a single
    printed line. Nothing of its own shape can survive that, and resampling it
    into a lattice only turns a city block into a lump of porridge. What prints
    crisply there is the merged block itself — an exact polygon with straight
    edges and a flat top, which is what this module builds.

    Both are the same bargain the rest of the massing makes; they differ only in
    what is left worth keeping once the nozzle has had its say.
    """
    if min_feature <= 0:
        return True                      # no nozzle to be narrower than
    return min_feature / proj.scale <= SHAPE_LIMIT_M


def footprint_of(xy: np.ndarray, faces: np.ndarray):
    """Union a feature's triangles (in print mm) into a 2D footprint polygon.

    `xy` is (N,2) vertex coordinates and `faces` (M,3) indices into it. Returns a
    shapely (Multi)Polygon, or None if nothing has area. Degenerate (collinear)
    triangles are dropped so shapely never sees a zero-area ring.
    """
    if len(faces) == 0:
        return None
    tris = xy[faces]                                  # (m,3,2)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - \
            (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    tris = tris[np.abs(area2) > 1e-9]
    if len(tris) == 0:
        return None
    polys = shapely.polygons(shapely.linearrings(tris))  # one triangle each
    fp = shapely.union_all(polys)
    return fp if not fp.is_empty else None


def polygon_parts(geom) -> list:
    """The polygonal parts of any geometry (a clip can also yield lines/points)."""
    if isinstance(geom, Polygon):
        return [geom] if not geom.is_empty and geom.area > 0 else []
    return [p for g in getattr(geom, "geoms", []) for p in polygon_parts(g)]


def _shaped(poly: Polygon, h: float) -> Polygon:
    """Drop sub-nozzle detail: simplify, then close (dilate then erode).

    mitre joins keep building corners square (blocky) rather than rounding them.
    """
    poly = poly.simplify(h, preserve_topology=True)
    return poly.buffer(h, join_style="mitre", mitre_limit=2.0).buffer(
        -h, join_style="mitre", mitre_limit=2.0
    )


def _widened(poly: Polygon, h: float) -> Polygon:
    """Grow a polygon that is thinner than the nozzle *everywhere* out to it."""
    if poly.buffer(-h).is_empty:
        return poly.buffer(h, join_style="mitre", mitre_limit=2.0)
    return poly


def outline_parts(geom, min_feature: float) -> list:
    """A footprint's separate parts, with wiggle finer than the nozzle removed.

    The building path's first step, and deliberately the *only* thing done to a
    building on its own: it is neither dropped for being small nor grown to
    printable width where it stands. At city-model scale one nozzle width can be
    a hundred metres of ground, so growing each building in place fuses it with
    everything across the street — that is what flattens a low-rise ward into a
    slab. A building is left at its true size here and made printable as part of
    the block it merges into (`blocks_of`).
    """
    if geom is None or geom.is_empty:
        return []
    return polygon_parts(geom.simplify(0.5 * min_feature, preserve_topology=True))


def blocks_of(polys, min_feature: float) -> list:
    """Merge footprints into city blocks, keeping the streets that can print.

    A morphological *closing* at the nozzle radius: dilate every footprint by
    half `min_feature`, union, then erode the union by the same amount. The
    dilation is what fuses neighbours — anything closer than one nozzle width
    becomes a single polygon and stays one, which is how alleys and lot lines
    disappear. The erosion is what gives the streets back: a gap too wide to be
    bridged reopens at its **true** width, where a bare dilation would leave it
    narrowed by a whole nozzle and print as a crack, or as nothing.

    So the line between a street and an alley is not a matter of taste — it is
    exactly the width the nozzle can lay down. Whatever is wider survives.

    Blocks that come out thinner than the nozzle everywhere are then grown out
    to it, so a lone house on its own plot is still printable.
    """
    polys = np.asarray(polys, dtype=object)
    if len(polys) == 0:
        return []
    h = 0.5 * min_feature
    grown = shapely.buffer(polys, h, join_style="mitre", mitre_limit=2.0)
    closed = shapely.union_all(grown).buffer(-h, join_style="mitre", mitre_limit=2.0)
    return [_widened(p, h) for p in polygon_parts(closed)]


def printable(geom, min_feature: float, clip: Polygon | None = None):
    """Simplify a footprint and guarantee no part is thinner than `min_feature`.

    Drops sub-feature noise, dissolves a feature's own slivers/notches, and
    grows any everywhere-thin feature out to the minimum width. Returns a
    (Multi)Polygon or None. Each component is handled on its own so a thin
    outbuilding next to a fat one is thickened without bloating the fat one.

    `clip` is the model outline in print mm, and it is applied **last**: the
    widening above is exactly what pushes a feature past the model edge, so
    trimming any earlier would let it grow straight back out. Nothing this
    returns ever leaves the outline; a feature straddling it comes back cut
    flush, or dropped if the cut leaves less than one printable feature.
    """
    if geom is None or geom.is_empty:
        return None
    h = 0.5 * min_feature
    out = []
    for poly in getattr(geom, "geoms", [geom]):
        poly = _shaped(poly, h)
        if poly.is_empty or poly.area < min_feature * min_feature:
            continue                                  # sub-feature noise
        poly = _widened(poly, h)
        if clip is None:
            out.extend(polygon_parts(poly))
            continue
        # Cut flush with the model edge. A feature that only grazes the outline
        # is left as an unprintable nub hanging off it, so hold what survives the
        # cut to the same noise floor as the uncut footprint above.
        out.extend(p for p in polygon_parts(poly.intersection(clip))
                   if p.area >= min_feature * min_feature)
    if not out:
        return None
    return out[0] if len(out) == 1 else MultiPolygon(out)


def _ring_xy(ring) -> np.ndarray:
    """LinearRing -> (k,2) vertices with the repeated closing point dropped."""
    return np.asarray(ring.coords, dtype=np.float64)[:-1, :2]


def _extrude_polygon(poly: Polygon, z_bottom: float, z_top: float):
    """One simple Polygon (with optional holes) -> watertight prism mm mesh."""
    poly = orient(poly, 1.0)                           # exterior CCW, holes CW
    rings = [_ring_xy(poly.exterior)] + [_ring_xy(r) for r in poly.interiors]
    rings = [r for r in rings if len(r) >= 3]
    if not rings:
        return None
    pts2d = np.vstack(rings)
    ring_ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    try:
        idx = earcut.triangulate_float64(np.ascontiguousarray(pts2d), ring_ends)
    except Exception:
        return None
    if len(idx) < 3:
        return None
    top_tris = np.asarray(idx, np.int64).reshape(-1, 3)
    n = len(pts2d)

    verts = np.vstack([
        np.column_stack([pts2d, np.full(n, z_top)]),       # 0..n-1  top
        np.column_stack([pts2d, np.full(n, z_bottom)]),    # n..2n-1 base
    ])
    walls = []
    start = 0
    for r in rings:
        k = len(r)
        i = start + np.arange(k)
        j = start + (np.arange(k) + 1) % k
        walls.append(np.column_stack([i, j, j + n]))       # two tris per edge
        walls.append(np.column_stack([i, j + n, i + n]))
        start += k
    faces = np.vstack([top_tris, top_tris[:, ::-1] + n, *walls])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.merge_vertices()
    mesh.fix_normals()                                 # outward (small prism, cheap)
    return mesh


def prism(geom, z_bottom: float, z_top: float):
    """Extrude any (Multi)Polygon between two z planes into one watertight mesh."""
    if geom is None or geom.is_empty or z_top - z_bottom <= 1e-6:
        return None
    parts = []
    for p in getattr(geom, "geoms", [geom]):
        m = _extrude_polygon(p, z_bottom, z_top)
        if m is not None:
            parts.append(m)
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else trimesh.util.concatenate(parts)
