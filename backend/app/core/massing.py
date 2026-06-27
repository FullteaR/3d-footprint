"""Printability "デフォルメ" (massing) helpers shared by buildings and bridges.

At print scale a PLATEAU LOD2 feature's fine detail (roofs, girders, railings)
falls below the FDM nozzle width and collapses. Rather than print the raw
geometry we reduce each feature to a clean, watertight *footprint prism* — its
outline extruded between two heights — while enforcing a minimum printable
feature width so nothing is left thinner than the nozzle can resolve.

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


def printable(geom, min_feature: float):
    """Simplify a footprint and guarantee no part is thinner than `min_feature`.

    Drops sub-feature noise, dissolves a feature's own slivers/notches, and
    grows any everywhere-thin feature out to the minimum width. Returns a
    (Multi)Polygon or None. Each component is handled on its own so a thin
    outbuilding next to a fat one is thickened without bloating the fat one.
    """
    if geom is None or geom.is_empty:
        return None
    h = 0.5 * min_feature
    out = []
    for poly in getattr(geom, "geoms", [geom]):
        poly = poly.simplify(h, preserve_topology=True)
        # close: merge the feature's own sub-parts and erase notches narrower
        # than the nozzle. mitre joins keep building corners square (blocky).
        poly = poly.buffer(h, join_style="mitre", mitre_limit=2.0).buffer(
            -h, join_style="mitre", mitre_limit=2.0
        )
        if poly.is_empty or poly.area < min_feature * min_feature:
            continue                                  # sub-feature noise
        if poly.buffer(-h).is_empty:                  # thinner than min everywhere
            poly = poly.buffer(h, join_style="mitre", mitre_limit=2.0)
        for p in getattr(poly, "geoms", [poly]):
            if not p.is_empty and p.area > 0:
                out.append(p)
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
