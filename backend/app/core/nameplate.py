"""銘板 (nameplate): a lectern-wedge plaque on the model's front edge whose
face artwork comes from a user-uploaded SVG, extruded in relief.

The design itself is the user's job (Figma / Illustrator / Inkscape — any
tool that exports SVG with the text outlined to paths); this module only
turns that vector ink into printable geometry. svgelements parses the file
(shapes, transforms), each element's subpath rings are flattened to
polylines, filled regions are resolved under the element's actual fill rule
(nonzero / evenodd) by polygonizing the ring arrangement and keeping the
faces whose winding says "inside", and strokes become buffered outlines.
The combined ink is auto-fitted onto the plate face and extruded with the
manifold-safe prism builder. The slab continues the model base
(z in [-base_thickness, 0]) so it prints as one piece with the model; the
ink sits proud on top as its own colour layer ("label") so multi-material
prints get contrasting artwork.
"""
from __future__ import annotations

import math
from io import BytesIO

import mapbox_earcut as earcut
import numpy as np
import shapely
import svgelements as se
import trimesh
from shapely import affinity
from shapely.geometry import box
from shapely.geometry.polygon import orient
from shapely.ops import polygonize

from .export import Body
from .massing import _ring_xy, prism

# Plate geometry limits (mm): keep the slab printable.
_DEPTH_MIN, _DEPTH_MAX = 4.0, 40.0
_SLOPE = 0.21        # wedge top: rise per mm of depth (~12° lectern tilt)
_MAX_SUBPATHS = 4000  # complexity guard: fill resolution is O(n²) contains


def _subpaths(path: se.Path) -> list[np.ndarray]:
    """Flatten a path's subpaths to polylines (SVG user units, y still down)."""
    subs: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] | None = None
    for seg in path:
        if isinstance(seg, se.Move):
            cur = []
            subs.append(cur)
            if seg.end is not None:
                cur.append((float(seg.end.x), float(seg.end.y)))
            continue
        if cur is None:
            cur = []
            subs.append(cur)
            if seg.start is not None:
                cur.append((float(seg.start.x), float(seg.start.y)))
        if isinstance(seg, (se.Line, se.Close)):
            if seg.end is not None:
                cur.append((float(seg.end.x), float(seg.end.y)))
        else:  # QuadraticBezier / CubicBezier / Arc: sample
            n = 16 if isinstance(seg, se.Arc) else 12
            for t in np.linspace(0.0, 1.0, n)[1:]:
                p = seg.point(t)
                cur.append((float(p.x), float(p.y)))
    return [np.asarray(s, dtype=np.float64) for s in subs if len(s) >= 2]


def _ring_area(r: np.ndarray) -> float:
    return 0.5 * float(
        np.sum(r[:, 0] * np.roll(r[:, 1], -1) - np.roll(r[:, 0], -1) * r[:, 1])
    )


def _fill_geom(subs: list[np.ndarray], evenodd: bool) -> shapely.Geometry | None:
    """Resolve one element's fill under its fill rule.

    SVG fill rules act on the whole path's ring arrangement, so the rings are
    polygonized into atomic faces first; each face keeps a uniform winding
    count, read off any interior point: nonzero keeps faces where the signed
    containment sum is non-zero, evenodd where the plain count is odd. This
    gets holes, islands-in-holes and self-overlapping strokes right without
    trusting ring orientation conventions.
    """
    rings = [r for r in subs if len(r) >= 3 and abs(_ring_area(r)) > 1e-12]
    if not rings:
        return None
    ring_polys: list[tuple[shapely.Geometry, int]] = []
    for r in rings:
        p = shapely.Polygon(r)
        if not p.is_valid:
            p = p.buffer(0)
        if not p.is_empty:
            ring_polys.append((p, 1 if _ring_area(r) > 0 else -1))
    if not ring_polys:
        return None
    closed = [shapely.LineString(np.vstack([r, r[:1]])) for r in rings]
    faces = list(polygonize(shapely.union_all(closed)))
    kept = []
    for f in faces:
        pt = f.representative_point()
        if evenodd:
            inside = sum(1 for p, _ in ring_polys if p.contains(pt)) % 2 == 1
        else:
            inside = sum(s for p, s in ring_polys if p.contains(pt)) != 0
        if inside:
            kept.append(f)
    if not kept:
        return None
    geom = shapely.union_all(kept)
    return None if geom.is_empty else geom


def _stroke_geom(subs: list[np.ndarray], width: float) -> shapely.Geometry | None:
    """Stroked outlines as ink: buffer the polylines by half the stroke width
    (round caps/joins — close enough to any SVG cap style at plaque scale,
    and friendlier to the nozzle than sharp miters)."""
    lines = [shapely.LineString(s) for s in subs if len(s) >= 2]
    if not lines:
        return None
    geom = shapely.union_all(lines).buffer(width / 2.0, quad_segs=6)
    return None if geom.is_empty else geom


def _walk(group) -> "list[se.SVGElement]":
    out = []
    for child in group:
        if isinstance(child, se.Group):
            out.extend(_walk(child))
        else:
            out.append(child)
    return out


def svg_ink(data: bytes) -> shapely.Geometry:
    """Uploaded SVG -> filled ink geometry (design units, y flipped to print
    orientation, not yet fitted to the plate)."""
    try:
        svg = se.SVG.parse(BytesIO(data), reify=True)
    except Exception as e:
        raise ValueError(f"SVGを解析できません: {e}")

    parts = []
    n_subs = 0
    for e in _walk(svg):
        if isinstance(e, se.SVGText):
            raise ValueError(
                "SVGの<text>要素は使えません。デザインツール側で文字を"
                "アウトライン化（パスに変換）して書き出してください"
            )
        if isinstance(e, se.SVGImage):
            raise ValueError(
                "SVG内の埋め込み画像<image>は使えません。パス化された図形のみ対応しています"
            )
        if not isinstance(e, se.Shape):
            continue
        values = getattr(e, "values", {}) or {}
        if values.get("display") == "none" or values.get("visibility") == "hidden":
            continue
        try:
            path = se.Path(e)
            path.reify()
        except Exception:
            continue
        subs = _subpaths(path)
        if not subs:
            continue
        n_subs += len(subs)
        if n_subs > _MAX_SUBPATHS:
            raise ValueError("SVGが複雑すぎます（パス数を減らして書き出してください）")

        has_fill = e.fill is not None and e.fill.value is not None
        stroke_w = 0.0
        if e.stroke is not None and e.stroke.value is not None:
            stroke_w = float(
                getattr(e, "implicit_stroke_width", None) or e.stroke_width or 0.0
            )
        if has_fill:
            g = _fill_geom(subs, values.get("fill-rule") == "evenodd")
            if g is not None:
                parts.append(g)
        if stroke_w > 0:
            g = _stroke_geom(subs, stroke_w)
            if g is not None:
                parts.append(g)

    if not parts:
        raise ValueError(
            "SVGに描画できる図形が見つかりません（塗りまたは線のあるパスが必要です）"
        )
    ink = shapely.union_all(parts)
    if ink.is_empty:
        raise ValueError("SVGの図形が空です")
    # SVG's y axis points down; the print frame's points up.
    return affinity.scale(ink, 1.0, -1.0, origin=(0, 0))


def _open_pinches(geom: shapely.Geometry, eps: float = 0.01) -> shapely.Geometry:
    """Split zero-width point contacts so the prism walls stay manifold.

    Ink can pinch to a single point — a counter (hole) touching its outline,
    or two shapes meeting after the mm-scale simplify. Extruding a pinch
    fuses four wall faces onto one vertical edge. A tiny morphological
    opening (erode then regrow by an invisible hair) turns every point
    contact into a clean hairline gap, whatever rings it joins.
    """
    return geom.buffer(-eps, quad_segs=2).buffer(eps, quad_segs=2)


def plate_outline(x0: float, x1: float, y0: float, depth: float) -> shapely.Geometry:
    """Plate footprint: front corners rounded, back edge square to the model.

    Built from explicit arc coordinates — a buffer/union construction leaves
    duplicate seam vertices on the ring, which double the extrusion's walls.
    """
    r = min(2.8, 0.22 * depth, 0.1 * (x1 - x0))
    yf = y0 - depth
    if r < 0.4:
        return box(x0, yf, x1, y0)
    left = np.linspace(np.pi, 1.5 * np.pi, 8)
    right = left + 0.5 * np.pi
    pts = (
        [(x0, y0)]
        + [(x0 + r + r * math.cos(t), yf + r + r * math.sin(t)) for t in left]
        + [(x1 - r + r * math.cos(t), yf + r + r * math.sin(t)) for t in right]
        + [(x1, y0)]
    )
    return shapely.Polygon(pts)


def _fit_to_plate(
    ink: shapely.Geometry,
    x0: float,
    x1: float,
    y0: float,
    depth: float,
    outline: shapely.Geometry,
) -> shapely.Geometry:
    """Uniformly scale + centre the ink onto the plate face (print mm).

    The user designed the whole face, so only a small safety inset is kept;
    the rounded front corners then clip whatever still pokes past them.
    """
    inset = max(0.8, 0.05 * depth)
    bw = (x1 - x0) - 2.0 * inset
    bh = depth - 2.0 * inset
    if bw <= 0 or bh <= 0:
        raise ValueError("nameplate is too small for any artwork")
    minx, miny, maxx, maxy = ink.bounds
    w, h = maxx - minx, maxy - miny
    if w <= 0 and h <= 0:
        raise ValueError("SVGの図形が小さすぎます")
    s = min(bw / max(w, 1e-9), bh / max(h, 1e-9))
    cx, cy = (x0 + x1) / 2.0, y0 - depth / 2.0
    g = affinity.affine_transform(
        ink, [s, 0, 0, s, cx - s * (minx + maxx) / 2.0, cy - s * (miny + maxy) / 2.0]
    )
    g = g.simplify(0.02).intersection(outline.buffer(-0.5, quad_segs=4))
    g = _open_pinches(g)
    if g.is_empty:
        raise ValueError(
            "SVGの図形が細すぎて印字できません（線を太くするか奥行きを増やしてください）"
        )
    return g


def _relief_prism(geom: shapely.Geometry, z0: float, z1: float) -> trimesh.Trimesh | None:
    """Extrude the ink; walls follow the triangulation's own boundary.

    Vector art routinely has a hole edge collinear with a shell notch (kanji
    crossbars, digit counters); earcut then bridges the hole through
    degenerate slivers whose boundary no longer matches the input rings, so
    the ring-based wall pass in `massing._extrude_polygon` would leave cracks.
    Deriving the walls from the directed boundary edges of the actual
    triangulation (the `terrain_solid` trick: an edge whose reverse is absent)
    closes the prism no matter how earcut carved the interior.
    """
    parts = []
    for p in getattr(geom, "geoms", [geom]):
        poly = orient(p, 1.0)                    # exterior CCW, holes CW
        rings = [_ring_xy(poly.exterior)] + [_ring_xy(r) for r in poly.interiors]
        rings = [r for r in rings if len(r) >= 3]
        if not rings:
            continue
        pts2d = np.vstack(rings)
        ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
        try:
            idx = earcut.triangulate_float64(np.ascontiguousarray(pts2d), ends)
        except Exception:
            continue
        if len(idx) < 3:
            continue
        F = np.asarray(idx, np.int64).reshape(-1, 3)
        a, b, c = pts2d[F[:, 0]], pts2d[F[:, 1]], pts2d[F[:, 2]]
        cw = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) \
           - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]) < 0
        F[cw] = F[cw, ::-1]                      # every top triangle CCW (+z)
        n = len(pts2d)
        di = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
        dj = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
        bnd = ~np.isin(dj * n + di, di * n + dj)
        bi, bj = di[bnd], dj[bnd]
        verts = np.vstack([
            np.column_stack([pts2d, np.full(n, z1)]),     # 0..n-1  top
            np.column_stack([pts2d, np.full(n, z0)]),     # n..2n-1 base
        ])
        faces = np.vstack([
            F,                                            # top (+z)
            F[:, ::-1] + n,                               # mirrored base (-z)
            np.column_stack([bi, bi + n, bj + n]),        # walls (outward)
            np.column_stack([bi, bj + n, bj]),
        ])
        parts.append(trimesh.Trimesh(vertices=verts, faces=faces, process=False))
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else trimesh.util.concatenate(parts)


def nameplate_bodies(
    svg_data: bytes,
    x0: float,
    x1: float,
    y0: float,
    base_thickness_mm: float,
    depth_mm: float = 16.0,
    relief_mm: float = 0.6,
) -> list[Body]:
    """Plate slab + raised SVG artwork as colour bodies ("plate" / "label").

    The plate is a lectern wedge hanging off the model's front edge (y0,
    usually 0 in the print frame): it shares the base's z range so the print
    stays one piece, and its top rises toward the model at ~12° so the
    artwork faces the viewer. Front corners are rounded; the ink rides the
    sloped top.
    """
    if not svg_data.strip():
        return []
    if base_thickness_mm < 0.5:
        raise ValueError("the nameplate needs base_thickness_mm >= 0.5")
    depth = min(_DEPTH_MAX, max(_DEPTH_MIN, depth_mm))

    outline = plate_outline(x0, x1, y0, depth)
    ink = _fit_to_plate(svg_ink(svg_data), x0, x1, y0, depth, outline)

    # The slope height is affine in y, so lifting a flat-topped prism's top
    # vertices keeps every face planar and the mesh watertight.
    y_front = y0 - depth
    rise = _SLOPE * depth

    def _top_z(y: np.ndarray) -> np.ndarray:
        return 0.8 + rise * (y - y_front) / depth

    slab = prism(outline, -base_thickness_mm, 0.0)
    v = slab.vertices.copy()
    on_top = v[:, 2] > -base_thickness_mm / 2.0
    v[on_top, 2] = _top_z(v[on_top, 1])
    bodies = [Body(trimesh.Trimesh(vertices=v, faces=slab.faces, process=False),
                   "plate")]

    relief = _relief_prism(ink, 0.0, max(0.1, relief_mm))
    if relief is not None:
        lv = relief.vertices.copy()
        lv[:, 2] += _top_z(lv[:, 1])
        bodies.append(Body(
            trimesh.Trimesh(vertices=lv, faces=relief.faces, process=False),
            "label",
        ))
    return bodies
