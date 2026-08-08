"""銘板 (nameplate): a user-uploaded SVG as a plate let into the map.

Think of a rectangular pocket routed into the model with a metal plaque
dropped in: the terrain itself is flattened to a pocket floor under the
plate (`mesh.Projection.flatten_under`), a `PLATE_THICK` tile — the
artwork's own bounding box — fills it, and the SVG's ink stands off that
tile's face just enough to finish level with the ground around it. Nothing
rises above the map. The tile's foot stays buried under the pocket floor,
so plate and model fuse by overlap without a boolean.

Tile and ink come back as **one body** with per-face labels ("plate" /
"label"), so a slicer lists a single nameplate object and gives it two
filaments.

The user places, sizes and turns the plate area on the map wherever the
track leaves room; that area is only the box the artwork is fitted into —
the tile itself hugs the ink. Geometry is built in the plate's own frame —
centred on the origin, axis aligned — and rotated onto the model last
(`at` = cx, cy, deg), so the ink fit and the outline cut never have to
reason about the angle.

The design itself is the user's job (Figma / Illustrator / Inkscape — any
tool that exports SVG with the text outlined to paths); this module only
turns that vector ink into printable geometry. svgelements parses the file
(shapes, transforms), each element's subpath rings are flattened to
polylines, filled regions are resolved under the element's actual fill rule
(nonzero / evenodd) by polygonizing the ring arrangement and keeping the
faces whose winding says "inside", and strokes become buffered outlines.
"""
from __future__ import annotations

import math
from io import BytesIO
from typing import NamedTuple

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
from .massing import _ring_xy

# Plate area limits (mm): keep the artwork printable.
_SIDE_MIN, _SIDE_MAX = 4.0, 200.0
_MAX_SUBPATHS = 4000  # complexity guard: fill resolution is O(n²) contains
PLATE_THICK = 2.0    # the tile's own thickness
BITE = 0.6           # how far the tile's foot stays buried under the pocket
INK_GAP = 0.06       # hairline the artwork keeps off the tile's edge


def clamp_side(mm: float) -> float:
    """Keep a plate edge within the printable range."""
    return min(_SIDE_MAX, max(_SIDE_MIN, mm))


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


def plate_outline(width: float, depth: float) -> shapely.Geometry:
    """Plate footprint (mm) in its own frame: a rounded rectangle about the
    origin.

    Built from explicit arc coordinates: a buffer/union construction leaves
    duplicate seam vertices on the ring, which double the extrusion's walls.
    """
    x1, y1 = width / 2.0, depth / 2.0
    r = min(2.8, 0.22 * min(width, depth), 0.1 * width)
    if r < 0.4:
        return box(-x1, -y1, x1, y1)

    def arc(cx: float, cy: float, t0: float) -> list[tuple[float, float]]:
        return [(cx + r * math.cos(t), cy + r * math.sin(t))
                for t in np.linspace(t0, t0 + 0.5 * np.pi, 8)]

    return shapely.Polygon(
        arc(-x1 + r, -y1 + r, np.pi) + arc(x1 - r, -y1 + r, 1.5 * np.pi)
        + arc(x1 - r, y1 - r, 0.0) + arc(-x1 + r, y1 - r, 0.5 * np.pi)
    )


def plate_to_print(
    geom: shapely.Geometry, at: tuple[float, float, float]
) -> shapely.Geometry:
    """Plate frame -> print frame: rotate about the plate centre, then move."""
    cx, cy, deg = at
    return affinity.translate(affinity.rotate(geom, deg, origin=(0, 0)), cx, cy)


def to_plate_frame(
    geom: shapely.Geometry, at: tuple[float, float, float]
) -> shapely.Geometry:
    """Inverse of `plate_to_print`.

    The model outline is brought into the plate's frame rather than the other
    way round, so everything downstream — the ink fit, the tile, the
    extrusion — stays axis-aligned whatever angle the plaque sits at.
    """
    cx, cy, deg = at
    return affinity.rotate(affinity.translate(geom, -cx, -cy), -deg, origin=(0, 0))


def _polygons(geom: shapely.Geometry) -> shapely.Geometry | None:
    """The polygonal part of a clip result (which can also yield lines)."""
    parts = [p for p in getattr(geom, "geoms", [geom])
             if isinstance(p, shapely.Polygon) and not p.is_empty]
    return shapely.union_all(parts) if parts else None


def inset_plate_outline(
    width: float, depth: float, model: shapely.Geometry
) -> shapely.Geometry:
    """The area the artwork is fitted into (plate frame), cut to the model.

    The frontend keeps the plate inside the outline as it is dragged, so the
    cut normally does nothing; it is what keeps a stale position (the model
    resized under it) from hanging a plaque off the edge.
    """
    kept = _polygons(plate_outline(width, depth).intersection(model))
    if kept is None or kept.area < 0.25 * width * depth:
        raise ValueError("銘板がモデルからはみ出しています。地図で位置を調整してください")
    return kept


def _fit_to_plate(
    ink: shapely.Geometry, outline: shapely.Geometry
) -> shapely.Geometry:
    """Uniformly scale + centre the ink onto the plate face (plate frame, mm).

    The user designed the whole face, so only a small safety inset is kept
    off the outline's bounding box; the rounded corners — and, on a plate the
    model edge has cut, the cut itself — clip whatever still pokes past.
    """
    x0, y0, x1, y1 = outline.bounds
    inset = max(0.8, 0.05 * min(x1 - x0, y1 - y0))
    bw = (x1 - x0) - 2.0 * inset
    bh = (y1 - y0) - 2.0 * inset
    if bw <= 0 or bh <= 0:
        raise ValueError("nameplate is too small for any artwork")
    minx, miny, maxx, maxy = ink.bounds
    w, h = maxx - minx, maxy - miny
    if w <= 0 and h <= 0:
        raise ValueError("SVGの図形が小さすぎます")
    s = min(bw / max(w, 1e-9), bh / max(h, 1e-9))
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
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


def _tri_2d(geom: shapely.Geometry) -> list[tuple[np.ndarray, np.ndarray]]:
    """Triangulate a polygonal region: [(points_2d, ccw_faces), ...]."""
    out = []
    for p in getattr(geom, "geoms", [geom]):
        if not isinstance(p, shapely.Polygon) or p.is_empty:
            continue
        poly = orient(p, 1.0)
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
        F[cw] = F[cw, ::-1]
        out.append((pts2d, F))
    return out


def _place(verts: np.ndarray, at: tuple[float, float, float]) -> np.ndarray:
    """Plate frame -> print frame, for mesh vertices (z is untouched)."""
    cx, cy, deg = at
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    v = np.asarray(verts, dtype=np.float64).copy()
    v[:, 0], v[:, 1] = (v[:, 0] * c - v[:, 1] * s + cx,
                        v[:, 0] * s + v[:, 1] * c + cy)
    return v


def plate_ink(svg_data: bytes, outline: shapely.Geometry) -> shapely.Geometry:
    """The SVG's ink, fitted to the plate area (plate frame, mm)."""
    return _fit_to_plate(svg_ink(svg_data), outline)


def plate_base(ink: shapely.Geometry, model: shapely.Geometry) -> shapely.Geometry:
    """The tile under the artwork: exactly the ink's bounding box.

    Square corners, no rounding, no border — the plate *is* the artwork's box,
    so ink that runs to the edge of its own design runs to the edge of the
    plate. Only the model outline cuts it, and only when the ink was clipped
    diagonally at the model edge (its box would otherwise reach back out over
    thin air).
    """
    kept = _polygons(box(*ink.bounds).intersection(model))
    if kept is None:
        raise ValueError("銘板がモデルからはみ出しています。地図で位置を調整してください")
    return kept


def inset_ink(ink: shapely.Geometry, tile: shapely.Geometry) -> shapely.Geometry:
    """Hold the artwork `INK_GAP` off the tile's edge.

    The two meet at different heights — the artwork's face and the tile's —
    and where the ink reaches the very edge those two heights would have to
    close on one line, which no watertight wall can do. A gap a fifteenth of
    a nozzle wide keeps a closed band of tile all the way round; it costs the
    artwork nothing anyone can see or print.
    """
    kept = _polygons(ink.intersection(tile.buffer(-INK_GAP, join_style="mitre")))
    if kept is None:
        raise ValueError("SVGの図形が板に収まりません")
    return kept


class Levels(NamedTuple):
    """The four z planes of an inlaid plaque (print mm)."""
    bottom: float   # the tile's underside, buried in the terrain
    pocket: float   # what the ground under the tile is flattened to
    face: float     # the tile's face, sitting in the pocket
    top: float      # the artwork on it, flush with the ground around it


def plate_levels(
    z_lo: float, relief_mm: float, z_floor: float, thickness: float = PLATE_THICK
) -> Levels:
    """Where a plaque sits in ground whose lowest point under it is `z_lo`.

    The artwork's face finishes level with that lowest point, so no part of
    the plaque stands above the map — it reads as a plate let into a routed
    pocket, which is what the pocket below it really is. The tile is
    `thickness` deep and its lower part stays buried under the pocket floor,
    so plate and model fuse without a boolean. `z_floor` (the model's own
    base) caps how deep the whole thing may go: the tile just gets thinner
    rather than poking out of the underside and holding the print off the bed.
    """
    top = z_lo
    room = top - z_floor          # all the material there is to sink into
    if room < 0.4:
        raise ValueError("銘板を沈める余裕がありません（底面厚を増やしてください）")
    # A plaque near the model's lowest ground on a thin base gets a shallower
    # relief and a thinner tile rather than an error.
    face = top - min(max(0.1, relief_mm), 0.4 * room)
    bottom = max(z_floor, top - thickness)
    # The pocket floor lands between the two: under the face so the plate sits
    # in the hole rather than being buried by it, over the underside so the
    # terrain still closes over the tile's foot.
    pocket = max(bottom + 0.2, face - BITE)
    return Levels(bottom, pocket, face, top)


class _Shell:
    """Accumulates labelled surfaces and walls into one closed mesh."""

    def __init__(self) -> None:
        self.verts: list[np.ndarray] = []
        self.faces: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []
        self.n = 0

    def _push(self, pts: np.ndarray, tris: np.ndarray, label: str) -> None:
        self.verts.append(pts)
        self.faces.append(tris + self.n)
        self.labels.append(np.full(len(tris), label, dtype="<U8"))
        self.n += len(pts)

    def surface(self, geom, z: float, label: str, down: bool = False) -> None:
        """A flat level of the solid, facing up (or down for an underside)."""
        for pts2d, F in _tri_2d(geom):
            self._push(np.column_stack([pts2d, np.full(len(pts2d), z)]),
                       F[:, ::-1] if down else F, label)

    def wall(self, geom, z0: float, z1: float, label: str) -> None:
        """Vertical skirt around every ring of `geom`, from z0 up to z1.

        Rings come out of `orient` with the material on their left, so the
        quads below always face away from it — outward on a shell, into the
        void on a hole.
        """
        for p in getattr(geom, "geoms", [geom]):
            if not isinstance(p, shapely.Polygon) or p.is_empty:
                continue
            p = orient(p, 1.0)
            for ring in (p.exterior, *p.interiors):
                c = _ring_xy(ring)
                m = len(c)
                if m < 3:
                    continue
                i, j = np.arange(m), (np.arange(m) + 1) % m
                self._push(
                    np.vstack([np.column_stack([c, np.full(m, z0)]),
                               np.column_stack([c, np.full(m, z1)])]),
                    np.vstack([np.column_stack([i, j, j + m]),
                               np.column_stack([i, j + m, i + m])]),
                    label,
                )

    def build(self, at: tuple[float, float, float]) -> Body:
        mesh = trimesh.Trimesh(
            vertices=_place(np.vstack(self.verts), at),
            faces=np.vstack(self.faces), process=False,
        )
        labels = np.concatenate(self.labels)
        mesh.merge_vertices()   # stitches the levels to the walls that carry them
        return Body(mesh, labels)


def nameplate_bodies(
    ink: shapely.Geometry, tile: shapely.Geometry,
    at: tuple[float, float, float], z: Levels,
) -> list[Body]:
    """The plaque as one connected solid, in two colour layers.

    `ink` and `tile` are in the plate's own frame and `at` puts them on the
    model. It is a single tile whose face is stepped up wherever the artwork
    is — not a slab with lettering parked on top — so the mesh comes out as
    one closed shell that a slicer lists as one object and prints in two
    filaments.
    """
    face = tile.difference(ink)
    if _polygons(face) is None and _polygons(tile.intersection(ink)) is None:
        raise ValueError("SVGの図形を立体化できませんでした")
    s = _Shell()
    s.surface(tile, z.bottom, "plate", down=True)  # underside
    s.wall(tile, z.bottom, z.face, "plate")        # the tile's own edge
    s.surface(face, z.face, "plate")               # the face, artwork cut out
    s.wall(ink, z.face, z.top, "label")            # the artwork's sides
    s.surface(ink, z.top, "label")                 # the artwork itself
    return [s.build(at)]
