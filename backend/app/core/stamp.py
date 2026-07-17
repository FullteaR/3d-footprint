"""裏面の出典刻印 (bottom credit stamp): the formal attribution debossed
on the model's flat underside.

The data sources (GSI / PLATEAU / JAXA) require attribution even on
distributed prints, and a shallow groove in the base means the credit
travels with the object itself. The formal 出典 sentence is rendered with
FreeType from Noto Sans CJK Bold (bold so the grooves stay near a nozzle
width), wrapped one source per line, mirrored in x so it reads correctly
when the model is flipped over.

The groove is carved by remeshing rather than a mesh boolean: the underside
is the flat plane z=-base, so the bottom faces crossing the text are
replaced by "region minus ink" back at the base plane, "region ∩ ink"
raised by the groove depth, and walls along the ink outline. This is a pure
2D/shapely operation per body, which matters because the per-colour
land-use bodies are open meshes (colour walls are shared between
neighbours) that no solid boolean engine would accept. Terrain bodies are
one solid per colour with a single string label, so the remesh is
label-safe; the stamp never fails a generation — on any problem (no font,
geometry hiccup) it just skips, and without a CJK font it falls back to
the short ASCII source list.
"""
from __future__ import annotations

import os
from functools import lru_cache

import freetype
import mapbox_earcut as earcut
import numpy as np
import shapely
import trimesh
from shapely import affinity
from shapely.geometry.polygon import orient

from .export import Body
from .massing import _ring_xy
from .nameplate import _fill_geom, _open_pinches

_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    # ASCII-only dev fallback: the formal Japanese line degrades to the
    # short source list (see engrave_credit).
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
_DEPTH = 0.4        # groove depth: two 0.2 mm layers
_EM = 3.0           # nominal font size (mm/em): kanji ~2.6 mm tall, discreet
_EM_MIN = 1.6       # unreadable below this: skip the stamp
_LEADING = 1.32     # line spacing in em
_MARGIN = 1.6       # inset from the front edge and the side limits
_TRACKING = 0.03    # extra letter spacing (em) — legibility in a groove


def credit_text(landuse: bool, buildings: bool) -> str:
    """Short ASCII source list (STL header, CJK-less fallback stamp)."""
    parts = ["GSI"]
    if buildings or landuse:
        parts.append("PLATEAU")
    if landuse:
        parts.append("JAXA")
    return " / ".join(parts)


def _sources(landuse: bool, buildings: bool) -> list[str]:
    parts = ["国土地理院「地理院タイル（標高タイル）」"]
    if buildings or landuse:
        parts.append("国土交通省「Project PLATEAU」CityGML")
    if landuse:
        parts.append("JAXA「高解像度土地利用土地被覆図」")
    return parts


def formal_credit(landuse: bool, buildings: bool) -> str:
    """The full 出典 sentence embedded in distributed model files.

    Follows the 政府標準利用規約 wording: name the sources and state that
    the work is a derivative (「…を加工して作成」).
    """
    return ("出典: " + "、".join(_sources(landuse, buildings))
            + " を加工して作成（3d-footprint）")


def formal_credit_lines(landuse: bool, buildings: bool) -> list[str]:
    """The same sentence wrapped for engraving: one source per line."""
    src = _sources(landuse, buildings)
    lines = ["出典: " + src[0]] + src[1:]
    lines[-1] += " を加工して作成"
    return lines


def ascii_credit(landuse: bool, buildings: bool) -> str:
    """80-byte-safe ASCII variant for the binary STL header."""
    return f"Source: {credit_text(landuse, buildings)} (modified) - 3d-footprint"[:80]


@lru_cache(maxsize=1)
def _face() -> freetype.Face | None:
    paths = [p for p in (os.environ.get("FONT_PATH"), *_FONT_CANDIDATES) if p]
    for path in paths:
        if not os.path.exists(path):
            continue
        face = freetype.Face(path)
        # A .ttc bundles regional faces: prefer the Japanese proportional one.
        fallback = None
        for i in range(face.num_faces):
            cand = freetype.Face(path, i) if i else face
            fam = cand.family_name or b""
            if b"JP" in fam:
                if b"Mono" not in fam:
                    return cand
                fallback = fallback or cand
        return fallback or face
    return None  # fontless dev box: the stamp is skipped, not fatal


class _Pen:
    """FreeType outline decomposer target: flattens béziers into polylines."""

    def __init__(self, x_offset: float):
        self.x = x_offset
        self.contours: list[list[tuple[float, float]]] = []
        self.cur: list[tuple[float, float]] = []

    def _pt(self, v) -> tuple[float, float]:
        return (v.x + self.x, float(v.y))

    def move_to(self, a):
        self.cur = [self._pt(a)]
        self.contours.append(self.cur)

    def line_to(self, a):
        self.cur.append(self._pt(a))

    def conic_to(self, c, a):
        x0, y0 = self.cur[-1]
        cx, cy = self._pt(c)
        x1, y1 = self._pt(a)
        for t in np.linspace(0.0, 1.0, 9)[1:]:
            u = 1.0 - t
            self.cur.append((u * u * x0 + 2 * u * t * cx + t * t * x1,
                             u * u * y0 + 2 * u * t * cy + t * t * y1))

    def cubic_to(self, c1, c2, a):
        x0, y0 = self.cur[-1]
        ax, ay = self._pt(c1)
        bx, by = self._pt(c2)
        x1, y1 = self._pt(a)
        for t in np.linspace(0.0, 1.0, 13)[1:]:
            u = 1.0 - t
            self.cur.append((
                u**3 * x0 + 3 * u * u * t * ax + 3 * u * t * t * bx + t**3 * x1,
                u**3 * y0 + 3 * u * u * t * ay + 3 * u * t * t * by + t**3 * y1,
            ))


def _fill(contours) -> shapely.Geometry | None:
    """Glyph rings -> filled geometry (font units) under the nonzero winding
    rule fonts use. Delegates to the nameplate's ring-arrangement resolver:
    a plain shells-minus-holes pass would erase islands inside holes (the
    玉 in 国, the 口 in 図 …)."""
    subs = [np.asarray(c, dtype=np.float64) for c in contours if len(c) >= 3]
    if not subs:
        return None
    return _fill_geom(subs, evenodd=False)


def _text_ink(face: freetype.Face, text: str) -> shapely.Geometry | None:
    """One ASCII line -> filled geometry in font units."""
    flags = freetype.FT_LOAD_NO_SCALE | freetype.FT_LOAD_NO_BITMAP
    pen_x = 0.0
    contours: list[list[tuple[float, float]]] = []
    for ch in text:
        idx = face.get_char_index(ch)
        if not idx:
            continue
        face.load_glyph(idx, flags)
        pen = _Pen(pen_x)
        face.glyph.outline.decompose(
            pen,
            move_to=lambda a, p: p.move_to(a),
            line_to=lambda a, p: p.line_to(a),
            conic_to=lambda c, a, p: p.conic_to(c, a),
            cubic_to=lambda c1, c2, a, p: p.cubic_to(c1, c2, a),
        )
        contours.extend(pen.contours)
        pen_x += face.glyph.advance.x + _TRACKING * face.units_per_EM
    return _fill(contours)


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


def _engrave_mesh(
    mesh: trimesh.Trimesh,
    ink: shapely.Geometry,
    boundary: shapely.Geometry,
    z_bot: float,
    z_g: float,
) -> trimesh.Trimesh | None:
    """Remesh one body's flat underside with the ink debossed into it.

    Bottom faces (all vertices on the base plane) that touch the ink are
    replaced by their union region re-triangulated in two levels — outside
    the ink back at z_bot, inside it at the groove ceiling z_g — plus
    vertical walls along the ink outline within the region. Faces the ink
    misses are kept verbatim, so the seam vertices line up and a closed
    body stays closed.
    """
    v, f = mesh.vertices.view(np.ndarray), mesh.faces.view(np.ndarray)
    cand = np.nonzero((v[f][:, :, 2] < z_bot + 1e-6).all(axis=1))[0]
    if not len(cand):
        return None
    tri = v[f[cand]][:, :, :2]
    ix0, iy0, ix1, iy1 = ink.bounds
    bb = ((tri[:, :, 0].min(1) <= ix1) & (tri[:, :, 0].max(1) >= ix0)
          & (tri[:, :, 1].min(1) <= iy1) & (tri[:, :, 1].max(1) >= iy0))
    cand, tri = cand[bb], tri[bb]
    if not len(cand):
        return None
    # ink first: the prepared-geometry fast path only applies to the first
    # argument, and per-triangle scans of the full text outline are ~1000x
    # slower without it.
    polys = shapely.polygons(tri)
    hit = shapely.intersects(ink, polys)
    cand = cand[hit]
    if not len(cand):
        return None

    region = shapely.union_all(polys[hit])
    keep = np.ones(len(f), bool)
    keep[cand] = False
    new_v, new_f = [v], [f[keep]]
    nv = len(v)
    for geom, z in ((region.difference(ink), z_bot),
                    (region.intersection(ink), z_g)):
        for pts2d, F in _tri_2d(geom):
            new_v.append(np.column_stack([pts2d, np.full(len(pts2d), z)]))
            new_f.append(F[:, ::-1] + nv)  # underside: faces point -z
            nv += len(pts2d)

    # Walls along the ink outline inside this body's region. The rings walk
    # with ink on the left (kept through the clip), so a left-pointing wall
    # normal faces into the groove void.
    lines = boundary.intersection(region)
    for ls in getattr(lines, "geoms", [lines]):
        if not isinstance(ls, shapely.LineString) or ls.length < 1e-9:
            continue
        c = np.asarray(ls.coords)
        m = len(c)
        new_v.append(np.vstack([
            np.column_stack([c, np.full(m, z_bot)]),   # 0..m-1  bottom
            np.column_stack([c, np.full(m, z_g)]),     # m..2m-1 groove ceiling
        ]))
        i = np.arange(m - 1)
        new_f.append(np.vstack([
            np.column_stack([i, i + m, i + 1 + m]),    # (p0, p1, q1)
            np.column_stack([i, i + 1 + m, i + 1]),    # (p0, q1, q0)
        ]) + nv)
        nv += 2 * m

    return trimesh.Trimesh(
        vertices=np.vstack(new_v), faces=np.vstack(new_f), process=False
    )


def engrave_credit(
    bodies: list[Body],
    landuse: bool,
    buildings: bool,
    x0: float,
    x1: float,
    base_thickness_mm: float,
) -> None:
    """Deboss the formal 出典 into the underside along the front edge.

    The band [x0, x1] is the model's front bottom edge in the print frame
    (same span the nameplate uses). One source per line, first line on top
    when read; mirrored in x so the block reads left-to-right off the
    underside. All lines share one font size, chosen so the longest fits
    the band (capped at `_EM`). Without a CJK font the Japanese sentence
    would render as gaps, so it degrades to the ASCII source list. Bodies
    that don't reach the base or don't cross the text are left untouched,
    and any per-body failure leaves that body as it was.
    """
    if base_thickness_mm < _DEPTH + 0.8:
        return  # too thin a base to carve into
    face = _face()
    if face is None:
        return
    if face.get_char_index("出"):
        lines = formal_credit_lines(landuse, buildings)
    else:
        lines = [credit_text(landuse, buildings)]

    upm = float(face.units_per_EM)
    inks = []
    for ln in lines:
        ink = _text_ink(face, ln)
        if ink is not None:
            inks.append(ink)
    if not inks:
        return

    w_avail = (x1 - x0) - 2 * _MARGIN
    if w_avail <= 0:
        return
    widest = max(ink.bounds[2] - ink.bounds[0] for ink in inks) / upm
    em = min(_EM, w_avail / widest)  # one shared size, longest line fits
    if em < _EM_MIN:
        return

    # Mirror in x (bottom view), scale to mm, centre each line on the band;
    # baselines stack upward so the first line reads on top.
    s = em / upm
    leading = _LEADING * em
    parts = []
    for i, ink in enumerate(inks):
        g = affinity.affine_transform(ink, [-s, 0, 0, s, 0, 0])
        gx0, _, gx1, _ = g.bounds
        parts.append(affinity.translate(
            g,
            xoff=(x0 + x1) / 2 - (gx0 + gx1) / 2,
            yoff=_MARGIN + (len(inks) - 1 - i) * leading,
        ))
    g = _open_pinches(shapely.union_all(parts).simplify(0.01))
    if g.is_empty:
        return
    # Ring direction is load-bearing for the wall winding (see _engrave_mesh):
    # exteriors CCW, holes CW puts the ink interior on the left throughout.
    polys = [orient(p, 1.0) for p in getattr(g, "geoms", [g])
             if isinstance(p, shapely.Polygon) and not p.is_empty]
    if not polys:
        return
    ink = shapely.MultiPolygon(polys)
    boundary = ink.boundary
    shapely.prepare(ink)

    z_bot = -base_thickness_mm
    tx0, ty0, tx1, ty1 = ink.bounds
    for b in bodies:
        if not isinstance(b.labels, str):
            continue  # per-face label arrays would not survive the remesh
        lo, hi = b.mesh.bounds
        if lo[2] > z_bot + 1e-6:
            continue  # body doesn't reach the base underside
        if hi[0] < tx0 or lo[0] > tx1 or hi[1] < ty0 or lo[1] > ty1:
            continue
        try:
            cut = _engrave_mesh(b.mesh, ink, boundary, z_bot, z_bot + _DEPTH)
        except BaseException:
            continue  # never fail a generation over the stamp
        if cut is not None and len(cut.faces):
            b.mesh = cut
