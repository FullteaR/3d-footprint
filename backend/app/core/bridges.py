"""PLATEAU brid (bridges/elevated structures) -> printable solids in the scene.

Source: PLATEAU CityGML `brid` files (one per 3rd-level mesh, 8-digit code),
resolved from a bbox via the same data-catalog API used for buildings. A mesh
can appear under several municipalities (a bridge belongs to exactly one city,
so the files partition rather than duplicate); we load *all* of them per mesh.

Unlike buildings — which sit *on* the terrain and get their own vertical
exaggeration — a bridge spans a river/valley at a fixed elevation, so it must
track the terrain's vertical transform: its CityGML heights (標高 T.P., same
datum as the GSI DEM) are projected through `Projection.z_of`, placing the deck
at its true height above the (exaggerated) relief, rising and falling with
`vertical_scale` exactly as the surrounding terrain does.

For printing, the raw deck/girder geometry is far below the FDM nozzle and a
deck floating in mid-air cannot be printed at all. So a bridge is NOT printed
as-is: each connected span is reduced to its footprint and rebuilt as a
printable solid — a thick deck slab at the real deck elevation, plus stout
pillars dropped from the deck down to the terrain surface so nothing floats and
"a bridge is here" still reads (see massing.py). Short spans become a single
solid block to the ground. `min_feature_mm` sets the minimum printable width.

Geometry parsing/triangulation is shared with the building pipeline; only the
feature namespace and the placement differ. Triangulated polygons are cached per
mesh as a compact npz in geographic coordinates so the heavy parse runs once.
"""
from __future__ import annotations

import hashlib

import numpy as np
import requests
import shapely
import trimesh
from lxml import etree
from shapely.geometry import box

from ..config import DATA_DIR
from .buildings import _mesh3_codes, _rings, _triangulate
from .export import Body
from .landuse import fetch_datacatalog_cities
from .massing import footprint_of, printable, prism
from .mesh import Projection

EMBED_MM = 0.5            # how far a pillar/block base sinks into the terrain
DECK_THICKNESS_MM = 1.2  # printable deck-slab thickness (raised to min_feature)
PILLAR_SPACING_MM = 8.0  # target gap between support pillars along a long span

_BRID_NS = "http://www.opengis.net/citygml/bridge/2.0"
_GML_NS = "http://www.opengis.net/gml"
_BRIDGE_TAG = f"{{{_BRID_NS}}}Bridge"
_POLYGON_TAG = f"{{{_GML_NS}}}Polygon"
# LOD1 geometry containers (a coarse prism). Used only when a feature carries no
# finer LOD2/LOD3 surfaces, so we never render a bridge twice.
_LOD1_TAGS = (
    f"{{{_BRID_NS}}}lod1Solid",
    f"{{{_BRID_NS}}}lod1Geometry",
    f"{{{_BRID_NS}}}lod1MultiSurface",
)


def _has_inline_ring(poly: etree._Element) -> bool:
    """True if a gml:Polygon has its own exterior ring (not an xlink reference)."""
    return poly.find(f"gml:exterior/gml:LinearRing/gml:posList",
                     {"gml": _GML_NS}) is not None


def _bridge_polygons(bridge: etree._Element):
    """Yield (exterior, holes) for one Bridge feature, preferring its finest LOD.

    Every real surface in PLATEAU brid is an inline gml:Polygon; the lodNSolid
    composites them by xlink (contributing no inline geometry), so iterating
    inline polygons yields each exactly once. If a feature carries both a coarse
    lod1Solid and finer surfaces, the lod1 polygons are dropped so it is not
    rendered twice.
    """
    inline = [p for p in bridge.iter(_POLYGON_TAG) if _has_inline_ring(p)]
    if not inline:
        return
    lod1_ids = {
        id(p)
        for tag in _LOD1_TAGS
        for container in bridge.iter(tag)
        for p in container.iter(_POLYGON_TAG)
    }
    finer = [p for p in inline if id(p) not in lod1_ids]
    for poly in (finer or inline):
        yield _rings(poly)


class PlateauBridgeProvider:
    """PLATEAU brid (bridge) provider. Covers PLATEAU cities only."""

    def _brid_urls(self, codes: list[str]) -> dict[str, list[str]]:
        """Map covered 8-digit mesh -> every brid GML URL (one per municipality)."""
        wanted = set(codes)
        out: dict[str, list[str]] = {}
        for city in fetch_datacatalog_cities(codes):
            for entry in city.get("files", {}).get("brid", []) or []:
                mesh = str(entry.get("code"))
                if mesh in wanted and entry.get("url"):
                    out.setdefault(mesh, []).append(entry["url"])
        return out

    def _geometry(self, mesh: str, url: str):
        """Cached geographic geometry for one brid GML.

        Returns (verts (N,3) lon/lat/h, faces (M,3)) or None on failure.
        """
        key = hashlib.sha1(url.encode()).hexdigest()[:16]
        cache = DATA_DIR / "bridges" / f"{mesh}_{key}.npz"
        if cache.is_file():
            d = np.load(cache)
            return d["verts"], d["faces"]

        lat_mid = (int(mesh[:2]) / 1.5) + 0.5  # rough, just for the metric basis
        all_v, all_f = [], []
        voff = 0
        try:
            with requests.get(
                url, headers={"User-Agent": "3d-footprint/0.1"}, stream=True, timeout=600
            ) as resp:
                resp.raise_for_status()
                resp.raw.decode_content = True
                for _, b in etree.iterparse(resp.raw, tag=_BRIDGE_TAG):
                    for ext, holes in _bridge_polygons(b):
                        if len(ext) < 3:
                            continue
                        tri = _triangulate(ext, holes, lat_mid)
                        if tri is None:
                            continue
                        pts, faces = tri
                        all_v.append(pts)
                        all_f.append(faces + voff)
                        voff += len(pts)
                    b.clear()
        except (requests.RequestException, OSError, ValueError):
            return None

        if not all_v:
            verts = np.empty((0, 3), np.float32)
            faces = np.empty((0, 3), np.int32)
        else:
            verts = np.vstack(all_v).astype(np.float32)
            faces = np.vstack(all_f).astype(np.int32)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, verts=verts, faces=faces)
        return verts, faces

    def bridge_body(self, proj: Projection, min_feature_mm: float = 0.8) -> Body | None:
        """One Body of every covered bridge, massed as deck slabs + ground pillars.

        Each connected span is reduced to a footprint, capped with a thick deck
        slab at the real deck elevation (so it tracks `vertical_scale`), and tied
        to the ground with stout pillars so nothing floats. `min_feature_mm` sets
        the minimum printable width.
        """
        grid = proj.grid
        bbox = (grid.lons.min(), grid.lats.min(), grid.lons.max(), grid.lats.max())
        urls = self._brid_urls(_mesh3_codes(bbox))
        if not urls:
            return None

        verts, faces = [], []
        voff = 0
        for mesh, mesh_urls in urls.items():
            for url in mesh_urls:
                geo = self._geometry(mesh, url)
                if geo is None or len(geo[0]) == 0:
                    continue
                v, f = geo
                verts.append(v)
                faces.append(f + voff)
                voff += len(v)
        if not verts:
            return None

        verts = np.vstack(verts)
        faces = np.vstack(faces)
        lon, lat, h = verts[:, 0], verts[:, 1], verts[:, 2]
        xy = np.column_stack([proj.x_of(lon), proj.y_of(lat)])  # print mm
        z_real = proj.z_of(h)  # deck elevation in mm (tracks vertical_scale)

        # Clip to the print footprint by face centroid, so a bridge crossing the
        # boundary keeps its inside portion instead of vanishing whole.
        clon = lon[faces].mean(axis=1)
        clat = lat[faces].mean(axis=1)
        inside = (
            (clon >= grid.lons.min()) & (clon <= grid.lons.max())
            & (clat >= grid.lats.min()) & (clat <= grid.lats.max())
        )
        faces = faces[inside]
        if len(faces) == 0:
            return None

        # One footprint polygon per connected span; mass each into deck + pillars.
        fp_all = footprint_of(xy, faces)
        if fp_all is None:
            return None
        pieces: list[trimesh.Trimesh] = []
        for span in getattr(fp_all, "geoms", [fp_all]):
            sp = printable(span, min_feature_mm)
            if sp is None:
                continue
            for poly in getattr(sp, "geoms", [sp]):
                pieces.extend(_span_pieces(poly, xy, z_real, proj, min_feature_mm))
        if not pieces:
            return None

        # Keep the deck slab and pillars as separate watertight shells (they
        # overlap; a slicer unions them). Welding them here would create
        # non-manifold edges where a pillar meets the deck, so no merge.
        mesh = pieces[0] if len(pieces) == 1 else trimesh.util.concatenate(pieces)
        # Bridges share the "building" colour layer (one structure category in UI).
        return Body(mesh, "building")


def _span_pieces(fp, xy, z_real, proj, min_feature) -> list[trimesh.Trimesh]:
    """Mass one span footprint into a deck slab plus supports reaching the terrain.

    The deck top is a high percentile of the bridge's real elevations under the
    span (robust to stray-high verts like railings/cables); the slab is thick
    enough to print. Short spans get a single solid block to the ground; long
    spans get periodic pillars plus guaranteed abutments at the two far ends, so
    the deck is always tied down and nothing floats.
    """
    deck_th = max(DECK_THICKNESS_MM, min_feature)
    probe = fp.buffer(min_feature)                       # verts may sit on the rim
    pm = shapely.contains_xy(probe, xy[:, 0], xy[:, 1])
    if not pm.any():
        return []
    deck_top = float(np.percentile(z_real[pm], 90.0))
    deck_bot = deck_top - deck_th

    pieces: list[trimesh.Trimesh] = []
    deck = prism(fp, deck_bot, deck_top)
    if deck is not None:
        pieces.append(deck)

    ext = np.asarray(fp.exterior.coords)[:-1, :2]
    if len(ext) < 2:
        return pieces
    step = max(1, len(ext) // 24)                        # terrain probe around outline
    probe_xy = ext[::step]
    terr = proj.sample_z(proj.lon_of(probe_xy[:, 0]), proj.lat_of(probe_xy[:, 1]))

    minx, miny, maxx, maxy = fp.bounds
    span_len = float(np.hypot(maxx - minx, maxy - miny))
    if span_len < 2.0 * PILLAR_SPACING_MM:
        # Short span: one solid block from the lowest ground up to the deck.
        block = prism(fp, float(np.min(terr)) - EMBED_MM, deck_top)
        if block is not None:
            pieces.append(block)
        return pieces

    # Long span: periodic pillars + abutments at the two far ends.
    s = max(0.75, 0.75 * min_feature)                    # pillar half-width
    inner = fp.buffer(-0.5 * min_feature)
    if inner.is_empty:
        inner = fp
    centers = _pillar_centers(fp, inner) + _span_ends(ext)
    for cx, cy in centers:
        terrain_z = float(proj.sample_z(proj.lon_of(cx), proj.lat_of(cy)))
        if terrain_z >= deck_bot - 0.1:                  # deck already meets the ground
            continue
        cell = box(cx - s, cy - s, cx + s, cy + s).intersection(fp)
        if cell.is_empty or cell.area < (0.4 * min_feature) ** 2:
            continue
        leg = prism(cell, terrain_z - EMBED_MM, deck_bot)
        if leg is not None:
            pieces.append(leg)
    return pieces


def _pillar_centers(fp, inner) -> list[tuple[float, float]]:
    """Grid points inside `inner` at ~PILLAR_SPACING apart (support placements)."""
    minx, miny, maxx, maxy = fp.bounds
    xs = np.arange(minx + PILLAR_SPACING_MM / 2, maxx, PILLAR_SPACING_MM)
    ys = np.arange(miny + PILLAR_SPACING_MM / 2, maxy, PILLAR_SPACING_MM)
    if xs.size == 0 or ys.size == 0:
        return []
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    m = shapely.contains_xy(inner, pts[:, 0], pts[:, 1])
    return [(float(x), float(y)) for x, y in pts[m]]


def _span_ends(ext: np.ndarray) -> list[tuple[float, float]]:
    """The two farthest-apart outline points (the span's abutments)."""
    c = ext - ext.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    t = c @ vt[0]                                        # project onto principal axis
    return [tuple(ext[int(t.argmin())]), tuple(ext[int(t.argmax())])]
