"""PLATEAU brid (bridges/elevated structures) -> printable solids in the scene.

Source: PLATEAU CityGML `brid` files (one per 3rd-level mesh, 8-digit code),
resolved from a bbox via the same data-catalog API used for buildings. A mesh
can appear under several municipalities (a bridge belongs to exactly one city,
so the files partition rather than duplicate); we load *all* of them per mesh.

Unlike buildings — which sit *on* the terrain and get their own vertical
exaggeration — a bridge spans a river/valley at a fixed elevation, so it must
track the terrain's vertical transform: its CityGML heights (標高 T.P., same
datum as the GSI DEM) are projected straight through `Projection.z_of`, placing
the deck at its true height above the (exaggerated) relief. The deck thus rises
and falls with `vertical_scale` exactly as the surrounding terrain does.

Geometry parsing/triangulation is shared with the building pipeline; only the
feature namespace and the placement differ. Triangulated polygons are cached per
mesh as a compact npz in geographic coordinates so the heavy parse runs once.
"""
from __future__ import annotations

import hashlib

import numpy as np
import requests
import trimesh
from lxml import etree

from ..config import DATA_DIR
from .buildings import _mesh3_codes, _rings, _triangulate
from .export import Body
from .landuse import fetch_datacatalog_cities
from .mesh import Projection

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

    def bridge_body(self, proj: Projection) -> Body | None:
        """One Body holding every covered bridge, placed at its true elevation.

        Bridges are projected through the terrain's full vertical transform
        (`z_of`), so the deck sits above the river/valley by the same amount the
        relief is exaggerated — no separate height knob and no terrain snapping.
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

        gx = proj.x_of(lon)
        gy = proj.y_of(lat)
        gz = proj.z_of(h)
        out_v = np.column_stack([gx, gy, gz])

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

        mesh = trimesh.Trimesh(vertices=out_v, faces=faces, process=False)
        mesh.merge_vertices()
        # Bridges share the "building" colour layer (one structure category in
        # the UI); only their vertical placement differs (real elevation above).
        return Body(mesh, "building")
