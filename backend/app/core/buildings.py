"""PLATEAU LOD2 buildings -> printable solids sitting on the terrain.

Source: PLATEAU CityGML `bldg` files (one per 3rd-level mesh, 8-digit code),
resolved from a bbox via the same data-catalog API used for luse.

Each building's surfaces are read from its semantic boundaries
(`bldg:boundedBy/{RoofSurface,WallSurface,GroundSurface}` with lod2MultiSurface
polygons). Buildings that only have LOD1 geometry (`bldg:lod1Solid`, a flat-top
prism) are handled too: their polygons are classified roof/wall/ground by
height. Ground polygons are dropped (the base is embedded into the terrain).

Polygons (lat/lon/height, EPSG 6697; height is 標高 T.P., same datum as the
GSI DEM) are triangulated once and cached per mesh as a compact npz in
geographic coordinates. At request time they are projected into the print's
millimetre space and snapped onto the terrain surface, so the heavy parse runs
only on the first use of an area.
"""
from __future__ import annotations

import hashlib

import mapbox_earcut as earcut
import numpy as np
import requests
import trimesh
from lxml import etree

from ..config import DATA_DIR
from .export import Body
from .mesh import _M_PER_DEG_LAT, _M_PER_DEG_LON, Projection

DATACATALOG_URL = "https://api.plateauview.mlit.go.jp/datacatalog/citygml/m:{codes}"
MESH3_DLAT = 1.0 / 120.0  # 3rd-level mesh latitude span (30 arc-sec)
MESH3_DLON = 1.0 / 80.0   # 3rd-level mesh longitude span (45 arc-sec)
EMBED_MM = 0.5            # how far building bases sink into the terrain

_BLDG_NS = "http://www.opengis.net/citygml/building/2.0"
_GML_NS = "http://www.opengis.net/gml"
_BUILDING_TAG = f"{{{_BLDG_NS}}}Building"
_NS = {"bldg": _BLDG_NS, "gml": _GML_NS}
# Semantic surface -> label. Ground surfaces are kept (they cap the bottom so
# each building stays a closed solid) but labelled "wall" since they sit hidden
# below the terrain surface.
_SURFACE_LABEL = {"RoofSurface": "roof", "WallSurface": "wall",
                  "GroundSurface": "wall", "ClosureSurface": "wall",
                  "OuterCeilingSurface": "roof", "OuterFloorSurface": "wall"}
_LABELS = ("wall", "roof")  # ftype 0 = wall, 1 = roof


def _mesh3_codes(bbox: tuple[float, float, float, float]) -> list[str]:
    """3rd-level (8-digit) JIS mesh codes covering a bbox."""
    min_lon, min_lat, max_lon, max_lat = bbox

    def code(lat: float, lon: float) -> str:
        p, u = int(lat * 1.5), int(lon) - 100
        lat1, lon1 = p / 1.5, u + 100
        q = int((lat - lat1) / (1.0 / 12.0))            # 2nd mesh row (0..7)
        v = int((lon - lon1) / (1.0 / 8.0))             # 2nd mesh col (0..7)
        r = int((lat - lat1 - q / 12.0) / MESH3_DLAT)   # 3rd mesh row (0..9)
        w = int((lon - lon1 - v / 8.0) / MESH3_DLON)    # 3rd mesh col (0..9)
        return f"{p:02d}{u:02d}{q}{v}{r}{w}"

    codes = set()
    lat = min_lat
    while lat <= max_lat + MESH3_DLAT:
        lon = min_lon
        while lon <= max_lon + MESH3_DLON:
            codes.add(code(lat, lon))
            lon += MESH3_DLON
        lat += MESH3_DLAT
    return sorted(codes)


def _poslist(ring: etree._Element) -> np.ndarray:
    """LinearRing -> (n,3) lon,lat,height (dropping the repeated closing point)."""
    vals = ring.findtext("gml:posList", namespaces=_NS)
    if not vals:
        return np.empty((0, 3))
    a = np.array(vals.split(), dtype=float).reshape(-1, 3)
    if len(a) > 1 and np.allclose(a[0], a[-1]):
        a = a[:-1]
    return a[:, [1, 0, 2]]  # posList is lat lon h -> store lon lat h


def _triangulate(ext: np.ndarray, holes: list[np.ndarray], lat_mid: float):
    """Triangulate a planar 3D polygon; return (points (k,3), faces (t,3))."""
    rings = [ext] + holes
    pts = np.vstack(rings)
    if len(pts) < 3:
        return None
    # Project to a local metric plane, drop the axis most aligned with the
    # polygon normal, and earcut the remaining two coordinates.
    klon = _M_PER_DEG_LON * np.cos(np.radians(lat_mid))
    metric = pts * np.array([klon, _M_PER_DEG_LAT, 1.0])
    x, y, z = metric[: len(ext)].T
    nx_ = np.sum((y - np.roll(y, -1)) * (z + np.roll(z, -1)))
    ny_ = np.sum((z - np.roll(z, -1)) * (x + np.roll(x, -1)))
    nz_ = np.sum((x - np.roll(x, -1)) * (y + np.roll(y, -1)))
    drop = int(np.argmax(np.abs([nx_, ny_, nz_])))
    keep = [i for i in range(3) if i != drop]
    verts2d = np.ascontiguousarray(metric[:, keep], dtype=np.float64)
    ring_ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    try:
        idx = earcut.triangulate_float64(verts2d, ring_ends)
    except Exception:
        return None
    if len(idx) < 3:
        return None
    return pts, np.asarray(idx, dtype=np.int64).reshape(-1, 3)


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
            for poly in surf.iter(f"{{{_GML_NS}}}Polygon"):
                yield (label, *_rings(poly))
        return

    solid = building.find(".//bldg:lod1Solid", _NS)  # LOD1 fallback (flat prism)
    if solid is None:
        return
    polys = []
    for poly in solid.iter(f"{{{_GML_NS}}}Polygon"):
        ext, holes = _rings(poly)
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


def _rings(poly: etree._Element) -> tuple[np.ndarray, list[np.ndarray]]:
    ext_el = poly.find("gml:exterior/gml:LinearRing", _NS)
    ext = _poslist(ext_el) if ext_el is not None else np.empty((0, 3))
    holes = [_poslist(r) for r in poly.findall("gml:interior/gml:LinearRing", _NS)]
    return ext, [h for h in holes if len(h) >= 3]


class PlateauBuildingProvider:
    """PLATEAU LOD2/LOD1 building provider. Covers PLATEAU cities only."""

    def _bldg_urls(self, codes: list[str]) -> dict[str, str]:
        try:
            resp = requests.get(
                DATACATALOG_URL.format(codes=",".join(codes)),
                headers={"User-Agent": "3d-footprint/0.1"},
                timeout=60,
            )
        except requests.RequestException:
            return {}
        if resp.status_code != 200:
            return {}
        wanted = set(codes)
        out: dict[str, str] = {}
        for city in resp.json().get("cities", []):
            for entry in city.get("files", {}).get("bldg", []) or []:
                mesh = str(entry.get("code"))
                if mesh in wanted and entry.get("url") and mesh not in out:
                    out[mesh] = entry["url"]
        return out

    def _geometry(self, mesh: str, url: str):
        """Cached geographic geometry for one bldg GML.

        Returns (verts (N,3) lon/lat/h, faces (M,3), ftype (M,), vbid (N,)).
        """
        key = hashlib.sha1(url.encode()).hexdigest()[:16]
        cache = DATA_DIR / "buildings" / f"{mesh}_{key}.npz"
        if cache.is_file():
            d = np.load(cache)
            return d["verts"], d["faces"], d["ftype"], d["vbid"]

        lat_mid = (int(mesh[:2]) / 1.5) + 0.5  # rough, just for the metric basis
        all_v, all_f, all_t, all_b = [], [], [], []
        voff = bid = 0
        try:
            with requests.get(
                url, headers={"User-Agent": "3d-footprint/0.1"}, stream=True, timeout=600
            ) as resp:
                resp.raise_for_status()
                resp.raw.decode_content = True
                for _, b in etree.iterparse(resp.raw, tag=_BUILDING_TAG):
                    started = voff
                    for label, ext, holes in _building_polygons(b):
                        if len(ext) < 3:
                            continue
                        tri = _triangulate(ext, holes, lat_mid)
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
                    b.clear()
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
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, verts=verts, faces=faces, ftype=ftype, vbid=vbid)
        return verts, faces, ftype, vbid

    def building_body(self, proj: Projection) -> Body | None:
        """One Body holding every covered building, snapped onto the terrain."""
        grid = proj.grid
        bbox = (grid.lons.min(), grid.lats.min(), grid.lons.max(), grid.lats.max())
        urls = self._bldg_urls(_mesh3_codes(bbox))
        if not urls:
            return None

        verts, faces, ftype, vbid = [], [], [], []
        voff = boff = 0
        for mesh, url in urls.items():
            geo = self._geometry(mesh, url)
            if geo is None or len(geo[0]) == 0:
                continue
            v, f, t, b = geo
            verts.append(v)
            faces.append(f + voff)
            ftype.append(t)
            vbid.append(b + boff)
            voff += len(v)
            boff += int(b.max()) + 1 if len(b) else 0
        if not verts:
            return None

        verts = np.vstack(verts)
        faces = np.vstack(faces)
        ftype = np.concatenate(ftype)
        vbid = np.concatenate(vbid)
        lon, lat, h = verts[:, 0], verts[:, 1], verts[:, 2]

        # Clip buildings to the print footprint (drop those outside the terrain).
        inside = (
            (lon >= grid.lons.min()) & (lon <= grid.lons.max())
            & (lat >= grid.lats.min()) & (lat <= grid.lats.max())
        )

        nb = int(vbid.max()) + 1
        ground = np.full(nb, np.inf)
        np.minimum.at(ground, vbid, h)
        counts = np.bincount(vbid, minlength=nb)
        clon = np.bincount(vbid, lon, minlength=nb) / counts
        clat = np.bincount(vbid, lat, minlength=nb) / counts
        keep_b = np.ones(nb, bool)  # keep a building only if all its verts are inside
        np.logical_and.at(keep_b, vbid, inside)
        surface = proj.sample_z(clon, clat)  # terrain surface (mm) under each building

        gx = proj.x_of(lon)
        gy = proj.y_of(lat)
        gz = surface[vbid] + (h - ground[vbid]) * proj.scale - EMBED_MM
        out_v = np.column_stack([gx, gy, gz])

        keep_face = keep_b[vbid[faces[:, 0]]]
        faces = faces[keep_face]
        ftype = ftype[keep_face]
        if len(faces) == 0:
            return None

        # process=False preserves face order so per-face labels stay aligned;
        # merge_vertices welds the shared surface edges (keeps face count).
        mesh = trimesh.Trimesh(vertices=out_v, faces=faces, process=False)
        mesh.merge_vertices()
        labels = np.array([_LABELS[t] for t in ftype], dtype="<U8")
        return Body(mesh, labels)
