"""Land-use categories for terrain coloring.

Two providers, both returning a (ny, nx) grid of canonical category names
aligned to the DEM ElevationGrid:

1. PLATEAU luse (preferred, finer vector polygons) — `PlateauLuseProvider`.
   The PLATEAU CityGML data-catalog API maps a bbox/mesh to land-use (luse)
   GML URLs; each file covers one 2nd-level mesh. We stream-parse the LandUse
   polygons and bake them once into a paletted PNG raster per mesh (cached),
   then sample that raster (so re-previews stay cheap even though luse GMLs can
   be hundreds of MB).

2. 国土数値情報 土地利用細分メッシュ ラスタ版 (L03-b_r) — `KsjRasterProvider`.
   Nationwide 100 m GeoTIFF fallback when no PLATEAU luse covers the area.
   Georeferencing is derived from the mesh code (no GDAL).

`resolve_category_grid()` tries PLATEAU first, then falls back to KSJ.
"""
from __future__ import annotations

import hashlib
import io
import math
import zipfile
from typing import Protocol

import numpy as np
import requests
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, maximum_filter

from ..config import DATA_DIR
from .terrain import ElevationGrid

L03_URL = "https://nlftp.mlit.go.jp/ksj/gml/data/L03-b_r/L03-b_r-14/L03-b-14_{code}.zip"
MESH_PX = 800            # cells per 1st-level mesh edge
MESH_DLAT = 2.0 / 3.0    # 1st-level mesh latitude span (40 arc-min)
MESH_DLON = 1.0          # 1st-level mesh longitude span

# Canonical categories (order = default 3MF/print palette order).
CATEGORIES = ["water", "forest", "field", "urban", "bare", "other"]

# Raw L03-b code -> canonical category.
_RAW_TO_CAT = {
    10: "field", 20: "field",            # 田 / その他農用地
    50: "forest", 160: "forest",         # 森林 / ゴルフ場
    60: "bare", 140: "bare",             # 荒地 / 海浜
    70: "urban", 91: "urban", 92: "urban", 100: "urban",  # 建物/道路/鉄道/その他用地
    110: "water", 150: "water",          # 河川湖沼 / 海水域
    0: "other",                          # 解析範囲外
}


class LandUseProvider(Protocol):
    def category_grid(self, grid: ElevationGrid) -> np.ndarray:
        """Return (ny, nx) array of canonical category names for the DEM grid."""
        ...


def _mesh_origin(code: str) -> tuple[float, float]:
    """(lon0 west, lat0 south) of a 1st-level mesh code."""
    return int(code[2:]) + 100.0, int(code[:2]) / 1.5


def _covering_meshes(bbox: tuple[float, float, float, float]) -> list[str]:
    min_lon, min_lat, max_lon, max_lat = bbox
    lat_codes = range(int(min_lat * 1.5), int(max_lat * 1.5) + 1)
    lon_codes = range(int(min_lon - 100), int(max_lon - 100) + 1)
    return [f"{la:02d}{lo:02d}" for la in lat_codes for lo in lon_codes]


class KsjRasterProvider:
    """国土数値情報 土地利用細分メッシュ (ラスタ版) provider. Nationwide."""

    def _load_mesh(self, code: str) -> np.ndarray | None:
        """Return the 800x800 raw-code raster for a mesh, or None if absent."""
        cache = DATA_DIR / "landuse" / f"L03-b-14_{code}.tif"
        if not cache.is_file():
            resp = requests.get(
                L03_URL.format(code=code),
                headers={"User-Agent": "3d-footprint/0.1"},
                timeout=60,
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(resp.content))
            tif = next(n for n in z.namelist() if n.lower().endswith((".tif", ".tiff")))
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(z.read(tif))
        return np.array(Image.open(cache))

    def category_grid(self, grid: ElevationGrid) -> np.ndarray:
        lon2d, lat2d = np.meshgrid(grid.lons, grid.lats)
        out = np.full(lon2d.shape, "other", dtype="<U8")
        bbox = (grid.lons.min(), grid.lats.min(), grid.lons.max(), grid.lats.max())

        for code in _covering_meshes(bbox):
            raster = self._load_mesh(code)
            if raster is None:
                continue
            lon0, lat0 = _mesh_origin(code)
            lat_n = lat0 + MESH_DLAT
            m = (
                (lon2d >= lon0) & (lon2d < lon0 + MESH_DLON)
                & (lat2d >= lat0) & (lat2d < lat_n)
            )
            if not m.any():
                continue
            rows = np.clip(((lat_n - lat2d[m]) / (MESH_DLAT / MESH_PX)).astype(int), 0, MESH_PX - 1)
            cols = np.clip(((lon2d[m] - lon0) / (MESH_DLON / MESH_PX)).astype(int), 0, MESH_PX - 1)
            raw = raster[rows, cols]
            names = np.array([_RAW_TO_CAT.get(int(v), "other") for v in raw], dtype="<U8")
            out[m] = names

        return out


# --------------------------------------------------------------------------- #
# PLATEAU luse provider
# --------------------------------------------------------------------------- #

DATACATALOG_URL = "https://api.plateauview.mlit.go.jp/datacatalog/citygml/m:{codes}"
# The datacatalog rejects more than 30 mesh codes per request ("too many
# bounds"). 3rd-level (bldg) meshes are ~1 km, so a multi-km route easily
# exceeds this; query in chunks and merge.
DATACATALOG_MAX_CODES = 30


def fetch_datacatalog_cities(codes: list[str]) -> list[dict]:
    """Return the merged ``cities`` list for ``codes``, chunked under the API cap."""
    cities: list[dict] = []
    for i in range(0, len(codes), DATACATALOG_MAX_CODES):
        chunk = codes[i : i + DATACATALOG_MAX_CODES]
        try:
            resp = requests.get(
                DATACATALOG_URL.format(codes=",".join(chunk)),
                headers={"User-Agent": "3d-footprint/0.1"},
                timeout=60,
            )
        except requests.RequestException:
            continue
        if resp.status_code == 200:
            cities.extend(resp.json().get("cities", []))
    return cities


_OTHER_IDX = CATEGORIES.index("other")
_CAT_IDX = {c: i for i, c in enumerate(CATEGORIES)}
PLATEAU_PX = 2048        # raster size per 2nd-level mesh (~5 m/px)
MESH2_DLAT = 1.0 / 12.0  # 2nd-level mesh latitude span (5 arc-min)
MESH2_DLON = 1.0 / 8.0   # 2nd-level mesh longitude span (7.5 arc-min)

_LANDUSE_TAG = "{http://www.opengis.net/citygml/landuse/2.0}LandUse"
_LU_NS = {
    "luse": "http://www.opengis.net/citygml/landuse/2.0",
    "gml": "http://www.opengis.net/gml",
}

# PLATEAU Common_landUseType code -> canonical category.
_PLATEAU_TO_CAT = {
    "201": "field", "202": "field", "260": "field",          # 田 / 畑 / 農地
    "203": "forest", "217": "forest", "220": "forest",        # 山林 / 公園緑地 / ゴルフ場
    "204": "water",                                           # 水面
    "205": "bare", "221": "bare", "223": "bare",              # その他自然地 / 太陽光 / その他都市的利用
    "224": "bare", "252": "bare", "263": "bare",              # 低未利用地 / 非可住地 / 空地
    "211": "urban", "212": "urban", "213": "urban", "214": "urban",
    "215": "urban", "216": "urban", "218": "urban", "219": "urban",
    "222": "urban", "251": "urban", "261": "urban", "262": "urban",
    "231": "other",                                           # 不明
}


def _mesh2_codes(bbox: tuple[float, float, float, float]) -> list[str]:
    """2nd-level (6-digit) JIS mesh codes covering a bbox."""
    min_lon, min_lat, max_lon, max_lat = bbox

    def code(lat: float, lon: float) -> str:
        p, u = int(lat * 1.5), int(lon) - 100      # 1st-level mesh
        q = int((lat - p / 1.5) / MESH2_DLAT)      # 0..7 within 1st mesh
        v = int((lon - (u + 100)) / MESH2_DLON)    # 0..7 within 1st mesh
        return f"{p:02d}{u:02d}{q}{v}"

    codes = set()
    lat = min_lat
    while lat <= max_lat + MESH2_DLAT:
        lon = min_lon
        while lon <= max_lon + MESH2_DLON:
            codes.add(code(lat, lon))
            lon += MESH2_DLON
        lat += MESH2_DLAT
    return sorted(codes)


def _mesh2_origin(code: str) -> tuple[float, float]:
    """(lon_west, lat_south) of a 6-digit 2nd-level mesh."""
    lat0 = int(code[:2]) / 1.5 + int(code[4]) * MESH2_DLAT
    lon0 = int(code[2:4]) + 100.0 + int(code[5]) * MESH2_DLON
    return lon0, lat0


class PlateauLuseProvider:
    """PLATEAU 土地利用 (luse) provider. Covers PLATEAU cities only."""

    def _luse_urls(self, codes: list[str]) -> dict[str, list[str]]:
        """Map each covered mesh code -> luse GML URLs (one per covering city)."""
        wanted = set(codes)
        out: dict[str, list[str]] = {}
        for city in fetch_datacatalog_cities(codes):
            for entry in city.get("files", {}).get("luse", []) or []:
                mesh = str(entry.get("code"))
                if mesh in wanted and entry.get("url"):
                    out.setdefault(mesh, []).append(entry["url"])
        return out

    def _raster(self, mesh: str, url: str) -> np.ndarray | None:
        """Category-index raster (PLATEAU_PX^2 uint8) for one luse GML, cached."""
        key = hashlib.sha1(url.encode()).hexdigest()[:16]
        cache = DATA_DIR / "landuse" / "plateau" / f"{mesh}_{key}.png"
        if cache.is_file():
            return np.array(Image.open(cache))

        lon0, lat0 = _mesh2_origin(mesh)
        lat_n = lat0 + MESH2_DLAT
        # Mode "L" (not "P"): pixel value == category index, preserved exactly
        # across PNG save/reload (a paletted "P" image remaps indices on save).
        img = Image.new("L", (PLATEAU_PX, PLATEAU_PX), _OTHER_IDX)
        draw = ImageDraw.Draw(img)
        try:
            with requests.get(
                url, headers={"User-Agent": "3d-footprint/0.1"}, stream=True, timeout=300
            ) as resp:
                resp.raise_for_status()
                resp.raw.decode_content = True
                from lxml import etree

                for _, el in etree.iterparse(resp.raw, tag=_LANDUSE_TAG):
                    cl = el.find("luse:class", _LU_NS)
                    idx = _CAT_IDX[_PLATEAU_TO_CAT.get(cl.text if cl is not None else "", "other")]
                    for pos in el.iter("{http://www.opengis.net/gml}posList"):
                        vals = pos.text.split()
                        lat = np.array(vals[0::3], dtype=float)
                        lon = np.array(vals[1::3], dtype=float)
                        col = (lon - lon0) / MESH2_DLON * PLATEAU_PX
                        row = (lat_n - lat) / MESH2_DLAT * PLATEAU_PX  # north at top
                        if len(col) >= 3:
                            draw.polygon(list(zip(col, row)), fill=idx)
                    el.clear()
        except (requests.RequestException, OSError, ValueError):
            return None

        cache.parent.mkdir(parents=True, exist_ok=True)
        img.save(cache)
        return np.array(img)

    def category_grid(self, grid: ElevationGrid) -> np.ndarray | None:
        bbox = (grid.lons.min(), grid.lats.min(), grid.lons.max(), grid.lats.max())
        urls = self._luse_urls(_mesh2_codes(bbox))
        if not urls:
            return None

        lon2d, lat2d = np.meshgrid(grid.lons, grid.lats)
        out = np.full(lon2d.shape, "other", dtype="<U8")
        # DEM cell size in raster pixels -> conservative-water footprint. Thin
        # PLATEAU waterways (canals) are ~1 px wide; plain sampling aliases them
        # into broken dots, so we dilate the water channel by about one DEM cell
        # and let water win, keeping fine waterways connected.
        dlon = float(np.mean(np.abs(np.diff(grid.lons)))) if grid.lons.size > 1 else MESH2_DLON
        foot = int(np.clip((int(round(dlon / MESH2_DLON * PLATEAU_PX)) | 1), 3, 7))
        any_cover = False
        for mesh, mesh_urls in urls.items():
            lon0, lat0 = _mesh2_origin(mesh)
            lat_n = lat0 + MESH2_DLAT
            m = (
                (lon2d >= lon0) & (lon2d < lon0 + MESH2_DLON)
                & (lat2d >= lat0) & (lat2d < lat_n)
            )
            if not m.any():
                continue
            cols = np.clip((lon2d[m] - lon0) / MESH2_DLON * PLATEAU_PX, 0, PLATEAU_PX - 1).astype(int)
            rows = np.clip((lat_n - lat2d[m]) / MESH2_DLAT * PLATEAU_PX, 0, PLATEAU_PX - 1).astype(int)
            for url in mesh_urls:
                raster = self._raster(mesh, url)
                if raster is None:
                    continue
                any_cover = True
                idx = raster[rows, cols]
                names = np.array([CATEGORIES[int(v)] for v in idx], dtype="<U8")
                water = maximum_filter(
                    (raster == _CAT_IDX["water"]).astype(np.uint8), size=foot
                )
                names[water[rows, cols] > 0] = "water"
                # Fill cells earlier cities left as "other"; water always wins.
                sub = out[m]
                take = (sub == "other") | (names == "water")
                sub[take] = names[take]
                out[m] = sub
        return out if any_cover else None


def _cell_size_m(grid: ElevationGrid) -> float:
    """Approximate ground size (m) of one DEM grid cell (x/y averaged)."""
    lat_mid = float(np.mean(grid.lats))
    dlon, dlat = np.abs(np.diff(grid.lons)), np.abs(np.diff(grid.lats))
    dx = float(np.mean(dlon)) * 111320.0 * math.cos(math.radians(lat_mid)) if dlon.size else 0.0
    dy = float(np.mean(dlat)) * 110540.0 if dlat.size else 0.0
    vals = [v for v in (dx, dy) if v > 0]
    return sum(vals) / len(vals) if vals else 1.0


def _smooth_categories(cat: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Round hard categorical block edges into natural-looking boundaries.

    Land-use rasters paint right-angle blocks at the source mesh resolution (the
    100 m KSJ fallback especially), which read as coarse staircases on the finer
    DEM grid. Category names are nominal and can't be blurred directly, so each
    category becomes a 0/1 indicator field; every field is Gaussian-blurred and
    each cell takes the argmax category. Blurring rounds the corners while the
    argmax keeps clean — but now curved — borders, with speckle absorbed.
    """
    sigma = min(sigma_cells, 12.0)  # cap so tiny-cell grids don't blur to mush
    if sigma < 0.5:
        return cat
    cats = np.unique(cat)
    if cats.size < 2:
        return cat
    stack = np.stack([
        gaussian_filter((cat == c).astype(np.float32), sigma, mode="nearest")
        for c in cats
    ])
    return cats[np.argmax(stack, axis=0)]


def resolve_category_grid(
    grid: ElevationGrid, smooth_m: float = 60.0
) -> tuple[np.ndarray, bool]:
    """Return ``(category_grid, used_plateau)`` for the DEM grid.

    PLATEAU luse is used if the area is covered, else the 国土数値情報 raster
    fallback. `smooth_m` rounds the nominal land-use borders over roughly that
    ground distance (0 disables), turning the coarse source-mesh staircase —
    worst in the 100 m KSJ-only areas — into natural curves.

    PLATEAU is already fine ~5 m vector data whose urban borders are genuinely
    straight (parcels, roads, reclaimed coastlines), so heavy smoothing there
    just blurs real detail and bends lines that should be straight. Where
    PLATEAU covers, smoothing is capped to ~one grid cell; the caller also dials
    the contour curving down so those edges stay crisp. `used_plateau` reports
    which regime applied.
    """
    plateau = PlateauLuseProvider().category_grid(grid)
    used_plateau = plateau is not None
    if not used_plateau:
        cat = KsjRasterProvider().category_grid(grid)
    else:
        # PLATEAU land-use only classifies parcels, so water surfaces (rivers)
        # and odd gaps fall through to "other". Fill those cells from the
        # nationwide KSJ raster, which classifies water/everything.
        cat = plateau
        gap = cat == "other"
        if gap.any():
            ksj = KsjRasterProvider().category_grid(grid)
            cat = cat.copy()
            cat[gap] = ksj[gap]
    # Smooth the (nominal) land-use borders before the sea is stamped in, so the
    # real DEM coastline stays crisp while the inland colour blocks get rounded.
    eff = min(smooth_m, _cell_size_m(grid)) if used_plateau else smooth_m
    if eff > 0:
        pre_water = cat == "water"
        cat = _smooth_categories(cat, eff / _cell_size_m(grid))
        if used_plateau and pre_water.any():
            # Keep PLATEAU's thin waterways; the despeckle pass would erase them.
            cat = np.where(pre_water, "water", cat)
    # Open sea (e.g. Tokyo Bay) is not classified by either land-use source and
    # the GSI DEM returns no-data there; treat those cells as water so the sea
    # is not left as the generic "other"/building-like colour.
    sea = ~np.isfinite(grid.elev)
    if sea.any():
        cat = cat.copy()
        cat[sea] = "water"
    return cat, used_plateau
