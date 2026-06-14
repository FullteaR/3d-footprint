"""Land-use categories for terrain coloring.

Primary fallback source: 国土数値情報 土地利用細分メッシュ ラスタ版 (L03-b_r).
  - GeoTIFF, 100 m mesh, one file per 1st-level mesh (e.g. L03-b-14_5339.zip).
  - Paletted (mode P); pixel value == the land-use code (0,10,20,...,160).
  - Georeferencing is derived from the mesh code (a 1st-level mesh spans
    1.0 deg lon x 0.6667 deg lat as 800x800 cells), so no GDAL is needed.

A LandUseProvider returns a (ny, nx) grid of canonical category names aligned
to the DEM ElevationGrid. PLATEAU luse will be a second provider (M5b).
"""
from __future__ import annotations

import io
import zipfile
from typing import Protocol

import numpy as np
import requests
from PIL import Image

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
