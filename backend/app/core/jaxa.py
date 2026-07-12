"""JAXA 高解像度土地利用土地被覆図 (HRLULC) v25.04 — fallback land cover.

Public COGs from the JAXA Earth API PDS (no auth): one 0.1°x0.1° uint8
GeoTIFF per tile at pyramid level 4 (9000 px/deg ≈ 10 m), observation year
2024. Tiles are tiny (~tens of KB), cached on disk, and georeferenced from
the tile grid itself (no GDAL). `class_grid` samples the raw HRLULC class per
DEM grid node; mapping to print categories stays in coloring.py so this
module is a pure data source.
"""
from __future__ import annotations

import math

import numpy as np
import requests
from PIL import Image

from ..config import DATA_DIR
from .terrain import ElevationGrid

TILE_URL = (
    "https://s3.ap-northeast-1.wasabisys.com/je-pds/cog/v1/"
    "JAXA.EORC_ALOS_HRLULC.v25.04_japan/{year}/4/"
    "E{lon0}-E{lon1}/E{lon0}-N{lat0}-E{lon1}-N{lat1}-HRLULC.tiff"
)
YEAR = "2024"
TILE_DEG = 0.1

# HRLULC v25.04 class -> print category. 0 (no data) and anything unlisted
# stay unclassified; wetland/tidal flat follow PLATEAU 205 その他自然地 -> bare.
CLASS_CATEGORY: dict[int, str] = {
    1: "water",    # Water bodies
    2: "urban",    # Built-up
    3: "field",    # Paddy field
    4: "field",    # Cropland
    5: "field",    # Grassland
    6: "forest",   # Deciduous broad-leaf forest
    7: "forest",   # Deciduous needle-leaf forest
    8: "forest",   # Evergreen broad-leaf forest
    9: "forest",   # Evergreen needle-leaf forest
    10: "bare",    # Bare
    11: "forest",  # Bamboo forest
    12: "bare",    # Solar panel
    13: "bare",    # Wetland
    14: "field",   # Greenhouse
    15: "bare",    # Rock reef and tidal flat
}


def _fetch_tile(li: int, la: int) -> np.ndarray | None:
    """Class raster of the 0.1° tile whose SW corner is (lon li/10, lat la/10).

    Cached on disk; a missing tile (outside coverage) is remembered with an
    ``.absent`` marker like the DEM tiles. None = no data for this tile.
    """
    cache = DATA_DIR / "jaxa_lulc" / f"{YEAR}_E{li / 10:.2f}_N{la / 10:.2f}.tif"
    absent = cache.with_suffix(".absent")
    if cache.is_file():
        return np.asarray(Image.open(cache))
    if absent.is_file():
        return None

    url = TILE_URL.format(
        year=YEAR,
        lon0=f"{li / 10:.2f}", lon1=f"{(li + 1) / 10:.2f}",
        lat0=f"{la / 10:.2f}", lat1=f"{(la + 1) / 10:.2f}",
    )
    try:
        resp = requests.get(url, headers={"User-Agent": "3d-footprint/0.1"}, timeout=60)
    except requests.RequestException:
        return None  # transient: retry on the next request, no marker
    if resp.status_code in (403, 404):  # S3 may answer either for a missing key
        absent.parent.mkdir(parents=True, exist_ok=True)
        absent.touch()
        return None
    if resp.status_code != 200:
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(resp.content)
    try:
        return np.asarray(Image.open(cache))
    except OSError:
        cache.unlink(missing_ok=True)
        return None


def class_grid(grid: ElevationGrid) -> np.ndarray | None:
    """(ny, nx) raw HRLULC class per DEM grid node; 0 = no data.

    None when no tile over the bbox exists at all (whole area outside the
    HRLULC coverage).
    """
    lon2d, lat2d = np.meshgrid(grid.lons, grid.lats)
    out = np.zeros(lon2d.shape, np.uint8)
    covered = False
    for la in range(math.floor(grid.lats[0] * 10), math.floor(grid.lats[-1] * 10) + 1):
        for li in range(math.floor(grid.lons[0] * 10), math.floor(grid.lons[-1] * 10) + 1):
            tile = _fetch_tile(li, la)
            if tile is None:
                continue
            covered = True
            lon_w, lat_s = li / 10, la / 10
            m = (
                (lon2d >= lon_w) & (lon2d < lon_w + TILE_DEG)
                & (lat2d >= lat_s) & (lat2d < lat_s + TILE_DEG)
            )
            if not m.any():
                continue
            h, w = tile.shape
            # GeoTIFF rows run north -> south.
            rows = np.clip(((lat_s + TILE_DEG - lat2d[m]) * h / TILE_DEG).astype(int), 0, h - 1)
            cols = np.clip(((lon2d[m] - lon_w) * w / TILE_DEG).astype(int), 0, w - 1)
            out[m] = tile[rows, cols]
    return out if covered else None
