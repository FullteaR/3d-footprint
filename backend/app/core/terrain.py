"""Fetch GSI (国土地理院) DEM PNG tiles and build an elevation grid for a bbox.

Tile source : https://cyberjapandata.gsi.go.jp/xyz/dem_png/{z}/{x}/{y}.png
Encoding    : x = R*65536 + G*256 + B  (u24)
              x == 2^23           -> invalid (sea / no data)
              x  < 2^23           -> elevation = x * 0.01  [m]
              x  > 2^23           -> elevation = (x - 2^24) * 0.01  [m] (below sea)
Tiles over the sea may 404; those are treated as no-data.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from ..config import DATA_DIR

TILE_URL = "https://cyberjapandata.gsi.go.jp/xyz/dem_png/{z}/{x}/{y}.png"
TILE_SIZE = 256
INVALID = 1 << 23  # 0x800000
MAX_TILES = 256  # guard against runaway bbox/zoom


@dataclass
class ElevationGrid:
    elev: np.ndarray  # (ny, nx) float meters, NaN = no data
    lons: np.ndarray  # (nx,) longitude of each column (ascending)
    lats: np.ndarray  # (ny,) latitude of each row (ascending = south->north)


def _lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[float, float]:
    n = 2**z
    x = (lon + 180.0) / 360.0 * n
    latr = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(latr)) / math.pi) / 2.0 * n
    return x, y


def _decode_tile(png: bytes) -> np.ndarray:
    """RGB PNG bytes -> (256,256) float meters, NaN for invalid."""
    arr = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.uint32)
    x = arr[:, :, 0] * 65536 + arr[:, :, 1] * 256 + arr[:, :, 2]
    elev = np.where(x < INVALID, x.astype(np.float64), (x.astype(np.float64) - 2**24))
    elev *= 0.01
    elev[x == INVALID] = np.nan
    return elev


def _fetch_tile(z: int, x: int, y: int) -> np.ndarray:
    """Fetch one tile (with on-disk cache). Returns NaN tile if missing."""
    cache = DATA_DIR / "dem_png" / str(z) / str(x) / f"{y}.png"
    if cache.is_file():
        return _decode_tile(cache.read_bytes())

    url = TILE_URL.format(z=z, x=x, y=y)
    resp = requests.get(url, headers={"User-Agent": "3d-footprint/0.1"}, timeout=20)
    if resp.status_code == 404:
        return np.full((TILE_SIZE, TILE_SIZE), np.nan)
    resp.raise_for_status()

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(resp.content)
    return _decode_tile(resp.content)


def fetch_elevation_grid(
    bbox: tuple[float, float, float, float], zoom: int, grid_max: int
) -> ElevationGrid:
    """Assemble a DEM mosaic over `bbox` and crop/downsample to <= grid_max cells/edge."""
    min_lon, min_lat, max_lon, max_lat = bbox
    n = 2**zoom

    xt0f, yt0f = _lonlat_to_tile(min_lon, max_lat, zoom)  # NW corner
    xt1f, yt1f = _lonlat_to_tile(max_lon, min_lat, zoom)  # SE corner
    xt0, xt1 = int(math.floor(xt0f)), int(math.floor(xt1f))
    yt0, yt1 = int(math.floor(yt0f)), int(math.floor(yt1f))

    ntiles = (xt1 - xt0 + 1) * (yt1 - yt0 + 1)
    if ntiles > MAX_TILES:
        raise ValueError(
            f"too many DEM tiles ({ntiles}); reduce area or zoom (max {MAX_TILES})"
        )

    # Mosaic covering the whole tile range.
    mosaic = np.full(
        ((yt1 - yt0 + 1) * TILE_SIZE, (xt1 - xt0 + 1) * TILE_SIZE), np.nan
    )
    for ty in range(yt0, yt1 + 1):
        for tx in range(xt0, xt1 + 1):
            tile = _fetch_tile(zoom, tx, ty)
            ry = (ty - yt0) * TILE_SIZE
            rx = (tx - xt0) * TILE_SIZE
            mosaic[ry : ry + TILE_SIZE, rx : rx + TILE_SIZE] = tile

    # Geographic coordinate of each mosaic pixel center.
    gx0 = xt0 * TILE_SIZE
    gy0 = yt0 * TILE_SIZE
    cols = np.arange(mosaic.shape[1])
    rows = np.arange(mosaic.shape[0])
    total = n * TILE_SIZE
    lons_all = (gx0 + cols + 0.5) / total * 360.0 - 180.0
    yy = (gy0 + rows + 0.5) / total
    lats_all = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * yy))))

    # Crop to bbox.
    col_mask = (lons_all >= min_lon) & (lons_all <= max_lon)
    row_mask = (lats_all <= max_lat) & (lats_all >= min_lat)
    if not col_mask.any() or not row_mask.any():
        raise ValueError("bbox produced an empty DEM crop")
    c0, c1 = np.argmax(col_mask), len(col_mask) - np.argmax(col_mask[::-1])
    r0, r1 = np.argmax(row_mask), len(row_mask) - np.argmax(row_mask[::-1])

    elev = mosaic[r0:r1, c0:c1]
    lons = lons_all[c0:c1]
    lats = lats_all[r0:r1]

    # Downsample by striding so the longer edge is <= grid_max.
    step = max(1, int(math.ceil(max(elev.shape) / grid_max)))
    elev = elev[::step, ::step]
    lons = lons[::step]
    lats = lats[::step]

    # Mosaic rows run north->south; flip so lats ascend (south->north).
    elev = elev[::-1, :]
    lats = lats[::-1]

    return ElevationGrid(elev=elev, lons=lons, lats=lats)
