"""Fetch GSI (国土地理院) DEM PNG tiles and build an elevation grid for a bbox.

Tile source : https://cyberjapandata.gsi.go.jp/xyz/{layer}/{z}/{x}/{y}.png
Encoding    : x = R*65536 + G*256 + B  (u24)
              x == 2^23           -> invalid (sea / no data)
              x  < 2^23           -> elevation = x * 0.01  [m]
              x  > 2^23           -> elevation = (x - 2^24) * 0.01  [m] (below sea)
Tiles over the sea may 404; those are treated as no-data.

Resolution: at zoom 15 the 5 m DEM (`dem5a_png`, photogrammetry/LiDAR) is used
where it exists, falling back per-tile to the 10 m `dem_png` (z14, upsampled) so
areas outside the 5 m coverage still render. At zoom <= 14 only `dem_png` is used.
All layers share the same RGB elevation encoding and 標高 T.P. datum.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import DATA_DIR
from .net import atomic_write_bytes, keyed_lock, session
from .parallel import thread_map

TILE_URL = "https://cyberjapandata.gsi.go.jp/xyz/{layer}/{z}/{x}/{y}.png"
TILE_SIZE = 256
TILE_WORKERS = 16  # concurrent tile downloads (I/O-bound; a tile is a few KB)
INVALID = 1 << 23  # 0x800000
MAX_TILES = 512  # mosaic memory guard (512 tiles ~ 270 MB of float64)
MIN_ZOOM = 8     # dem_png overview floor; z8 tiles span ~150 km already
MAX_ZOOM = 15    # the finest the GSI DEM is published at (the 5 m layers)
DEM10_LAYER = "dem_png"            # 10 m, served up to z14
DEM5_LAYERS = ("dem5a_png", "dem5b_png")  # 5 m, z15; tried in order


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


def _fetch_raw(layer: str, z: int, x: int, y: int) -> np.ndarray | None:
    """Fetch one source tile (on-disk cache). None if the tile is absent (404).

    A 404 is recorded with an empty ``.absent`` marker so repeated previews
    don't re-request tiles outside a layer's coverage. Parallel mosaic workers
    can want the same tile at once (2x2 z15 neighbours share one z14 fallback
    tile), so the fetch is deduped with a per-tile lock and the cache write is
    atomic — a concurrent reader never sees a torn file.
    """
    cache = DATA_DIR / layer / str(z) / str(x) / f"{y}.png"
    absent = cache.with_suffix(".absent")
    if cache.is_file():
        return _decode_tile(cache.read_bytes())
    if absent.is_file():
        return None

    with keyed_lock((layer, z, x, y)):
        if cache.is_file():  # another worker fetched it while we waited
            return _decode_tile(cache.read_bytes())
        if absent.is_file():
            return None
        url = TILE_URL.format(layer=layer, z=z, x=x, y=y)
        resp = session().get(url, timeout=20)
        if resp.status_code == 404:
            absent.parent.mkdir(parents=True, exist_ok=True)
            absent.touch()
            return None
        resp.raise_for_status()

        atomic_write_bytes(cache, resp.content)
        return _decode_tile(resp.content)


def _fetch_dem_tile(zoom: int, x: int, y: int) -> np.ndarray:
    """One DEM tile at `zoom`, preferring 5 m at z15 and *filling gaps* from 10 m.

    At z15 the 5 m layers are tried first. A 5 m tile can be **present but partly
    no-data** — the survey leaves holes over water edges and low reclaimed land —
    so its NaN pixels are filled from the 10 m z14 tile that contains it (matching
    quadrant upsampled 2x), not only when the whole 5 m tile is absent (404).
    Without this, low reclaimed land the 10 m DEM *does* cover (e.g. Tokyo Bay's
    surveyed islands, ~0–7 m) shows up as no-data and floods as sea. The 10 m
    fetch is skipped when the 5 m tile is already gap-free (the common inland
    case). Returns an all-NaN tile only where neither layer covers it (open sea).
    At zoom <= 14 the 10 m layer is used directly.
    """
    if zoom >= 15:
        fine = None
        for layer in DEM5_LAYERS:
            t = _fetch_raw(layer, 15, x, y)
            if t is not None:
                fine = t
                break
        if fine is not None and np.isfinite(fine).all():
            return fine                       # fully covered by 5 m; no fallback
        coarse = _fetch_raw(DEM10_LAYER, zoom - 1, x // 2, y // 2)
        coarse_up = None
        if coarse is not None:
            qx, qy = (x % 2) * (TILE_SIZE // 2), (y % 2) * (TILE_SIZE // 2)
            quad = coarse[qy : qy + TILE_SIZE // 2, qx : qx + TILE_SIZE // 2]
            coarse_up = np.repeat(np.repeat(quad, 2, axis=0), 2, axis=1)
        if fine is None:                      # no 5 m: use 10 m, else open sea
            return coarse_up if coarse_up is not None else np.full(
                (TILE_SIZE, TILE_SIZE), np.nan)
        if coarse_up is not None:             # 5 m with holes: patch from 10 m
            fine = fine.copy()
            holes = ~np.isfinite(fine)
            fine[holes] = coarse_up[holes]
        return fine

    tile = _fetch_raw(DEM10_LAYER, zoom, x, y)
    return tile if tile is not None else np.full((TILE_SIZE, TILE_SIZE), np.nan)


def fetch_elevation_grid(
    bbox: tuple[float, float, float, float], zoom: int, grid_max: int
) -> ElevationGrid:
    """Assemble a DEM mosaic over `bbox` and crop/downsample to <= grid_max cells/edge.

    Wide areas step the zoom down until the mosaic fits MAX_TILES (each step
    quarters the tile count) instead of failing: the output is capped at
    grid_max cells per edge anyway, so past that cap the extra tiles were
    stride waste, not resolution — the printed model is unchanged.
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    def _tile_range(z: int) -> tuple[int, int, int, int]:
        xt0f, yt0f = _lonlat_to_tile(min_lon, max_lat, z)  # NW corner
        xt1f, yt1f = _lonlat_to_tile(max_lon, min_lat, z)  # SE corner
        return (int(math.floor(xt0f)), int(math.floor(xt1f)),
                int(math.floor(yt0f)), int(math.floor(yt1f)))

    def _pixels(z: int) -> float:
        """Mosaic pixels across the wider edge of the bbox at zoom `z`."""
        xa, ya = _lonlat_to_tile(min_lon, max_lat, z)
        xb, yb = _lonlat_to_tile(max_lon, min_lat, z)
        return max(xb - xa, yb - ya) * TILE_SIZE

    # Don't buy detail the output cannot carry. The mosaic is strided down to
    # `grid_max` cells per edge at the end, so while the next zoom *coarser*
    # would still cover the bbox with that many pixels, everything the finer
    # one costs is thrown away by the stride. This is what makes `grid_max` a
    # limit on the download and not only on the model: an 18 km area at the
    # default falls from 399 tiles to 28 with the same grid coming out.
    while zoom > MIN_ZOOM and _pixels(zoom - 1) >= grid_max:
        zoom -= 1

    xt0, xt1, yt0, yt1 = _tile_range(zoom)
    while (xt1 - xt0 + 1) * (yt1 - yt0 + 1) > MAX_TILES and zoom > MIN_ZOOM:
        zoom -= 1
        xt0, xt1, yt0, yt1 = _tile_range(zoom)
    n = 2**zoom

    ntiles = (xt1 - xt0 + 1) * (yt1 - yt0 + 1)
    if ntiles > MAX_TILES:  # only reachable for a near-continental bbox
        raise ValueError(
            f"too many DEM tiles ({ntiles}); reduce area or zoom (max {MAX_TILES})"
        )

    # Mosaic covering the whole tile range. Tiles are independent HTTP GETs,
    # so they download in parallel; pasting stays sequential.
    coords = [(tx, ty) for ty in range(yt0, yt1 + 1) for tx in range(xt0, xt1 + 1)]
    tiles = thread_map(
        lambda tx, ty: _fetch_dem_tile(zoom, tx, ty), coords, TILE_WORKERS
    )
    mosaic = np.full(
        ((yt1 - yt0 + 1) * TILE_SIZE, (xt1 - xt0 + 1) * TILE_SIZE), np.nan
    )
    for (tx, ty), tile in zip(coords, tiles):
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
