"""Shared fixtures: synthetic elevation grids, projections and GPX documents.

Nothing in the suite touches the network. Everything the pipeline takes from
outside — the DEM mosaic, the land-use raster — arrives as a plain array, so
the whole mesh/export path runs off grids made up here. The grids mirror the
shapes the real pipeline has to survive:

  * ``flat_grid``   — uniform land, the baseline.
  * ``hill_grid``   — a cone with relief but no no-data.
  * ``island_grid`` — a hill ringed by no-data reaching the border (open sea).
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.mesh import MeshParams, make_projection
from app.core.terrain import ElevationGrid

N = 32                       # grid nodes per edge
LAT0, LON0 = 35.0, 139.0     # arbitrary; every elevation here is invented
DEG = 0.01                   # grid span in degrees (~1.1 km) on both axes


def make_grid(elev: np.ndarray) -> ElevationGrid:
    """Wrap an (ny, nx) elevation array in a grid with matching axes."""
    ny, nx = elev.shape
    return ElevationGrid(
        elev=np.asarray(elev, dtype=np.float64),
        lons=LON0 + np.linspace(0.0, DEG, nx),
        lats=LAT0 + np.linspace(0.0, DEG, ny),   # ascending south -> north
    )


@pytest.fixture
def flat_grid() -> ElevationGrid:
    return make_grid(np.full((N, N), 5.0))


@pytest.fixture
def hill_grid() -> ElevationGrid:
    """Cone: 2 m at the rim, 50 m at the summit, every cell valid."""
    yy, xx = np.mgrid[0:N, 0:N]
    c = (N - 1) / 2.0
    r = np.hypot(yy - c, xx - c) / c
    return make_grid(2.0 + 48.0 * np.clip(1.0 - r, 0.0, 1.0))


@pytest.fixture
def island_grid() -> ElevationGrid:
    """Central cone (2..50 m) surrounded by no-data reaching the border."""
    yy, xx = np.mgrid[0:N, 0:N]
    c = (N - 1) / 2.0
    r = np.hypot(yy - c, xx - c)
    R = N * 0.35
    return make_grid(np.where(r < R, 2.0 + (R - r) / R * 48.0, np.nan))


@pytest.fixture
def make_proj():
    """Factory: a Projection off any grid, overriding MeshParams fields."""
    def _build(grid: ElevationGrid, span_m: float | None = None, **kw):
        return make_projection(grid, MeshParams(**kw), span_m=span_m)
    return _build


@pytest.fixture
def flat_proj(flat_grid, make_proj):
    return make_proj(flat_grid)


@pytest.fixture
def hill_proj(hill_grid, make_proj):
    return make_proj(hill_grid)


@pytest.fixture
def split_cats():
    """(ny, nx) label grid: west half water, east half forest — one border."""
    cats = np.full((N, N), "forest", dtype="<U8")
    cats[:, : N // 2] = "water"
    return cats


@pytest.fixture
def make_gpx():
    """Factory: a GPX document from (lat, lon[, iso time]) tuples."""
    def _build(points, *, tag: str = "trkpt", ns: bool = True) -> bytes:
        xmlns = ' xmlns="http://www.topografix.com/GPX/1/1"' if ns else ""
        body = []
        for pt in points:
            lat, lon = pt[0], pt[1]
            t = f"<time>{pt[2]}</time>" if len(pt) > 2 and pt[2] else ""
            body.append(f'<{tag} lat="{lat}" lon="{lon}">{t}</{tag}>')
        inner = "".join(body)
        if tag == "trkpt":
            inner = f"<trk><trkseg>{inner}</trkseg></trk>"
        elif tag == "rtept":
            inner = f"<rte>{inner}</rte>"
        return f'<?xml version="1.0"?><gpx{xmlns} version="1.1">{inner}</gpx>'.encode()
    return _build
