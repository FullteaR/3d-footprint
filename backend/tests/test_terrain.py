"""Choosing how much DEM to fetch: the zoom the mosaic is assembled at.

The grid is strided down to `grid_max` cells per edge at the end, so any tile
bought finer than that is thrown away again — which for a city-sized area used
to be most of them. These are the rules for how far the zoom steps down, with
the tiles counted rather than downloaded.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core import terrain
from app.core.terrain import MAX_TILES, MIN_ZOOM, fetch_elevation_grid


@pytest.fixture
def counted(monkeypatch):
    """`fetch_elevation_grid` with the network replaced by a tile counter."""
    seen: dict = {"zooms": set(), "tiles": 0}

    def fake_tile(zoom, tx, ty):
        seen["zooms"].add(zoom)
        seen["tiles"] += 1
        return np.zeros((terrain.TILE_SIZE, terrain.TILE_SIZE))

    monkeypatch.setattr(terrain, "_fetch_dem_tile", fake_tile)
    return seen


def around(lon, lat, km):
    """A square bbox of roughly `km` on a side."""
    d_lat = km / 111.0 / 2.0
    d_lon = d_lat / np.cos(np.radians(lat))
    return (lon - d_lon, lat - d_lat, lon + d_lon, lat + d_lat)


def plan(counted, km, grid_max=1000, zoom=15):
    grid = fetch_elevation_grid(around(139.7, 35.6, km), zoom=zoom, grid_max=grid_max)
    return {"zoom": max(counted["zooms"]), "tiles": counted["tiles"],
            "cells": max(grid.elev.shape)}


# ---- how far the zoom steps down -------------------------------------------

def test_a_walk_keeps_the_finest_zoom(counted):
    """The 5 m DEM only exists at z15, so a walk-sized model must not be
    quietly moved off it to save tiles it was never going to waste."""
    assert plan(counted, km=2)["zoom"] == 15


def test_a_city_stops_buying_detail_the_grid_cannot_carry(counted):
    """This is the whole point: at 18 km the finest zoom covers the area with
    thousands of pixels and the stride then discards nine in ten of them."""
    got = plan(counted, km=18)
    assert got["zoom"] < 15
    assert got["tiles"] < 60          # was ~400 before the step-down
    assert got["cells"] >= 500        # and the grid is still most of grid_max


def test_asking_for_a_finer_grid_buys_more_tiles(counted, monkeypatch):
    """`grid_max` now governs the download and not only the model, which is
    what makes it a single detail knob instead of two half-connected ones."""
    coarse = plan(counted, km=18, grid_max=1000)
    counted["zooms"], counted["tiles"] = set(), 0
    fine = plan(counted, km=18, grid_max=4000)
    assert fine["zoom"] > coarse["zoom"]
    assert fine["tiles"] > coarse["tiles"]
    assert fine["cells"] > coarse["cells"]


def test_a_flight_falls_back_to_an_overview(counted):
    got = plan(counted, km=400)
    assert got["zoom"] <= 10
    assert got["tiles"] <= MAX_TILES


def test_the_zoom_never_falls_through_the_floor(counted):
    """Below MIN_ZOOM there is no overview left to ask for."""
    got = plan(counted, km=2500)
    assert got["zoom"] >= MIN_ZOOM


# ---- the memory guard behind it --------------------------------------------

def test_the_mosaic_is_never_larger_than_the_guard_allows(counted):
    for km in (2, 18, 400, 2500):
        counted["zooms"], counted["tiles"] = set(), 0
        assert plan(counted, km=km)["tiles"] <= MAX_TILES
