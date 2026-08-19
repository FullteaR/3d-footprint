"""Bridges: kept at their real elevation, and never left standing on nothing.

The massing itself lives in voxel.py and is tested there. What is left here is
what `bridge_body` alone decides — that a deck sits at the height the data says
and rises and falls with the exaggerated relief around it, and that a span
PLATEAU modelled without piers still reaches the ground.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from shapely.geometry import box

from app.core import bridges
from app.core.bridges import PlateauBridgeProvider

MIN = 0.8
LAT0, LON0 = 35.0, 139.0          # matches conftest's grids
M_PER_DEG_LAT = 110540.0
M_PER_DEG_LON = 111320.0 * np.cos(np.radians(LAT0))

_BOX_FACES = np.array([
    [0, 1, 3], [0, 3, 2], [4, 7, 5], [4, 6, 7],
    [0, 4, 5], [0, 5, 1], [1, 5, 7], [1, 7, 3],
    [3, 7, 6], [3, 6, 2], [2, 6, 4], [2, 4, 0],
])


def slab(lon, lat, length_m, width_m, low, high):
    """A box in lon/lat/標高: a deck, a pier, whatever the data holds."""
    dlon = 0.5 * length_m / M_PER_DEG_LON
    dlat = 0.5 * width_m / M_PER_DEG_LAT
    verts = np.array([[lon + sx * dlon, lat + sy * dlat, z]
                      for z in (low, high) for sy in (-1, 1) for sx in (-1, 1)],
                     float)
    return verts, _BOX_FACES


def structure(*parts):
    V, F, off = [], [], 0
    for v, f in parts:
        V.append(v)
        F.append(f + off)
        off += len(v)
    return np.vstack(V), np.vstack(F)


def massed(proj, *parts, min_feature=MIN, clip=None):
    geo = structure(*parts)
    with patch.object(PlateauBridgeProvider, "_brid_urls",
                      lambda self, codes: {"53393599": ["u"]}), \
         patch.object(bridges, "_geometry", lambda m, u: geo), \
         patch.object(bridges, "process_map", lambda fn, jobs: [True] * len(jobs)):
        return PlateauBridgeProvider().bridge_body(proj, min_feature, clip=clip)


DECK = (LON0 + 0.005, LAT0 + 0.005, 400.0, 30.0)


# ---- placement -------------------------------------------------------------

def test_the_deck_sits_at_the_elevation_the_data_gives_it(flat_proj):
    """Unlike a building, a bridge is not snapped onto the terrain: it spans
    something, and its height above that something is the whole point."""
    body = massed(flat_proj, slab(*DECK, 40.0, 45.0))
    assert body is not None
    top = float(flat_proj.z_of(45.0))
    assert body.mesh.bounds[1][2] == pytest.approx(top, abs=2.0)


def test_the_deck_rises_and_falls_with_the_exaggerated_relief(make_proj,
                                                              flat_grid):
    """It has to track `vertical_scale` or it sinks into the terrain the moment
    the relief is exaggerated."""
    heights = []
    for vs in (1.0, 4.0):
        proj = make_proj(flat_grid, vertical_scale=vs)
        body = massed(proj, slab(*DECK, 40.0, 45.0))
        heights.append(body.mesh.bounds[1][2])
    assert heights[1] == pytest.approx(4.0 * heights[0], rel=0.2)


# ---- nothing floats --------------------------------------------------------

def test_a_deck_modelled_without_piers_is_propped_down_to_the_terrain(flat_proj):
    """PLATEAU does not always model what holds a bridge up, and a deck in
    mid-air cannot be printed at all."""
    body = massed(flat_proj, slab(*DECK, 40.0, 45.0))
    ground = float(flat_proj.sample_z(DECK[0], DECK[1]))
    assert body.mesh.bounds[0][2] < ground
    assert body.mesh.body_count == 1


def test_piers_in_the_data_are_what_it_stands_on(flat_proj):
    """Where the piers are modelled they are kept, so nothing is invented."""
    lon, lat = DECK[0], DECK[1]
    body = massed(flat_proj, slab(*DECK, 40.0, 45.0),
                  slab(lon - 0.0012, lat, 20.0, 20.0, 5.0, 40.0),
                  slab(lon + 0.0012, lat, 20.0, 20.0, 5.0, 40.0))
    assert body.mesh.body_count == 1
    assert body.mesh.bounds[0][2] < float(flat_proj.sample_z(lon, lat))


# ---- the print outline -----------------------------------------------------

def test_a_bridge_outside_the_print_is_not_in_it(flat_proj):
    far = box(-400.0, -400.0, -300.0, -300.0)
    assert massed(flat_proj, slab(*DECK, 40.0, 45.0), clip=far) is None


def test_no_bridges_at_all_is_no_body(flat_proj):
    with patch.object(PlateauBridgeProvider, "_brid_urls",
                      lambda self, codes: {}):
        assert PlateauBridgeProvider().bridge_body(flat_proj, MIN) is None


def test_the_body_is_one_watertight_solid_on_the_structure_layer(flat_proj):
    body = massed(flat_proj, slab(*DECK, 40.0, 45.0))
    assert body.labels == "building"        # bridges share the structure colour
    assert body.mesh.is_watertight
    assert body.mesh.volume > 0.0
