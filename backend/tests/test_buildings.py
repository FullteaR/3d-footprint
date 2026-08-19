"""City-block massing: what individual buildings become on the plate.

A whole city on a 120 mm plate is around 1:100,000 — one nozzle width is some
eighty metres of ground, so a building cannot be a block of its own. The first
half of this covers the rules that decide which buildings merge, how tall the
block they merge into stands, and where it sits on the terrain.

Zoomed in the bargain changes and each building keeps its own shape instead
(voxel.py, tested there). The second half covers what `building_body` itself
decides either way: that a building stands on the terrain the model has rather
than at the elevation PLATEAU recorded for it, that only its height is
exaggerated, and that the print outline bounds the body.
"""
from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box

from unittest.mock import patch

from app.core import buildings
from app.core.buildings import EMBED_MM, PlateauBuildingProvider, _blocks

MIN = 0.8            # a 0.4 mm nozzle's minimum printable width
WIDE = box(-1e4, -1e4, 1e4, 1e4)   # a clip that cuts nothing


def massed(parts, cls, heights, surf, min_feature=MIN, clip=WIDE):
    """`_blocks` with the arrays spelled out, sorted biggest footprint first."""
    out = _blocks(parts, np.array(cls), np.array(heights, float),
                  np.array(surf, float), min_feature, clip)
    return sorted(out, key=lambda b: -b[0].area)


def test_touching_buildings_of_one_class_become_a_single_block():
    parts = [box(0.0, 0.0, 4.0, 4.0), box(3.0, 0.0, 7.0, 4.0)]
    blocks = massed(parts, [0, 0], [2.0, 2.0], [0.0, 0.0])
    assert len(blocks) == 1
    assert blocks[0][0].area == pytest.approx(4 * 4 + 4 * 4 - 1 * 4)


def test_buildings_that_do_not_touch_stay_separate_blocks():
    parts = [box(0.0, 0.0, 4.0, 4.0), box(40.0, 0.0, 44.0, 4.0)]
    assert len(massed(parts, [0, 0], [2.0, 2.0], [0.0, 0.0])) == 2


def test_a_street_between_two_rows_is_not_paved_over():
    """A ward of uniform low-rise is one height class, so if the merge fused
    across roads too the whole ward would print as a single flat slab."""
    parts = [box(0.0, 0.0, 10.0, 10.0), box(10.0 + 3 * MIN, 0.0, 20.0, 10.0)]
    assert len(massed(parts, [0, 0], [2.0, 2.0], [0.0, 0.0])) == 2


def test_a_tower_is_not_averaged_into_the_crust_it_stands_in():
    """Same footprints, different height classes: the tower keeps its own
    block and its own height, and simply overlaps the low-rise one."""
    parts = [box(0.0, 0.0, 10.0, 10.0), box(4.0, 4.0, 6.0, 6.0)]
    blocks = massed(parts, [0, 4], [0.6, 6.0], [0.0, 0.0])
    assert len(blocks) == 2
    crust, tower = blocks
    assert crust[2] - crust[1] == pytest.approx(0.6 + EMBED_MM)
    assert tower[2] - tower[1] == pytest.approx(6.0 + EMBED_MM)


def test_a_blocks_height_is_the_footprint_weighted_mean_of_its_members():
    """A big shed next to a small one is a shed-high block, not the average of
    two buildings — the block's bulk is what a massing model shows."""
    parts = [box(0.0, 0.0, 10.0, 10.0), box(9.0, 0.0, 10.0, 1.0)]
    (_, z0, z1), = massed(parts, [1, 1], [1.0, 5.0], [0.0, 0.0])
    assert z1 - z0 == pytest.approx((100 * 1.0 + 1 * 5.0) / 101 + EMBED_MM)


def test_a_block_reaches_from_the_lowest_ground_it_covers_to_above_the_highest():
    """Nothing floats over the low end of a slope and nothing is buried in the
    high end: the base sinks under the lowest terrain, the top clears the
    highest by the block's full height."""
    parts = [box(0.0, 0.0, 4.0, 4.0), box(3.0, 0.0, 7.0, 4.0)]
    (_, z0, z1), = massed(parts, [0, 0], [2.0, 2.0], [1.0, 5.0])
    assert z0 == pytest.approx(1.0 - EMBED_MM)
    assert z1 == pytest.approx(5.0 + 2.0)


def test_a_block_is_cut_flush_with_the_model_outline():
    parts = [box(0.0, 0.0, 10.0, 10.0)]
    (poly, _, _), = massed(parts, [0], [2.0], [0.0], clip=box(-5.0, -5.0, 6.0, 5.0))
    assert poly.area == pytest.approx(6.0 * 5.0)
    assert poly.bounds[2] <= 6.0 + 1e-9


def test_a_block_that_only_grazes_the_outline_is_dropped():
    """What is left after the cut has to be printable in its own right, or the
    model edge is left with unprintable nubs hanging off it."""
    parts = [box(0.0, 0.0, 10.0, 10.0)]
    graze = box(-5.0, -5.0, 0.05, 5.0)                 # 0.25 mm² < 0.8² mm²
    assert massed(parts, [0], [2.0], [0.0], clip=graze) == []


def test_no_buildings_at_all_is_no_blocks():
    assert _blocks([], np.array([]), np.array([]), np.array([]), MIN, WIDE) == []


# ---- the whole provider, zoomed in ------------------------------------------

LAT0, LON0 = 35.0, 139.0          # matches conftest's grids
M_PER_DEG_LAT = 110540.0
M_PER_DEG_LON = 111320.0 * np.cos(np.radians(LAT0))

_BOX_FACES = np.array([
    [0, 1, 3], [0, 3, 2], [4, 7, 5], [4, 6, 7],
    [0, 4, 5], [0, 5, 1], [1, 5, 7], [1, 7, 3],
    [3, 7, 6], [3, 6, 2], [2, 6, 4], [2, 4, 0],
])


def block(lon, lat, width_m, height_m, ground=5.0):
    """One box building, as PLATEAU hands it over: lon/lat/標高 corners."""
    dlon = 0.5 * width_m / M_PER_DEG_LON
    dlat = 0.5 * width_m / M_PER_DEG_LAT
    verts = np.array([[lon + sx * dlon, lat + sy * dlat, z]
                      for z in (ground, ground + height_m)
                      for sy in (-1, 1) for sx in (-1, 1)], float)
    return verts, _BOX_FACES


def city(*blocks):
    """Stack blocks into the (verts, faces, ftype, vbid) an npz cache holds."""
    V, F, B, off = [], [], [], 0
    for i, (v, f) in enumerate(blocks):
        V.append(v)
        F.append(f + off)
        B.append(np.full(len(v), i, np.int32))
        off += len(v)
    faces = np.vstack(F)
    return (np.vstack(V), faces, np.zeros(len(faces), np.uint8), np.concatenate(B))


def provider_body(proj, *blocks, height_scale=1.0, min_feature=MIN, clip=None):
    geo = city(*blocks)
    with patch.object(PlateauBuildingProvider, "_bldg_urls",
                      lambda self, codes: {"53393599": ["u"]}), \
         patch.object(buildings, "_geometry", lambda m, u: geo), \
         patch.object(buildings, "process_map", lambda fn, jobs: [True] * len(jobs)):
        return PlateauBuildingProvider().building_body(
            proj, height_scale, min_feature, clip=clip)


# ---- placement -------------------------------------------------------------

def test_a_building_stands_on_the_terrain_rather_than_at_its_own_elevation(
        flat_proj):
    """PLATEAU heights and the GSI DEM are separate measurements and do not
    have to agree; trusting the record would bury a building or float it."""
    body = provider_body(flat_proj, block(LON0 + 0.005, LAT0 + 0.005, 60.0, 30.0,
                                   ground=400.0))
    assert body is not None
    ground_mm = float(flat_proj.sample_z(LON0 + 0.005, LAT0 + 0.005))
    assert body.mesh.bounds[0][2] == pytest.approx(ground_mm - EMBED_MM, abs=1.0)


def test_the_base_is_embedded_so_it_fuses_to_the_terrain(flat_proj):
    """Resting on the surface would print as a seam around every block."""
    body = provider_body(flat_proj, block(LON0 + 0.005, LAT0 + 0.005, 60.0, 30.0))
    ground_mm = float(flat_proj.sample_z(LON0 + 0.005, LAT0 + 0.005))
    assert body.mesh.bounds[0][2] < ground_mm


def test_exaggeration_lifts_the_top_and_leaves_the_footing_alone(flat_proj):
    """`height_scale` is the building's own knob — the terrain's
    `vertical_scale` moves the ground it stands on, never its height."""
    plain = provider_body(flat_proj, block(LON0 + 0.005, LAT0 + 0.005, 60.0, 30.0))
    tall = provider_body(flat_proj, block(LON0 + 0.005, LAT0 + 0.005, 60.0, 30.0),
                  height_scale=3.0)
    assert tall.mesh.bounds[0][2] == pytest.approx(plain.mesh.bounds[0][2], abs=0.3)
    plain_h = plain.mesh.bounds[1][2] - plain.mesh.bounds[0][2]
    tall_h = tall.mesh.bounds[1][2] - tall.mesh.bounds[0][2]
    assert tall_h == pytest.approx(3.0 * plain_h, rel=0.15)


# ---- what the outline keeps ------------------------------------------------

def test_a_building_outside_the_print_is_not_in_it(flat_proj):
    far = box(-400.0, -400.0, -300.0, -300.0)
    assert provider_body(flat_proj, block(LON0 + 0.005, LAT0 + 0.005, 60.0, 30.0),
                  clip=far) is None


def test_a_building_straddling_the_outline_is_cut_flush_with_it(flat_proj):
    """Dropping it would punch a hole in the edge of the city; leaving it whole
    would hang it over the printed edge."""
    lon, lat = LON0 + 0.005, LAT0 + 0.005
    x, y = float(flat_proj.x_of(lon)), float(flat_proj.y_of(lat))
    cut = box(x - 40.0, y - 40.0, x, y + 40.0)      # the outline ends mid-building
    body = provider_body(flat_proj, block(lon, lat, 120.0, 30.0), clip=cut)
    assert body is not None
    assert body.mesh.bounds[1][0] <= x + MIN


def test_no_buildings_at_all_is_no_body(flat_proj):
    with patch.object(PlateauBuildingProvider, "_bldg_urls",
                      lambda self, codes: {}):
        assert PlateauBuildingProvider().building_body(flat_proj, 1.0, MIN) is None


# ---- what the nozzle merges ------------------------------------------------

def test_neighbours_closer_than_the_nozzle_come_out_as_one_block(flat_proj):
    """The city block, arrived at by the massing rather than declared: at this
    scale one nozzle is about seven metres of ground."""
    gap_m = 3.0 * MIN / flat_proj.scale
    apart = provider_body(flat_proj,
                   block(LON0 + 0.004, LAT0 + 0.005, 60.0, 30.0),
                   block(LON0 + 0.004 + (60.0 + gap_m) / M_PER_DEG_LON,
                         LAT0 + 0.005, 60.0, 30.0))
    touching = provider_body(flat_proj,
                      block(LON0 + 0.004, LAT0 + 0.005, 60.0, 30.0),
                      block(LON0 + 0.004 + 61.0 / M_PER_DEG_LON,
                            LAT0 + 0.005, 60.0, 30.0))
    assert apart.mesh.body_count == 2
    assert touching.mesh.body_count == 1


def test_the_body_is_one_watertight_solid(flat_proj):
    body = provider_body(flat_proj,
                  block(LON0 + 0.004, LAT0 + 0.005, 60.0, 30.0),
                  block(LON0 + 0.006, LAT0 + 0.005, 60.0, 80.0))
    assert body.labels == "building"
    assert body.mesh.is_watertight
    assert body.mesh.volume > 0.0
