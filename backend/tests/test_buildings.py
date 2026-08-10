"""City-block massing: what individual buildings become on the plate.

A whole city on a 120 mm plate is around 1:100,000 — one nozzle width is some
eighty metres of ground, so a building cannot be a block of its own. These are
the rules that decide which buildings merge, how tall the block they merge into
stands, and where it sits on the terrain.
"""
from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box

from app.core.buildings import EMBED_MM, _blocks

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
