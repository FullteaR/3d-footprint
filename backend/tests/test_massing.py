"""Printability massing: footprint algebra and the prism it extrudes.

At print scale a PLATEAU feature's detail falls below the nozzle width and
collapses. Zoomed out far enough, a feature is reduced to a footprint prism with
nothing left thinner than `min_feature`; these are the rules that decide what
survives that, and the rule that decides when it is the right massing at all.
"""
from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon, box

from app.core.massing import (
    SHAPE_LIMIT_M, blocks_of, footprint_of, keeps_its_shape, outline_parts,
    printable, prism,
)

MIN = 0.8   # a 0.4 mm nozzle's minimum printable width


def quad(x0, y0, x1, y1):
    """A rectangle as the (xy, faces) pair the providers hand over."""
    xy = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
    return xy, np.array([[0, 1, 2], [0, 2, 3]])


# ---- footprints ------------------------------------------------------------

def test_triangles_union_into_one_footprint():
    fp = footprint_of(*quad(0.0, 0.0, 4.0, 2.0))
    assert fp.area == pytest.approx(8.0)


def test_a_feature_with_no_triangles_has_no_footprint():
    xy, _ = quad(0.0, 0.0, 1.0, 1.0)
    assert footprint_of(xy, np.empty((0, 3), int)) is None


def test_collinear_triangles_are_dropped_before_shapely_sees_them():
    """A zero-area ring is not a polygon; the union would fail on it."""
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert footprint_of(xy, np.array([[0, 1, 2]])) is None


# ---- printability ----------------------------------------------------------

def test_sub_feature_noise_is_dropped():
    speck = box(0.0, 0.0, 0.3, 0.3)
    assert printable(speck, MIN) is None


def test_an_everywhere_thin_feature_is_grown_to_the_minimum():
    """A hairline wall prints as nothing at all; it comes back nozzle-wide."""
    bar = box(0.0, 0.0, 20.0, 0.2)
    out = printable(bar, MIN)
    _, y0, _, y1 = out.bounds
    assert y1 - y0 >= MIN - 1e-9


def test_a_fat_feature_keeps_its_size():
    fat = box(0.0, 0.0, 20.0, 10.0)
    assert printable(fat, MIN).area == pytest.approx(fat.area, rel=0.02)


def test_a_notch_narrower_than_the_nozzle_is_closed_up():
    """The close (dilate then erode) is what erases unprintable detail."""
    notched = box(0.0, 0.0, 10.0, 10.0).difference(box(4.9, 0.0, 5.1, 6.0))
    out = printable(notched, MIN)
    assert out.area == pytest.approx(100.0, rel=0.02)


def test_each_component_is_thickened_on_its_own():
    """A thin outbuilding next to a fat one must not bloat the fat one."""
    both = MultiPolygon([box(0.0, 0.0, 20.0, 10.0), box(30.0, 0.0, 50.0, 0.2)])
    out = printable(both, MIN)
    fat = max(out.geoms, key=lambda p: p.area)
    thin = min(out.geoms, key=lambda p: p.area)
    assert fat.area == pytest.approx(200.0, rel=0.02)
    assert thin.bounds[3] - thin.bounds[1] >= MIN - 1e-9


def test_the_clip_is_applied_after_the_widening():
    """Trimming earlier would let the widening push the feature straight back
    out over the model edge."""
    bar = box(0.0, 0.0, 20.0, 0.2)
    clip = box(-100.0, -100.0, 10.0, 100.0)
    out = printable(bar, MIN, clip=clip)
    assert out.bounds[2] <= 10.0 + 1e-9
    assert out.bounds[3] - out.bounds[1] >= MIN - 1e-9


def test_a_feature_that_only_grazes_the_outline_is_dropped():
    """What survives the cut is held to the same noise floor as the uncut
    footprint — an area of min_feature² — so the model edge is not left with
    an unprintable nub hanging off it."""
    building = box(0.0, 0.0, 10.0, 10.0)
    graze = box(-100.0, -100.0, 0.05, 100.0)      # 0.5 mm² < 0.8² mm²
    assert printable(building, MIN, clip=graze) is None


def test_nothing_at_all_stays_nothing():
    assert printable(None, MIN) is None
    assert printable(Polygon(), MIN) is None


# ---- outlines (the building path) ------------------------------------------

def test_a_building_is_left_at_its_true_size():
    """Where `printable` sees noise and the old code saw something to fatten,
    the building path sees a whole building and touches neither its size nor
    its position — growing it here is what would pave over its street."""
    speck = box(0.0, 0.0, 0.3, 0.3)
    assert printable(speck, MIN) is None                  # the bridge rule
    (out,) = outline_parts(speck, MIN)
    assert out.area == pytest.approx(0.09)


def test_detail_finer_than_the_nozzle_is_taken_out_of_an_outline():
    """A 0.1 mm jog in a wall cannot print; carrying its vertices through the
    merge below only makes the union slower."""
    jogged = Polygon([(0, 0), (5, 0), (5, 0.1), (10, 0.1), (10, 5), (0, 5)])
    (out,) = outline_parts(jogged, MIN)
    assert len(out.exterior.coords) < len(jogged.exterior.coords)
    assert out.area == pytest.approx(50.0, rel=0.02)


def test_each_part_of_a_split_building_is_returned_on_its_own():
    """Two detached wings are two footprints, so each can join its own block."""
    wings = MultiPolygon([box(0.0, 0.0, 4.0, 4.0), box(40.0, 0.0, 44.0, 4.0)])
    assert len(outline_parts(wings, MIN)) == 2


def test_no_footprint_at_all_is_no_outline():
    assert outline_parts(None, MIN) == []
    assert outline_parts(Polygon(), MIN) == []


# ---- city blocks -----------------------------------------------------------

def test_an_alley_narrower_than_the_nozzle_is_paved_over():
    """Lot lines and 路地 are below the nozzle: the two sides fuse into one
    block, and no erosion reopens what genuinely overlapped."""
    near = [box(0.0, 0.0, 10.0, 10.0), box(10.0 + 0.5 * MIN, 0.0, 20.0, 10.0)]
    assert len(blocks_of(near, MIN)) == 1


def test_a_street_wider_than_the_nozzle_keeps_its_true_width():
    """The reason a low-rise ward printed as one flat slab. Merging by growing
    footprints narrows every gap that survives by a whole nozzle width, so a
    real road came out a crack; the closing hands it back at full width."""
    apart = [box(0.0, 0.0, 10.0, 10.0), box(10.0 + 3 * MIN, 0.0, 20.0, 10.0)]
    west, east = sorted(blocks_of(apart, MIN), key=lambda p: p.bounds[0])
    assert east.bounds[0] - west.bounds[2] == pytest.approx(3 * MIN, rel=0.02)


def test_a_courtyard_narrower_than_the_nozzle_is_filled_in():
    ring_geom = box(0.0, 0.0, 10.0, 10.0).difference(box(5.0, 5.0, 5.0 + 0.5 * MIN, 6.0))
    (block,) = blocks_of([ring_geom], MIN)
    assert block.area == pytest.approx(100.0, rel=0.02)


def test_a_lone_house_is_grown_until_it_can_print():
    """Nothing is dropped for being small, so the one thing with no neighbours
    to merge with has to be made printable on its own."""
    speck = box(0.0, 0.0, 0.3, 0.3)
    (block,) = blocks_of([speck], MIN)
    x0, y0, x1, y1 = block.bounds
    assert min(x1 - x0, y1 - y0) >= MIN - 1e-9


def test_a_fat_block_is_not_grown():
    fat = box(0.0, 0.0, 20.0, 10.0)
    assert blocks_of([fat], MIN)[0].area == pytest.approx(fat.area, rel=0.02)


def test_no_footprints_at_all_is_no_blocks():
    assert blocks_of([], MIN) == []


# ---- prisms ----------------------------------------------------------------

def test_a_prism_is_watertight_and_has_the_expected_volume():
    mesh = prism(box(0.0, 0.0, 4.0, 3.0), 1.0, 6.0)
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(4.0 * 3.0 * 5.0)
    assert mesh.bounds[0][2] == pytest.approx(1.0)
    assert mesh.bounds[1][2] == pytest.approx(6.0)


def test_a_prism_with_a_hole_keeps_the_hole():
    ring_geom = box(0.0, 0.0, 10.0, 10.0).difference(box(4.0, 4.0, 6.0, 6.0))
    mesh = prism(ring_geom, 0.0, 2.0)
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx((100.0 - 4.0) * 2.0)


def test_a_multipart_footprint_becomes_one_mesh():
    parts = MultiPolygon([box(0.0, 0.0, 1.0, 1.0), box(5.0, 5.0, 6.0, 6.0)])
    mesh = prism(parts, 0.0, 2.0)
    assert mesh.volume == pytest.approx(4.0)
    assert len(mesh.split(only_watertight=False)) == 2


def test_a_prism_of_no_height_is_not_a_solid():
    assert prism(box(0.0, 0.0, 1.0, 1.0), 3.0, 3.0) is None
    assert prism(None, 0.0, 1.0) is None


# ---- the massing switched off ----------------------------------------------

def test_with_no_minimum_width_a_footprint_is_left_exactly_as_it_is():
    """0 is the slider's left stop: print what PLATEAU drew. Nothing is
    simplified away, nothing is grown, nothing is dropped."""
    speck = box(0.0, 0.0, 0.3, 0.3)
    assert printable(speck, 0.0).area == pytest.approx(0.09)   # no noise floor
    assert outline_parts(speck, 0.0)[0].area == pytest.approx(0.09)
    (block,) = blocks_of([speck], 0.0)
    assert block.area == pytest.approx(0.09)                   # not widened


def test_with_no_minimum_width_only_buildings_that_really_touch_merge():
    """No closing radius, so a gap is a gap however small — the model is the
    city as drawn rather than as printable."""
    apart = [box(0.0, 0.0, 10.0, 10.0), box(10.1, 0.0, 20.0, 10.0)]
    touching = [box(0.0, 0.0, 10.0, 10.0), box(9.9, 0.0, 20.0, 10.0)]
    assert len(blocks_of(apart, 0.0)) == 2
    assert len(blocks_of(touching, 0.0)) == 1


# ---- which massing -----------------------------------------------------------

def test_a_building_wider_than_the_nozzle_keeps_its_own_shape(make_proj, flat_grid):
    """Zoomed in the nozzle is a few metres of ground, so a podium, a setback or
    a splayed foot are all things the printer can actually lay down."""
    proj = make_proj(flat_grid, size_mm=120.0)
    assert MIN / proj.scale < SHAPE_LIMIT_M
    assert keeps_its_shape(proj, MIN)


def test_a_building_narrower_than_the_nozzle_has_no_shape_left_to_keep(
        make_proj, flat_grid):
    """Zoomed out the nozzle is tens of metres: a building's outline, its height
    and the street beside it are all finer than one printed line, and what
    prints crisply is the merged block rather than any of them."""
    proj = make_proj(flat_grid, size_mm=40.0)
    assert MIN / proj.scale > SHAPE_LIMIT_M
    assert not keeps_its_shape(proj, MIN)


def test_with_the_massing_switched_off_the_shape_is_always_kept(make_proj,
                                                                flat_grid):
    """No minimum width is no nozzle to be narrower than."""
    assert keeps_its_shape(make_proj(flat_grid, size_mm=40.0), 0.0)
