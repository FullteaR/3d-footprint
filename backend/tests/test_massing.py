"""Printability massing: footprint algebra and the prism it extrudes.

At print scale a PLATEAU feature's detail falls below the nozzle width and
collapses, so a feature is reduced to a footprint prism with nothing left
thinner than `min_feature`. These are the rules that decide what survives.
"""
from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon, box

from app.core.massing import footprint_of, printable, prism

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
