"""The print outline: shape inscription, rotation, and the print frame.

The invariant under all of it is that "square" and "regular hexagon" are
regular in printed *millimetres*, not in degrees — the frontend draws the same
shapes off the same constants, so these numbers are a contract between the two.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import Point as shapely_point, Polygon

from app.core.gpx import Track
from app.core.mesh import _M_PER_DEG_LAT, _M_PER_DEG_LON
from app.core.region import (
    Region, clip_track_to_polygon, parse_lonlat_param, parse_rotation_param,
    parse_shape_param,
)

BBOX = (139.0, 35.0, 139.02, 35.01)


def side_lengths_m(region: Region) -> np.ndarray:
    """Edge lengths of the outline, in metres about the region's centre."""
    ring = np.asarray(region.polygon_lonlat().exterior.coords)
    _, clat = region.center
    xy = np.column_stack([
        ring[:, 0] * _M_PER_DEG_LON * math.cos(math.radians(clat)),
        ring[:, 1] * _M_PER_DEG_LAT,
    ])
    return np.linalg.norm(np.diff(xy, axis=0), axis=1)


# ---- parameter parsing -----------------------------------------------------

@pytest.mark.parametrize("shape", ["rect", "square", "hex"])
def test_shape_param_round_trip(shape):
    assert parse_shape_param(shape) == shape


def test_unknown_shape_is_rejected():
    with pytest.raises(ValueError, match="shape must be"):
        parse_shape_param("triangle")


@pytest.mark.parametrize("value,expected", [(0.0, 0.0), (30.0, 30.0),
                                            (-90.0, 270.0), (400.0, 40.0)])
def test_rotation_is_wrapped_into_one_turn(value, expected):
    assert parse_rotation_param(value) == pytest.approx(expected)


def test_rotation_must_be_finite():
    with pytest.raises(ValueError, match="finite"):
        parse_rotation_param(float("nan"))


def test_lonlat_param_round_trip():
    assert parse_lonlat_param("139.5,35.5") == (139.5, 35.5)


@pytest.mark.parametrize("value", ["139.5", "a,b", "200,35", "139,95"])
def test_bad_lonlat_is_rejected(value):
    with pytest.raises(ValueError):
        parse_lonlat_param(value)


# ---- shape inscription -----------------------------------------------------

def test_a_plain_rect_is_exactly_its_bbox():
    region = Region(bbox=BBOX)
    assert region.is_plain
    hw, hh = region.half_extents_m
    clon, clat = region.center
    assert hw == pytest.approx(
        0.01 * _M_PER_DEG_LON * math.cos(math.radians(clat)))
    assert hh == pytest.approx(0.005 * _M_PER_DEG_LAT)


def test_rotation_of_360_still_counts_as_plain():
    """The fast path keys off the angle, not on whether one was supplied."""
    assert Region(bbox=BBOX, rotation_deg=360.0).is_plain
    assert not Region(bbox=BBOX, rotation_deg=0.1).is_plain
    assert not Region(bbox=BBOX, shape="hex").is_plain


def test_a_square_is_square_in_metres():
    region = Region(bbox=BBOX, shape="square")
    hw, hh = region.half_extents_m
    assert hw == pytest.approx(hh)
    # Inscribed, so it never grows past the bbox.
    assert hh == pytest.approx(Region(bbox=BBOX).half_extents_m[1])
    sides = side_lengths_m(region)
    assert sides == pytest.approx(sides[0], rel=1e-6)


def test_a_hexagon_is_regular_in_metres():
    region = Region(bbox=BBOX, shape="hex")
    sides = side_lengths_m(region)
    assert len(sides) == 6
    assert sides == pytest.approx(sides[0], rel=1e-6)
    # Flat-top: half-height is sqrt(3)/2 of the circumradius.
    hw, hh = region.half_extents_m
    assert hh == pytest.approx(math.sqrt(3.0) / 2.0 * hw)
    assert sides[0] == pytest.approx(hw, rel=1e-6)   # side == circumradius


def test_the_span_matches_the_frontends():
    """The panel's scale readout is computed in the browser and the model is
    cut here; the same bbox has to come out the same length in both, or the
    printed scale stops matching the number the user was shown. The twin of
    this assertion is in frontend/src/MapPicker.test.ts."""
    assert Region(bbox=(139.0, 35.0, 139.02, 35.01)).span_m == pytest.approx(
        1823.6487, abs=1e-3)


def test_span_is_the_longest_edge_of_the_shape():
    """size_mm maps onto this, so it must follow the shape, not the bbox."""
    assert Region(bbox=BBOX).span_m == pytest.approx(
        2 * Region(bbox=BBOX).half_extents_m[0])
    sq = Region(bbox=BBOX, shape="square")
    assert sq.span_m == pytest.approx(2 * sq.half_extents_m[0])


# ---- rotation --------------------------------------------------------------

def test_rotating_does_not_change_the_shape():
    """Only the orientation moves: same area, same centre, same side lengths."""
    a, b = Region(bbox=BBOX, shape="hex"), Region(bbox=BBOX, shape="hex",
                                                  rotation_deg=37.0)
    assert side_lengths_m(b) == pytest.approx(side_lengths_m(a), rel=1e-6)
    assert b.center == a.center


def test_the_fetch_bbox_covers_a_turned_outline():
    """The DEM is fetched over the axis-aligned bounds — which a rotated
    outline pushes outside the user's own bbox."""
    region = Region(bbox=BBOX, rotation_deg=45.0)
    w, s, e, n = region.fetch_bbox()
    assert w < BBOX[0] and s < BBOX[1] and e > BBOX[2] and n > BBOX[3]
    assert region.polygon_lonlat().within(
        Polygon([(w, s), (e, s), (e, n), (w, n)]).buffer(1e-12))


def test_a_plain_region_needs_no_transform(flat_proj):
    region = Region(bbox=BBOX)
    assert region.print_xy(3.0, 4.0, flat_proj) == (3.0, 4.0)
    assert region.map_xy(3.0, 4.0, flat_proj) == (3.0, 4.0)


def test_print_and_map_frames_are_inverses(flat_proj):
    region = Region(bbox=BBOX, shape="hex", rotation_deg=23.0)
    x, y = np.array([1.0, 40.0, -12.0]), np.array([2.0, -5.0, 30.0])
    bx, by = region.map_xy(*region.print_xy(x, y, flat_proj), flat_proj)
    assert bx == pytest.approx(x)
    assert by == pytest.approx(y)


def test_the_outline_prints_axis_aligned_at_the_origin(flat_proj):
    """What to_print_frame is for: the finished model starts at (0,0) and
    spans the shape's own extents, whatever angle it was drawn at."""
    region = Region(bbox=BBOX, shape="hex", rotation_deg=31.0)
    hw, hh = region.half_extents_m
    x0, y0, x1, y1 = region.outline_print_mm(flat_proj).bounds
    assert (x0, y0) == pytest.approx((0.0, 0.0), abs=1e-9)
    assert x1 == pytest.approx(2 * hw * flat_proj.scale)
    assert y1 == pytest.approx(2 * hh * flat_proj.scale)


def test_the_rotated_outline_maps_onto_the_print_outline(flat_proj):
    """The map-frame outline, pushed through print_xy, is the print outline —
    the transform bodies get is the one the outline was measured with."""
    region = Region(bbox=BBOX, shape="hex", rotation_deg=31.0)
    ring = np.asarray(region.polygon_mm(flat_proj).exterior.coords)
    px, py = region.print_xy(ring[:, 0], ring[:, 1], flat_proj)
    target = np.asarray(region.outline_print_mm(flat_proj).exterior.coords)
    # Same vertex set (the ring may start anywhere and run either way).
    for x, y in np.column_stack([px, py]):
        assert np.min(np.hypot(target[:, 0] - x, target[:, 1] - y)) < 1e-6


# ---- track clipping --------------------------------------------------------

def test_track_is_cut_at_the_outline_not_at_the_bbox():
    region = Region(bbox=BBOX, shape="hex")
    poly = region.polygon_lonlat()
    clon, clat = region.center
    track = Track(lats=[clat, clat], lons=[clon - 1.0, clon + 1.0])
    (piece,) = clip_track_to_polygon(track, poly)
    assert len(piece.lons) == 2
    assert min(piece.lons) > clon - 1.0 and max(piece.lons) < clon + 1.0
    # Both cuts land on the outline, not on the bbox that contains it.
    for lon, lat in zip(piece.lons, piece.lats):
        assert poly.exterior.distance(shapely_point(lon, lat)) < 1e-9


def test_a_track_that_misses_the_outline_yields_nothing():
    region = Region(bbox=BBOX, shape="hex")
    assert clip_track_to_polygon(Track(lats=[0.0, 1.0], lons=[0.0, 1.0]),
                                 region.polygon_lonlat()) == []


def test_a_single_point_track_yields_nothing():
    region = Region(bbox=BBOX)
    assert clip_track_to_polygon(Track(lats=[35.005], lons=[139.01]),
                                 region.polygon_lonlat()) == []
