"""GPX parsing, time trimming, extent padding and polyline clipping."""
from __future__ import annotations

import pytest

from app.core.gpx import (
    Track, clip_track, expand_bbox, parse_bbox_param, parse_gpx,
    parse_time_range_param, trim_track,
)

T0 = "2026-05-01T00:00:00Z"
T1 = "2026-05-01T00:10:00Z"
T2 = "2026-05-01T00:20:00Z"


# ---- parsing ---------------------------------------------------------------

@pytest.mark.parametrize("ns", [True, False])
def test_parse_trkpt_with_and_without_namespace(make_gpx, ns):
    """Real files vary; the finder is namespace-agnostic on purpose."""
    track = parse_gpx(make_gpx([(35.0, 139.0), (35.1, 139.2)], ns=ns))
    assert track.lats == [35.0, 35.1]
    assert track.lons == [139.0, 139.2]
    assert track.times is None


@pytest.mark.parametrize("tag", ["rtept", "wpt"])
def test_route_and_waypoint_files_still_yield_a_path(make_gpx, tag):
    track = parse_gpx(make_gpx([(35.0, 139.0), (35.1, 139.1)], tag=tag))
    assert len(track.lons) == 2


def test_trkpt_wins_over_other_point_kinds(make_gpx):
    doc = make_gpx([(35.0, 139.0), (35.1, 139.1)]).replace(
        b"</gpx>", b'<wpt lat="1" lon="2"/></gpx>'
    )
    track = parse_gpx(doc)
    assert track.lats == [35.0, 35.1]


def test_a_point_with_no_coordinates_is_a_plain_value_error():
    """It used to reach `float(None)` and come back a 500 — a broken export is
    the caller's problem to see, not the server's to crash on."""
    with pytest.raises(ValueError, match="point 1 is missing"):
        parse_gpx(b'<gpx><trk><trkseg><trkpt lon="139.5"/></trkseg></trk></gpx>')


@pytest.mark.parametrize("lat,lon", [
    ("999", "139.5"),        # off the earth
    ("35.5", "-181"),
    ("nan", "nan"),          # unordered, so the range check refuses it
    ("inf", "139.5"),
])
def test_a_point_that_is_not_on_the_earth_is_rejected_here(lat, lon):
    """Not several steps later inside the DEM crop, which could only answer
    with a message about itself."""
    doc = (f'<gpx><trk><trkseg><trkpt lat="{lat}" lon="{lon}"/>'
           f'<trkpt lat="35.6" lon="139.6"/></trkseg></trk></gpx>').encode()
    with pytest.raises(ValueError, match="off the earth"):
        parse_gpx(doc)


def test_the_edges_of_the_earth_are_still_on_it():
    track = parse_gpx(b'<gpx><trk><trkseg><trkpt lat="-90" lon="-180"/>'
                      b'<trkpt lat="90" lon="180"/></trkseg></trk></gpx>')
    assert track.lats == [-90.0, 90.0]


def test_pointless_gpx_is_rejected():
    with pytest.raises(ValueError, match="no trkpt"):
        parse_gpx(b'<?xml version="1.0"?><gpx version="1.1"><trk/></gpx>')


@pytest.mark.parametrize("data", [b"hello, I am not XML", b"", b"<gpx>"])
def test_a_file_that_is_not_xml_is_a_plain_value_error(data):
    """lxml raises a SyntaxError subclass, which the route would not catch —
    a mistyped upload has to come back as a message, not a 500."""
    with pytest.raises(ValueError, match="could not be parsed"):
        parse_gpx(data)


def test_bbox_is_min_max_over_the_points():
    track = Track(lats=[35.0, 36.0, 34.0], lons=[139.0, 138.0, 140.0])
    assert track.bbox == (138.0, 34.0, 140.0, 36.0)


# ---- timestamps ------------------------------------------------------------

def test_times_are_epoch_seconds(make_gpx):
    track = parse_gpx(make_gpx([(35.0, 139.0, T0), (35.1, 139.1, T1)]))
    assert track.times is not None
    assert track.times[1] - track.times[0] == 600.0


def test_naive_stamps_are_read_as_utc(make_gpx):
    naive = parse_gpx(make_gpx([(35.0, 139.0, "2026-05-01T00:00:00")])).times
    aware = parse_gpx(make_gpx([(35.0, 139.0, T0)])).times
    assert naive == aware


def test_partially_stamped_files_fill_forward_and_back(make_gpx):
    """A gap inherits the last stamp; a leading gap inherits the first."""
    track = parse_gpx(make_gpx([
        (35.0, 139.0), (35.1, 139.1, T1), (35.2, 139.2), (35.3, 139.3, T2),
    ]))
    t = track.times
    assert t[0] == t[1]           # leading gap took the first known stamp
    assert t[2] == t[1]           # middle gap held the last one
    assert t[3] - t[1] == 600.0


def test_unparsable_time_is_ignored(make_gpx):
    assert parse_gpx(make_gpx([(35.0, 139.0, "yesterday")])).times is None


# ---- time_range ------------------------------------------------------------

def test_time_range_param_round_trip():
    assert parse_time_range_param("100,200.5") == (100.0, 200.5)


@pytest.mark.parametrize("value", ["100", "100,200,300", "a,b", "200,100", "5,5"])
def test_bad_time_range_is_rejected(value):
    with pytest.raises(ValueError):
        parse_time_range_param(value)


def test_time_range_rejects_infinities():
    with pytest.raises(ValueError, match="finite"):
        parse_time_range_param("-inf,inf")


def test_trim_keeps_the_points_inside_the_window():
    track = Track(lats=[0, 1, 2, 3], lons=[0, 1, 2, 3], times=[0, 10, 20, 30])
    kept = trim_track(track, 5, 25)
    assert kept.lats == [1, 2]
    assert kept.times == [10, 20]


def test_trim_needs_timestamps():
    with pytest.raises(ValueError, match="no timestamps"):
        trim_track(Track(lats=[0, 1], lons=[0, 1]), 0, 1)


def test_trim_to_a_single_point_is_rejected():
    track = Track(lats=[0, 1, 2], lons=[0, 1, 2], times=[0, 10, 20])
    with pytest.raises(ValueError, match="fewer than 2"):
        trim_track(track, 8, 12)


# ---- extents ---------------------------------------------------------------

def test_expand_bbox_pads_by_a_fraction_of_the_span():
    assert expand_bbox((0.0, 0.0, 10.0, 20.0), 0.1) == (-1.0, -2.0, 11.0, 22.0)


def test_expand_bbox_guards_a_degenerate_span():
    """A single-point track has no span of its own; the margin is taken off
    the 1e-3 deg floor instead of collapsing to nothing."""
    w, s, e, n = expand_bbox((139.0, 35.0, 139.0, 35.0), 0.08)
    assert e - w == pytest.approx(2 * 1e-3 * 0.08)
    assert n - s == pytest.approx(2 * 1e-3 * 0.08)


def test_bbox_param_round_trip():
    assert parse_bbox_param("139.0,35.0,139.1,35.1") == (139.0, 35.0, 139.1, 35.1)


@pytest.mark.parametrize("value", [
    "1,2,3",                        # wrong arity
    "a,b,c,d",                      # not numbers
    "139.1,35.0,139.0,35.1",        # east <= west
    "139.0,35.1,139.1,35.0",        # north <= south
    "-181,35.0,139.0,35.1",         # off the globe
    "139.0,35.0,139.0001,35.1",     # below the 0.001 deg floor
])
def test_bad_bbox_is_rejected(value):
    with pytest.raises(ValueError):
        parse_bbox_param(value)


# ---- clipping --------------------------------------------------------------

BOX = (0.0, 0.0, 1.0, 1.0)


def test_a_track_wholly_inside_survives_as_one_piece():
    track = Track(lats=[0.2, 0.5, 0.8], lons=[0.2, 0.5, 0.8])
    (piece,) = clip_track(track, BOX)
    assert piece.lons == [0.2, 0.5, 0.8]


def test_leaving_the_box_cuts_exactly_on_the_border():
    track = Track(lats=[0.5, 0.5], lons=[0.5, 1.5])
    (piece,) = clip_track(track, BOX)
    assert piece.lons == [0.5, 1.0]


def test_re_entering_yields_two_pieces_not_one_shortcut():
    """The whole point of clipping per segment: no straight jump across the
    part that was left out."""
    track = Track(lats=[0.5, 0.5, 0.5], lons=[0.5, 1.5, 0.5])
    first, second = clip_track(track, BOX)
    assert first.lons == [0.5, 1.0]
    assert second.lons == [1.0, 0.5]


def test_a_track_wholly_outside_yields_nothing():
    track = Track(lats=[5.0, 6.0], lons=[5.0, 6.0])
    assert clip_track(track, BOX) == []


def test_a_segment_crossing_right_through_keeps_both_cuts():
    track = Track(lats=[0.5, 0.5], lons=[-1.0, 2.0])
    (piece,) = clip_track(track, BOX)
    assert piece.lons == [0.0, 1.0]
