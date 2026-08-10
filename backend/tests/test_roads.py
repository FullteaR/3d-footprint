"""The grooves cut for 大通り: which roads divide a block, and how wide.

Below about 1:40,000 no real road is a nozzle wide, so the arterials are drawn
at a printable width instead of their own. These are the rules that decide
which road earns a groove and what the groove measures.
"""
from __future__ import annotations

import pytest
import shapely
import shapely.affinity
from lxml import etree
from shapely.geometry import Polygon, box

from app.core.roads import ARTERIAL_WIDTHS_M, _road_surface, grooves

MIN = 0.8      # a 0.4 mm nozzle's minimum printable width
NARROW = 4.0   # the narrowest road that still divides a block, for these tests


def road(length, width, x0=0.0, y0=0.0):
    """A straight road of a given width, lying along x."""
    return box(x0, y0, x0 + length, y0 + width)


def cut_of(geom, min_road=NARROW):
    return grooves(geom, MIN, min_road)


# ---- which roads earn a groove ---------------------------------------------

def test_a_road_narrower_than_the_minimum_is_dropped():
    assert cut_of(road(100.0, 3.9)) is None


def test_a_road_at_the_minimum_earns_a_groove():
    assert cut_of(road(100.0, 4.1)) is not None


def test_the_alleys_go_and_the_arterial_among_them_stays():
    """The whole point: a road network is mostly 路地, and only the few
    streets wide enough to mean something come through."""
    net = shapely.union_all([road(100.0, 6.0)] +
                            [road(100.0, 1.0, y0=y) for y in (20.0, 40.0, 60.0)])
    cut = cut_of(net)
    assert isinstance(cut, Polygon)              # one groove, not four
    assert cut.bounds[1] > 0.0 and cut.bounds[3] < 6.0


def test_nothing_at_all_is_no_groove():
    assert cut_of(Polygon()) is None


# ---- how wide the groove comes out -----------------------------------------

def test_a_groove_is_at_least_one_nozzle_wide_however_thin_the_road():
    """A 30 m arterial at 1:150,000 is 0.2 mm. It has to come back printable
    or the cut is invisible, which is the whole reason for cutting at all."""
    _, y0, _, y1 = cut_of(road(100.0, 4.02)).bounds
    assert y1 - y0 >= MIN - 1e-9


def test_a_wide_road_keeps_its_extra_width():
    """An interchange or a plaza is not squeezed down to one nozzle — the
    exaggeration is a floor, not a target."""
    _, y0, _, y1 = cut_of(road(100.0, 30.0)).bounds
    assert y1 - y0 == pytest.approx(30.0 - NARROW + MIN, rel=0.02)


def test_the_groove_runs_down_the_middle_of_the_road():
    """It is grown from the road's own centre ribbon, so a groove never
    wanders off the carriageway onto the blocks behind it."""
    _, y0, _, y1 = cut_of(road(100.0, 20.0, y0=50.0)).bounds
    assert 0.5 * (y0 + y1) == pytest.approx(60.0)


def test_a_road_that_pinches_shut_is_severed_there():
    """Honest rather than convenient: where a road narrows past the minimum it
    is not wide enough to divide anything, so the groove stops."""
    pinched = shapely.union_all([road(40.0, 10.0), road(40.0, 10.0, x0=42.0),
                                 box(40.0, 4.9, 42.0, 5.1)])
    assert len(cut_of(pinched).geoms) == 2


# ---- picking the threshold -------------------------------------------------

def test_the_cut_starts_at_the_width_a_city_plan_calls_arterial():
    """22 m is a 幹線街路 — four lanes and footways — which is why there is no
    slider. The steps above it are the standard widths a 主要幹線 is planned
    at, for when even the arterial frame is more street than the plate holds."""
    assert ARTERIAL_WIDTHS_M[0] == 22.0
    assert list(ARTERIAL_WIDTHS_M) == sorted(ARTERIAL_WIDTHS_M)


# ---- reading the surface out of tran ---------------------------------------

_TRAN = "http://www.opengis.net/citygml/transportation/2.0"
_GML = "http://www.opengis.net/gml"


def road_element(exterior, holes=()):
    """A tran:Road carrying one LOD1 surface, shaped as PLATEAU writes it."""
    el = etree.Element(f"{{{_TRAN}}}Road", nsmap={"tran": _TRAN, "gml": _GML})
    ms = etree.SubElement(etree.SubElement(el, f"{{{_TRAN}}}lod1MultiSurface"),
                          f"{{{_GML}}}MultiSurface")
    poly = etree.SubElement(etree.SubElement(ms, f"{{{_GML}}}surfaceMember"),
                            f"{{{_GML}}}Polygon")

    def ring(side, pts):
        r = etree.SubElement(etree.SubElement(poly, f"{{{_GML}}}{side}"),
                             f"{{{_GML}}}LinearRing")
        # posList is "lat lon height" — the order PLATEAU writes, not x y.
        etree.SubElement(r, f"{{{_GML}}}posList").text = " ".join(
            f"{lat} {lon} 0" for lon, lat in pts)

    ring("exterior", exterior)
    for h in holes:
        ring("interior", h)
    return el


def test_a_road_surface_is_read_as_its_plan():
    """Roads are flat, so the posList's height is dropped and what comes back
    is the carriageway's area in lon/lat."""
    square = [(139.0, 35.0), (139.1, 35.0), (139.1, 35.1), (139.0, 35.1)]
    (poly,) = _road_surface(road_element(square))
    assert poly.area == pytest.approx(0.01)
    assert poly.bounds == pytest.approx((139.0, 35.0, 139.1, 35.1))


def test_a_median_stays_a_hole_in_the_carriageway():
    outer = [(139.0, 35.0), (139.4, 35.0), (139.4, 35.4), (139.0, 35.4)]
    hole = [(139.1, 35.1), (139.2, 35.1), (139.2, 35.2), (139.1, 35.2)]
    (poly,) = _road_surface(road_element(outer, [hole]))
    assert poly.area == pytest.approx(0.16 - 0.01)


def test_a_self_touching_road_is_repaired_rather_than_lost():
    """PLATEAU surfaces are occasionally invalid, and `union_all` would throw
    on one — dropping it instead would break the network exactly where the
    erosion needs it continuous."""
    bowtie = [(139.0, 35.0), (139.2, 35.2), (139.2, 35.0), (139.0, 35.2)]
    out = _road_surface(road_element(bowtie))
    assert out and all(p.is_valid for p in out)
    assert sum(p.area for p in out) == pytest.approx(0.02)


def test_a_crumb_of_a_road_is_not_left_as_a_dot():
    """A road that only just clears the threshold erodes to almost nothing,
    and dilating almost nothing gives a disc — a dot punched in a block
    rather than a street through it."""
    crumb = box(0.0, 0.0, 4.2, 4.2)             # barely over the 4.0 threshold
    assert cut_of(crumb) is None
    # ...while a real street of the same width comes through.
    assert cut_of(road(100.0, 4.2)) is not None


# ---- the automatic threshold ------------------------------------------------

class _FakeProj:
    """Just the scale: `road_cut` only needs it to turn metres into print mm."""

    def __init__(self, scale):
        self.scale = scale


def cut_for(roads_mm, scale, plate):
    """`road_cut`'s ladder, run against a road surface already in print mm."""
    from app.core.roads import ARTERIAL_WIDTHS_M, MAX_STREET_SHARE
    cut = None
    for width_m in ARTERIAL_WIDTHS_M:
        wider = grooves(roads_mm, MIN, width_m * scale)
        if wider is None:
            break
        cut = wider
        if cut.area <= MAX_STREET_SHARE * plate.area:
            break
    return cut


def test_at_a_normal_scale_the_arterials_are_taken_as_they_stand():
    """A 24 m road is a 幹線街路 and gets its groove; a 12 m 区画街路 does not,
    and no scale had to be told about either."""
    scale = 0.02                                    # 1:50,000
    plate = box(0.0, 0.0, 120.0, 120.0)
    roads = shapely.union_all([
        box(0.0, 0.0, 6000.0, 24.0),                # arterial, in metres...
        box(0.0, 500.0, 6000.0, 512.0),             # ...and a 12 m street
    ])
    roads = shapely.affinity.scale(roads, scale, scale, origin=(0, 0))
    cut = cut_for(roads, scale, plate)
    assert cut is not None
    assert cut.bounds[3] < 500.0 * scale            # the 12 m one is untouched


def test_a_plate_that_cannot_hold_the_arterials_shows_only_the_widest():
    """Past about 1:150,000 a groove is more ground than an arterial is wide,
    so taking every 幹線街路 would pave the model. The cut steps up the
    standard widths until what it draws fits."""
    scale, plate = 0.0015, box(0.0, 0.0, 40.0, 40.0)      # ~1:670,000
    grid = shapely.union_all(
        [box(x, 0.0, x + 24.0, 26000.0) for x in range(0, 26000, 900)] +
        [box(0.0, y, 26000.0, y + 40.0) for y in range(0, 26000, 4000)])
    roads = shapely.affinity.scale(grid, scale, scale, origin=(0, 0))
    every = grooves(roads, MIN, 22.0 * scale)
    assert every.area > 0.5 * plate.area            # 22 m would pave it
    cut = cut_for(roads, scale, plate)
    assert cut.area < every.area                    # so it stepped up
