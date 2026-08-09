"""Land-use colouring: the class maps, the GML parse, and generalisation.

`generalize` is the one piece of colour post-processing there is, and it is
about the printer rather than the data: it dissolves colour features finer than
a requested print size so what is left can actually be laid down.
"""
from __future__ import annotations

import io

import numpy as np
import pytest

from app.core import jaxa
from app.core.coloring import (
    LUSE_CATEGORY, UNCLASSIFIED, _HOLE, _PALETTE, _mesh2_codes, _parse_luse,
    category_grid, generalize,
)

GML = """<?xml version="1.0" encoding="UTF-8"?>
<core:CityModel xmlns:core="http://www.opengis.net/citygml/2.0"
                xmlns:luse="http://www.opengis.net/citygml/landuse/2.0"
                xmlns:gml="http://www.opengis.net/gml">
  {members}
</core:CityModel>"""

MEMBER = """<core:cityObjectMember><luse:LandUse>
  <luse:class>{code}</luse:class>
  <luse:lod1MultiSurface><gml:MultiSurface><gml:surfaceMember>
    <gml:Polygon>
      <gml:exterior><gml:LinearRing><gml:posList>{outer}</gml:posList>
      </gml:LinearRing></gml:exterior>
      {holes}
    </gml:Polygon>
  </gml:surfaceMember></gml:MultiSurface></luse:lod1MultiSurface>
</luse:LandUse></core:cityObjectMember>"""

HOLE = """<gml:interior><gml:LinearRing><gml:posList>{ring}</gml:posList>
       </gml:LinearRing></gml:interior>"""


def ring(pts, z=True):
    """PLATEAU posLists are "lat lon height" tuples."""
    return " ".join(f"{lat} {lon}" + (" 0.0" if z else "") for lat, lon in pts)


SQUARE = [(35.0, 139.0), (35.0, 139.1), (35.1, 139.1), (35.1, 139.0), (35.0, 139.0)]
INNER = [(35.02, 139.02), (35.02, 139.08), (35.08, 139.08), (35.08, 139.02),
         (35.02, 139.02)]


def parse(members: str):
    return _parse_luse(io.BytesIO(GML.format(members=members).encode()))


# ---- class maps ------------------------------------------------------------

def test_every_luse_class_lands_in_the_palette():
    assert set(LUSE_CATEGORY.values()) <= set(_PALETTE)
    assert UNCLASSIFIED not in LUSE_CATEGORY.values()   # 231 不明 falls through


def test_every_jaxa_class_lands_in_the_palette():
    """JAXA is gap-fill only, but it is painted through the same palette."""
    assert set(jaxa.CLASS_CATEGORY.values()) <= set(_PALETTE)
    assert 0 not in jaxa.CLASS_CATEGORY                 # 0 is no-data


def test_the_water_and_road_classes_are_the_ones_expected():
    assert LUSE_CATEGORY[204] == "water"     # 水面
    assert LUSE_CATEGORY[203] == "forest"    # 山林
    assert LUSE_CATEGORY[211] == "urban"     # 住宅用地
    assert LUSE_CATEGORY[215] == "road"      # 道路用地
    assert 231 not in LUSE_CATEGORY          # 不明


# ---- mesh codes ------------------------------------------------------------

def test_a_tokyo_bbox_asks_for_the_right_mesh():
    """533946 is the 2nd-level mesh over Tokyo Station — a wrong formula here
    would fetch some other city's land use entirely."""
    assert _mesh2_codes((139.76, 35.68, 139.77, 35.69)) == ["533946"]


def test_a_bbox_spanning_a_mesh_edge_asks_for_both():
    """Meshes are 1/8 deg of longitude wide; 139.75 is one of the edges."""
    assert _mesh2_codes((139.74, 35.68, 139.76, 35.69)) == ["533945", "533946"]


# ---- GML parsing -----------------------------------------------------------

def test_parcels_come_back_as_rings_with_their_class():
    coords, starts, codes, feats = parse(
        MEMBER.format(code=204, outer=ring(SQUARE), holes=""))
    assert codes.tolist() == [204]
    assert len(feats) == 2 and len(starts) == 2
    assert coords.shape == (5, 2)
    # lon/lat order (the source is lat/lon), so it can be drawn on the grid.
    assert coords[0].tolist() == [139.0, 35.0]


def test_two_dimensional_poslists_are_read_too():
    coords, _, codes, _ = parse(
        MEMBER.format(code=203, outer=ring(SQUARE, z=False), holes=""))
    assert codes.tolist() == [203]
    assert coords[0].tolist() == [139.0, 35.0]


def test_interior_rings_are_marked_as_holes():
    _, _, codes, feats = parse(MEMBER.format(
        code=211, outer=ring(SQUARE), holes=HOLE.format(ring=ring(INNER))))
    assert codes.tolist() == [211, _HOLE]
    assert feats.tolist() == [0, 2]      # both rings belong to one parcel


def test_parcels_with_holes_are_painted_first():
    """A parcel sitting inside another's hole has to be drawn after the hole
    punched it, or the painter's algorithm erases it."""
    simple = MEMBER.format(code=204, outer=ring(INNER), holes="")
    holed = MEMBER.format(code=211, outer=ring(SQUARE),
                          holes=HOLE.format(ring=ring(INNER)))
    _, _, codes, _ = parse(simple + holed)
    assert codes.tolist() == [211, _HOLE, 204]


def test_an_unparsable_class_does_not_lose_the_parcel():
    _, _, codes, _ = parse(
        MEMBER.format(code="", outer=ring(SQUARE), holes=""))
    assert codes.tolist() == [0]        # unmapped -> painted as unclassified


def test_a_file_with_no_parcels_parses_to_nothing():
    coords, starts, codes, feats = parse("")
    assert len(codes) == 0 and len(coords) == 0
    assert starts.tolist() == [0] and feats.tolist() == [0]


def test_degenerate_rings_are_dropped():
    two_points = ring([(35.0, 139.0), (35.1, 139.1)])
    _, _, codes, _ = parse(MEMBER.format(code=204, outer=two_points, holes=""))
    assert len(codes) == 0


# ---- generalisation --------------------------------------------------------

def checkerboard(n=40):
    yy, xx = np.mgrid[0:n, 0:n]
    return np.where((yy // 4 + xx // 4) % 2 == 0, "water", "forest").astype("<U8")


def test_generalize_is_a_no_op_below_one_cell():
    """Nothing on the grid is finer than one cell, so there is nothing to do."""
    cats = checkerboard()
    assert generalize(cats, 1.0) is cats
    assert generalize(cats, 0.0) is cats


def test_generalize_is_a_no_op_on_a_single_colour():
    cats = np.full((20, 20), "forest", dtype="<U8")
    assert generalize(cats, 8.0) is cats


def test_generalize_dissolves_a_speck_into_its_surroundings():
    cats = np.full((40, 40), "forest", dtype="<U8")
    cats[20, 20] = "water"                       # one cell: unprintable
    out = generalize(cats, 6.0)
    assert (out == "forest").all()


def test_generalize_keeps_a_feature_bigger_than_the_floor():
    cats = np.full((40, 40), "forest", dtype="<U8")
    cats[10:30, 10:30] = "water"
    out = generalize(cats, 6.0)
    assert (out[15:25, 15:25] == "water").all()
    assert (out[:5, :5] == "forest").all()


def test_generalize_leaves_a_straight_border_straight():
    """Blurring a straight edge does not move it — a reclaimed coastline or an
    urban parcel edge has to come back exactly where it was."""
    cats = np.full((40, 40), "forest", dtype="<U8")
    cats[:, :20] = "water"
    out = generalize(cats, 6.0)
    assert (out == cats).all()


def test_generalize_never_invents_a_colour():
    out = generalize(checkerboard(), 9.0)
    assert set(np.unique(out)) <= {"water", "forest"}
    assert out.shape == (40, 40)
    assert out.dtype == checkerboard().dtype


def test_generalize_is_deterministic():
    cats = checkerboard()
    assert (generalize(cats, 5.0) == generalize(cats, 5.0)).all()


# ---- the composed grid -----------------------------------------------------

def test_a_grid_too_small_to_paint_is_skipped(make_grid_fn=None):
    """Guarded before anything is fetched, so this stays offline."""
    from app.core.terrain import ElevationGrid
    tiny = ElevationGrid(elev=np.zeros((1, 1)), lons=np.array([139.0]),
                         lats=np.array([35.0]))
    assert category_grid(tiny) is None
