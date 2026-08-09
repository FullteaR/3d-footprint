"""The 銘板: SVG ink, the frame it lives in, and the plate let into the map."""
from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import box

from app.core.nameplate import (
    BITE, INK_GAP, PLATE_THICK, _SIDE_MAX, _SIDE_MIN,
    clamp_side, inset_ink, inset_plate_outline,
    nameplate_bodies, plate_base, plate_ink, plate_levels, plate_outline,
    plate_to_print, svg_ink, to_plate_frame,
)

MODEL = box(-60.0, -60.0, 60.0, 60.0)     # a roomy model outline, plate frame

SVG_RECT = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 40">
  <path d="M10,10 H90 V30 H10 Z" fill="#000"/>
</svg>"""

SVG_TWO_SHAPES = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 40">
  <path d="M5,5 H45 V35 H5 Z" fill="#000"/>
  <path d="M55,5 H95 V35 H55 Z" fill="#000"/>
</svg>"""


# ---- the uploaded artwork --------------------------------------------------

def test_a_filled_path_becomes_ink():
    ink = svg_ink(SVG_RECT)
    assert ink.area == pytest.approx(80.0 * 20.0)


def test_svg_y_is_flipped_into_print_orientation():
    """SVG's y axis points down and the print frame's points up; artwork that
    came out mirrored would only be noticed after printing."""
    ink = svg_ink(b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">
      <path d="M0,0 H10 V2 H0 Z" fill="#000"/></svg>""")
    assert ink.bounds[1] == pytest.approx(-2.0)   # the top of the design


def test_text_elements_are_refused_with_an_explanation():
    """A <text> renders with whatever font the machine happens to have, so the
    UI asks for outlines — and the backend holds the same line."""
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><text x="0" y="0">A</text></svg>'
    with pytest.raises(ValueError, match="アウトライン化"):
        svg_ink(svg)


def test_embedded_images_are_refused():
    svg = (b'<svg xmlns="http://www.w3.org/2000/svg">'
           b'<image href="a.png" width="10" height="10"/></svg>')
    with pytest.raises(ValueError, match="image"):
        svg_ink(svg)


def test_an_svg_with_nothing_drawable_is_refused():
    with pytest.raises(ValueError, match="図形が見つかりません"):
        svg_ink(b'<svg xmlns="http://www.w3.org/2000/svg"></svg>')


def test_unparsable_bytes_are_refused():
    with pytest.raises(ValueError, match="解析できません"):
        svg_ink(b"this is not an svg")


# ---- the plate's own frame -------------------------------------------------

def test_plate_sides_are_held_to_the_printable_range():
    """The UI clamps too, but the API is callable on its own."""
    assert clamp_side(1.0) == clamp_side(-100.0) == _SIDE_MIN
    assert clamp_side(10_000.0) == _SIDE_MAX
    assert clamp_side(40.0) == 40.0


def test_the_outline_is_a_rounded_rectangle_of_the_asked_size():
    outline = plate_outline(40.0, 16.0)
    assert outline.bounds == pytest.approx((-20.0, -8.0, 20.0, 8.0))
    assert outline.area < 40.0 * 16.0            # corners are rounded off
    assert outline.area > 0.95 * 40.0 * 16.0


def test_a_tiny_plate_is_a_plain_rectangle():
    """Below a 0.4 mm radius the rounding is not worth the extra vertices."""
    assert plate_outline(1.6, 1.6).equals(box(-0.8, -0.8, 0.8, 0.8))


@pytest.mark.parametrize("at", [(0.0, 0.0, 0.0), (30.0, -12.0, 37.0)])
def test_the_plate_frame_and_the_print_frame_are_inverses(at):
    g = box(-5.0, -2.0, 5.0, 2.0)
    back = to_plate_frame(plate_to_print(g, at), at)
    assert back.equals_exact(g, 1e-9)


def test_a_turned_plate_keeps_its_size():
    g = plate_outline(40.0, 16.0)
    assert plate_to_print(g, (10.0, 5.0, 45.0)).area == pytest.approx(g.area)


def test_a_plate_hanging_off_the_model_is_refused():
    """The frontend clamps as you drag; this catches a stale position after
    the model was resized under it."""
    with pytest.raises(ValueError, match="はみ出しています"):
        inset_plate_outline(40.0, 16.0, box(-100.0, -100.0, -50.0, -50.0))


def test_a_plate_the_model_edge_only_clips_is_kept():
    cut = inset_plate_outline(40.0, 16.0, box(-10.0, -100.0, 100.0, 100.0))
    assert cut.bounds[0] == pytest.approx(-10.0)
    assert cut.area > 0.25 * 40.0 * 16.0


# ---- fitting the artwork ---------------------------------------------------

def test_the_artwork_is_scaled_and_centred_on_the_plate():
    outline = inset_plate_outline(40.0, 16.0, MODEL)
    ink = plate_ink(SVG_RECT, outline)
    x0, y0, x1, y1 = ink.bounds
    assert (x0 + x1) / 2 == pytest.approx(0.0, abs=0.05)
    assert (y0 + y1) / 2 == pytest.approx(0.0, abs=0.05)
    assert x1 - x0 <= 40.0 and y1 - y0 <= 16.0
    # Uniformly scaled: the 4:1 design stays 4:1.
    assert (x1 - x0) / (y1 - y0) == pytest.approx(4.0, rel=0.05)


def test_the_tile_is_exactly_the_artwork_s_own_box():
    outline = inset_plate_outline(40.0, 16.0, MODEL)
    ink = plate_ink(SVG_TWO_SHAPES, outline)
    tile = plate_base(ink, MODEL)
    assert tile.bounds == pytest.approx(ink.bounds)
    # Two separate shapes, one tile spanning both.
    assert tile.area > ink.area


def test_the_artwork_keeps_a_hairline_off_the_tile_s_edge():
    """Ink reaching the very edge would ask two heights to close on one line,
    which no watertight wall can do."""
    outline = inset_plate_outline(40.0, 16.0, MODEL)
    ink = plate_ink(SVG_RECT, outline)
    tile = plate_base(ink, MODEL)
    held = inset_ink(ink, tile)
    assert held.within(tile)
    assert tile.exterior.distance(held) >= INK_GAP - 1e-6


# ---- where the plaque sits in the ground -----------------------------------

def test_the_artwork_finishes_level_with_the_ground():
    """Nothing stands above the map: it reads as a plate let into a pocket."""
    z = plate_levels(z_lo=10.0, relief_mm=0.6, z_floor=-3.0)
    assert z.top == 10.0
    assert z.bottom < z.pocket < z.face < z.top
    assert z.top - z.face == pytest.approx(0.6)
    assert z.top - z.bottom == pytest.approx(PLATE_THICK)
    assert z.face - z.pocket == pytest.approx(BITE)


def test_the_tile_never_pokes_out_of_the_underside():
    """A plaque sticking out below would hold the whole print off the bed."""
    z = plate_levels(z_lo=0.2, relief_mm=0.6, z_floor=-2.6)
    assert z.bottom >= -2.6
    assert z.bottom < z.pocket < z.face <= z.top


def test_a_shallow_base_gets_a_shallower_relief_not_an_error():
    z = plate_levels(z_lo=0.0, relief_mm=0.6, z_floor=-0.5)
    assert 0.0 < z.top - z.face <= 0.6
    assert z.bottom >= -0.5


def test_no_room_at_all_is_an_error_the_user_can_act_on():
    with pytest.raises(ValueError, match="底面厚"):
        plate_levels(z_lo=0.0, relief_mm=0.6, z_floor=-0.2)


# ---- the finished plaque ---------------------------------------------------

def build_plaque(at=(0.0, 0.0, 0.0)):
    outline = inset_plate_outline(40.0, 16.0, MODEL)
    ink = plate_ink(SVG_RECT, outline)
    tile = plate_base(ink, MODEL)
    ink = inset_ink(ink, tile)
    z = plate_levels(z_lo=5.0, relief_mm=0.6, z_floor=-3.0)
    return nameplate_bodies(ink, tile, at, z), tile, z


def test_the_plaque_is_one_object_in_two_colour_layers():
    """A slicer should list one "nameplate" and give it two filaments — not a
    slab with lettering parked on it."""
    (body,), _, _ = build_plaque()
    assert set(np.unique(body.face_labels())) == {"label", "plate"}
    assert len(body.face_labels()) == len(body.mesh.faces)


def test_the_plaque_is_a_closed_shell():
    (body,), _, _ = build_plaque()
    assert body.mesh.is_watertight
    assert body.mesh.volume > 0


def test_the_plaque_spans_exactly_its_own_z_levels():
    (body,), _, z = build_plaque()
    lo, hi = body.mesh.bounds
    assert lo[2] == pytest.approx(z.bottom)
    assert hi[2] == pytest.approx(z.top)


def test_placing_the_plaque_moves_it_without_resizing_it():
    (here,), tile, _ = build_plaque()
    (there,), _, _ = build_plaque(at=(25.0, -13.0, 30.0))
    assert there.mesh.volume == pytest.approx(here.mesh.volume, rel=1e-9)
    centre = shapely.centroid(shapely.MultiPoint(there.mesh.vertices[:, :2]))
    assert centre.distance(shapely.Point(25.0, -13.0)) < 1.0
