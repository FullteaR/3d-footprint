"""The 出典 credit: its wording, and the groove it is debossed into.

Attribution is a hard requirement of the data licenses, not a feature — the
sentence has to name exactly the sources a given model used, and it has to
survive onto the print itself. There is no switch for turning it off, so the
only thing to test is that it says the right thing and lands on the model.
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from app.core.export import Body
from app.core.stamp import (
    _face, ascii_credit, credit_text, engrave_credit, formal_credit,
    formal_credit_lines,
)

FLAGS = [(False, False), (True, False), (False, True), (True, True)]


def slab(width=120.0, base=3.0, label="terrain") -> Body:
    """A model-sized base slab: flat underside at -base, like the terrain's."""
    mesh = trimesh.creation.box(extents=(width, 80.0, base + 10.0))
    mesh.apply_translation((width / 2, 40.0, (base + 10.0) / 2 - base))
    return Body(mesh, label)


# ---- wording ---------------------------------------------------------------

@pytest.mark.parametrize("landuse,buildings", FLAGS)
def test_the_dem_source_is_always_named(landuse, buildings):
    """Every model is built on the GSI elevation tiles, whatever else is on."""
    assert "国土地理院" in formal_credit(landuse, buildings)
    assert "GSI" in credit_text(landuse, buildings)


@pytest.mark.parametrize("landuse,buildings", FLAGS)
def test_only_the_sources_actually_used_are_named(landuse, buildings):
    """Naming a source the model never used is its own kind of wrong."""
    credit = formal_credit(landuse, buildings)
    assert ("PLATEAU" in credit) == (landuse or buildings)
    assert ("JAXA" in credit) == landuse


def test_the_sentence_states_that_the_work_is_derived():
    """政府標準利用規約 wants the derivative stated, not just the source."""
    assert formal_credit(True, True).startswith("出典: ")
    assert formal_credit(True, True).endswith("を加工して作成（3d-footprint）")


def test_the_engraved_lines_are_the_same_sentence_one_source_per_line():
    lines = formal_credit_lines(True, True)
    assert len(lines) == 3
    assert lines[0].startswith("出典: ")
    assert lines[-1].endswith("を加工して作成")
    for source in ("国土地理院", "PLATEAU", "JAXA"):
        assert any(source in line for line in lines)


@pytest.mark.parametrize("landuse,buildings", FLAGS)
def test_the_stl_header_variant_fits_in_the_header(landuse, buildings):
    """80 bytes, ASCII only — a binary STL header holds nothing else."""
    credit = ascii_credit(landuse, buildings)
    assert len(credit.encode("ascii")) <= 80
    assert credit.startswith("Source: ")
    assert "modified" in credit


# ---- the groove ------------------------------------------------------------

def test_the_credit_is_carved_into_the_underside():
    body = slab()
    before = len(body.mesh.faces)
    engrave_credit([body], True, True, 0.0, 120.0, 3.0)
    assert len(body.mesh.faces) > before
    # The groove is cut up into the base, so nothing hangs below it.
    assert body.mesh.bounds[0][2] == pytest.approx(-3.0)


def test_the_groove_is_two_layers_deep():
    body = slab()
    engrave_credit([body], True, True, 0.0, 120.0, 3.0)
    z = body.mesh.vertices[:, 2]
    floor = z[z < -3.0 + 1e-6]
    ceiling = z[(z > -3.0 + 1e-6) & (z < -2.0)]
    assert len(floor) and len(ceiling)
    assert ceiling.min() == pytest.approx(-3.0 + 0.4)


def test_a_base_too_thin_to_carve_is_left_alone():
    body = slab(base=0.8)
    before = body.mesh.faces.shape
    engrave_credit([body], True, True, 0.0, 120.0, 0.8)
    assert body.mesh.faces.shape == before


def test_a_body_that_does_not_reach_the_underside_is_left_alone():
    """Buildings, the track, the plaque — nothing above the base is touched."""
    floating = Body(trimesh.creation.box(extents=(10.0, 10.0, 2.0)), "building")
    floating.mesh.apply_translation((60.0, 40.0, 20.0))
    before = floating.mesh.faces.shape
    engrave_credit([floating], True, True, 0.0, 120.0, 3.0)
    assert floating.mesh.faces.shape == before


def test_a_body_with_per_face_labels_is_left_alone():
    """The remesh cannot carry a per-face label array through, so it declines
    rather than mislabelling the plaque's two colour layers."""
    mesh = trimesh.creation.box(extents=(120.0, 80.0, 13.0))
    mesh.apply_translation((60.0, 40.0, 13.0 / 2 - 3.0))
    body = Body(mesh, np.full(len(mesh.faces), "plate", dtype="<U8"))
    before = body.mesh.faces.shape
    engrave_credit([body], True, True, 0.0, 120.0, 3.0)
    assert body.mesh.faces.shape == before


def test_a_band_too_narrow_for_the_text_is_skipped():
    """Below _EM_MIN the groove would be unreadable; better none at all."""
    body = slab()
    before = body.mesh.faces.shape
    engrave_credit([body], True, True, 0.0, 6.0, 3.0)
    assert body.mesh.faces.shape == before


def test_the_stamp_stays_inside_the_band_it_was_given():
    body = slab()
    engrave_credit([body], True, True, 20.0, 100.0, 3.0)
    v = body.mesh.vertices
    carved = v[(v[:, 2] > -3.0 + 1e-6) & (v[:, 2] < -2.0)]
    assert len(carved)
    assert carved[:, 0].min() >= 20.0
    assert carved[:, 0].max() <= 100.0


@pytest.mark.skipif(_face() is None, reason="no font on this box")
def test_the_font_carries_the_japanese_sentence():
    """Without a CJK face the stamp degrades to the ASCII list; the shipped
    image has fonts-noto-cjk precisely so it does not."""
    assert _face().get_char_index("出")
