"""Printability massing in 3D: what the lattice keeps, fuses, thickens and drops.

Everything the buildings and bridges come out looking like is decided here, so
these are the rules rather than the plumbing: which gaps survive as streets,
what counts as too thin to print, what is allowed to float, and what the
iso-surface is allowed to smooth away.
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh
from scipy import ndimage
from shapely.geometry import box

from app.core import voxel
from app.core.voxel import (
    Lattice, MAX_VOXELS, clip_to, mass, pitch_for, rasterize, solid_from,
    solidify, stand_on, to_mesh,
)

MIN = 0.8                    # a 0.4 mm nozzle's minimum printable width
PITCH = 0.5 * MIN            # what `pitch_for` picks: the nozzle is two voxels


def lat_of(occ, pitch=PITCH, origin=(0.0, 0.0, 0.0)) -> Lattice:
    return Lattice(np.asarray(occ, bool), np.asarray(origin, float), pitch)


def cube(size=1.0, at=(0.0, 0.0, 0.0)):
    """(verts, faces) of a box, the shape the providers hand over."""
    m = trimesh.creation.box(extents=(size, size, size))
    m.apply_translation(at)
    return np.asarray(m.vertices, float), np.asarray(m.faces)


def bodies(occ) -> int:
    return int(ndimage.label(occ, structure=np.ones((3, 3, 3), bool))[1])


# ---- pitch -----------------------------------------------------------------

def test_the_lattice_is_half_a_nozzle_across():
    """One voxel is the nozzle radius, which is what makes the closing below a
    closing at the nozzle rather than at some number of voxels."""
    verts, _ = cube(40.0)
    assert pitch_for(verts, MIN) == pytest.approx(0.4)


def test_a_model_too_big_to_afford_is_coarsened_rather_than_refused():
    verts = np.array([[0.0, 0.0, 0.0], [4000.0, 4000.0, 400.0]])
    pitch = pitch_for(verts, 0.05)
    assert pitch > 0.5 * 0.05
    assert np.prod(np.ceil((verts[1] - verts[0]) / pitch) + 4) <= MAX_VOXELS


def test_with_no_minimum_width_the_model_sets_the_scale():
    """Nothing to key off the nozzle, so the pitch comes off the model's own
    longest edge — fine enough that the shape survives being resampled, and
    still inside the budget."""
    verts = np.array([[0.0, 0.0, 0.0], [120.0, 120.0, 12.0]])
    pitch = pitch_for(verts, 0.0)
    assert pitch >= 120.0 / 600.0
    assert pitch < 1.0                              # far finer than a nozzle
    assert np.prod(np.ceil((verts[1] - verts[0]) / pitch) + 4) <= MAX_VOXELS


# ---- rasterising -----------------------------------------------------------

def test_a_surface_marks_the_voxels_it_passes_through():
    lat = rasterize(*cube(8.0), PITCH)
    assert lat.occ.any()
    # A box is hollow until it is filled: its middle is not marked.
    mid = tuple(s // 2 for s in lat.occ.shape)
    assert not lat.occ[mid]


def test_nothing_to_rasterise_is_no_lattice():
    verts, _ = cube(1.0)
    assert rasterize(verts, np.empty((0, 3), int), PITCH) is None
    assert rasterize(*cube(1.0), 0.0) is None


def test_the_margin_leaves_empty_room_around_the_geometry():
    """`to_mesh` needs the surface to fall away inside the array, and `mass`
    needs somewhere to dilate into."""
    lat = rasterize(*cube(8.0), PITCH, margin=2.0)
    assert not lat.occ[0].any() and not lat.occ[-1].any()
    assert not lat.occ[:, :, 0].any() and not lat.occ[:, :, -1].any()


# ---- solidifying -----------------------------------------------------------

def test_a_closed_surface_becomes_a_solid():
    lat = solidify(rasterize(*cube(8.0), PITCH, margin=1.0))
    mid = tuple(s // 2 for s in lat.occ.shape)
    assert lat.occ[mid]


def test_a_surface_with_a_hole_in_it_stays_hollow_rather_than_flooding():
    """PLATEAU records are not all closed. One broken building costs its own
    interior; it must not fill the scene around it."""
    verts, faces = cube(8.0)
    lat = solidify(rasterize(verts, faces[2:], PITCH, margin=1.0))   # a face short
    mid = tuple(s // 2 for s in lat.occ.shape)
    assert not lat.occ[mid]
    assert lat.occ.mean() < 0.2


# ---- the massing rules -----------------------------------------------------

def two_slabs(gap_vox: int):
    """Two blocks with a street of `gap_vox` voxels between them."""
    occ = np.zeros((30, 24, 12), bool)
    occ[4:13, 5:19, 2:9] = True
    occ[13 + gap_vox:22 + gap_vox, 5:19, 2:9] = True
    return occ


@pytest.mark.parametrize("gap,expected", [(1, 1), (2, 1), (3, 2), (6, 2)],
                         ids=["half a nozzle", "one nozzle", "wider", "a street"])
def test_the_nozzle_decides_what_is_one_block(gap, expected):
    """The closing, which is the city-block merge: an alley the nozzle cannot
    lay down closes up, and anything wider than one nozzle stays a gap."""
    out = mass(lat_of(two_slabs(gap)), MIN)
    assert bodies(out.occ) == expected


def test_a_block_thicker_than_the_nozzle_comes_back_its_own_size():
    """The rule that decides what is too thin has to leave everything else
    alone. Comparing against the plain opening instead reports every convex
    corner as thin, pads each one by half a nozzle, and inflates this by 22%."""
    occ = np.zeros((24, 24, 24), bool)
    occ[7:17, 7:17, 7:17] = True
    assert mass(lat_of(occ), MIN).occ.sum() == occ.sum()


def test_a_sheet_thinner_than_the_nozzle_is_grown_out_to_it():
    occ = np.zeros((24, 24, 24), bool)
    occ[6:18, 6:18, 12] = True                      # one voxel: half a nozzle
    out = mass(lat_of(occ), MIN).occ
    thickness = out[12, 12].sum()
    assert thickness * PITCH >= MIN
    assert bodies(out) == 1


def test_asking_for_no_minimum_width_leaves_the_lattice_alone():
    occ = np.zeros((12, 12, 12), bool)
    occ[4:8, 4:8, 6] = True
    assert (mass(lat_of(occ), 0.0).occ == occ).all()


# ---- the print outline -----------------------------------------------------

def test_the_outline_cuts_the_lattice_flush():
    occ = np.ones((20, 20, 6), bool)
    out = clip_to(lat_of(occ), box(0.0, 0.0, 4.0, 8.0))
    kept = out.occ.any(axis=2)
    assert kept[:9, :19].all()                      # inside, every column stands
    assert not kept[12:, :].any()                   # past 4 mm at 0.4 mm a voxel
    assert (out.occ[kept].all()) and not out.occ[~kept].any()


def test_a_hole_in_the_outline_is_cut_out_of_the_lattice_too():
    """The road grooves reach the massing as holes punched in the outline
    polygon; losing them would pave the arterials back over."""
    occ = np.ones((20, 20, 6), bool)
    hole = box(0.0, 0.0, 8.0, 8.0).difference(box(2.0, 2.0, 6.0, 6.0))
    kept = clip_to(lat_of(occ), hole).occ.any(axis=2)
    assert kept[1, 1] and not kept[10, 10]          # rim stands, middle is cut


def test_no_outline_at_all_keeps_everything():
    lat = lat_of(np.ones((6, 6, 6), bool))
    assert clip_to(lat, None) is lat


# ---- standing it up --------------------------------------------------------

def flat_floor(shape, mm):
    return np.full(shape[:2], mm, float)


def test_what_reaches_the_ground_is_sunk_into_it():
    """A building resting on the terrain as a separate shell would print as a
    seam; it has to be embedded so the two fuse."""
    occ = np.zeros((8, 8, 20), bool)
    occ[2:6, 2:6, 10:16] = True
    out = stand_on(lat_of(occ), flat_floor(occ.shape, 10 * PITCH),
                   embed=2 * PITCH, min_feature=MIN)
    assert out.occ[4, 4, 8:16].all()                # down two voxels below 10
    assert not out.occ[4, 4, :8].any()


def test_a_deck_left_in_mid_air_is_propped_down_to_the_terrain():
    """PLATEAU does not always model the piers, and nothing floating prints."""
    occ = np.zeros((40, 8, 24), bool)
    occ[4:36, 2:6, 18:20] = True                    # a deck, nothing under it
    out = stand_on(lat_of(occ), flat_floor(occ.shape, 0.0), embed=PITCH,
                   spacing=8.0, min_feature=MIN)
    assert bodies(out.occ) == 1
    assert out.occ[:, :, 0].any()                   # it now reaches the ground


def test_a_speck_in_mid_air_is_left_where_it_is():
    """Propping every stray voxel would grow a forest of sticks out of the
    noise in the data; below a nozzle cube there is nothing worth standing up."""
    occ = np.zeros((20, 20, 20), bool)
    occ[10, 10, 15] = True
    out = stand_on(lat_of(occ), flat_floor(occ.shape, 0.0), embed=PITCH,
                   spacing=4.0, min_feature=MIN)
    assert out.occ.sum() == 1


def test_without_a_spacing_nothing_is_propped():
    """Buildings are snapped onto the terrain already; only bridges ask."""
    occ = np.zeros((20, 20, 20), bool)
    occ[4:16, 4:16, 14:18] = True
    out = stand_on(lat_of(occ), flat_floor(occ.shape, 0.0), embed=PITCH)
    assert (out.occ == occ).all()


# ---- reading it back -------------------------------------------------------

def test_the_surface_is_one_watertight_solid():
    occ = np.zeros((20, 20, 20), bool)
    occ[5:15, 5:15, 5:15] = True
    mesh = to_mesh(lat_of(occ), thinnest=2.0)
    assert mesh.is_watertight
    assert mesh.body_count == 1
    assert mesh.volume == pytest.approx((10 * PITCH) ** 3, rel=0.15)


def test_the_thinnest_thing_in_the_lattice_survives_the_smoothing():
    """The blur is what takes the staircase off the surface, and it is also what
    can erase a wall: too wide and a one-voxel sheet peaks under the iso-level
    and is simply gone. `thinnest` is what holds it back."""
    occ = np.zeros((24, 24, 24), bool)
    occ[6:18, 6:18, 12] = True
    mesh = to_mesh(lat_of(occ), thinnest=1.0)
    assert mesh is not None and mesh.is_watertight
    assert mesh.volume > 0.0


def test_an_empty_lattice_is_no_mesh():
    assert to_mesh(lat_of(np.zeros((6, 6, 6), bool))) is None


# ---- the whole pipeline ----------------------------------------------------

def stepped_tower():
    """Three stacked boxes, each narrower than the one below."""
    parts = [trimesh.creation.box(extents=(w, w, 6.0)).apply_translation((0, 0, z))
             for w, z in ((18.0, 3.0), (10.0, 9.0), (4.0, 15.0))]
    m = trimesh.util.concatenate(parts)
    return np.asarray(m.vertices, float), np.asarray(m.faces)


def test_a_feature_that_narrows_as_it_rises_still_narrows_when_printed():
    """The whole reason the massing moved into three dimensions: a footprint
    prism would print all three storeys at the width of the widest."""
    mesh = solid_from(*stepped_tower(), MIN)
    assert mesh is not None and mesh.is_watertight
    lo, hi = mesh.bounds
    at = lambda z: mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    widths = []
    for z in (4.0, 10.0, 16.0):
        sec = at(z)
        b = sec.bounds
        widths.append(float(b[1][0] - b[0][0]))
    assert widths[0] > widths[1] > widths[2]
    assert widths[0] == pytest.approx(18.0, abs=2.0)
    assert widths[2] == pytest.approx(4.0, abs=2.0)


def test_nothing_comes_out_of_the_pipeline_thinner_than_the_nozzle():
    verts = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [20.0, 20.0, 0.0],
                      [0.0, 20.0, 0.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])        # a bare sheet, no thickness
    mesh = solid_from(verts, faces, MIN)
    assert mesh is not None and mesh.is_watertight
    assert mesh.bounds[1][2] - mesh.bounds[0][2] >= MIN


def test_no_faces_at_all_is_no_solid():
    verts, _ = cube(4.0)
    assert solid_from(verts, np.empty((0, 3), int), MIN) is None
