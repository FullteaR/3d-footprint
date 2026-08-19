"""Printability massing in three dimensions: the real geometry, made thick.

At print scale a PLATEAU LOD2 feature's own detail is finer than the FDM nozzle
and cannot be laid down. The answer used to be to throw the shape away and print
the plan instead — one footprint prism per building, one deck slab per bridge.
That is cheap, and it is a lie about anything that changes shape as it rises:
Tokyo Tower came out as the 127 m column of its own feet, and a suspension
bridge as a plate on stilts.

So ask the printer's question rather than the plan's. The scene is rasterised
into a single occupancy lattice at half the nozzle width, and printability is
settled there, in three dimensions, with the same morphology the 2D massing used
to do in two:

  * a **closing** — dilate by the nozzle radius, then erode by the same radius —
    fuses whatever stands closer together than one nozzle width and reopens
    every gap wider than that at its *true* width. That is the city-block merge,
    one dimension up: alleys and lot lines disappear, streets survive as streets.
  * whatever is *still* thinner than the nozzle after that — a tower leg, a
    bridge hanger, a parapet — is grown out to it, and only it. An opening finds
    those parts and they alone are dilated. Dilating everything instead is the
    obvious shortcut and it is wrong: it costs a city 72% of its volume and
    paves its streets over.

The lattice is then read back with marching cubes over its signed distance, so
what comes out is a smooth watertight solid rather than a staircase of cubes.

The cost is set by the size of the print, not by the number of features: a
120 mm model is a few million voxels whether it holds one bridge or forty
thousand buildings, which is why doing every building this way is *faster* than
massing each one separately. And coarsen the scale and the nozzle swallows the
detail on its own — there is no rule here about when to stop simplifying,
because the nozzle is the rule.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage import measure

# The lattice is sized by the print, so this caps memory rather than ambition:
# the distance transforms below allocate float64 over the whole grid, several
# times the bool. Past this the pitch is coarsened until it fits.
MAX_VOXELS = 32_000_000
# Points held at once while rasterising triangles. A single plate the width of
# the model can be tens of thousands of samples; this keeps that bounded.
_SAMPLE_BUDGET = 4_000_000
# Blur applied to the occupancy before the iso-surface is taken (`to_mesh`), as
# a fraction of the thinnest feature the lattice is known to hold. Two things
# bound it. Above, the feature has to survive: blur wider than it and its peak
# sinks under the iso-level and it is simply gone — a lone one-voxel sheet peaks
# at 0.499 under a blur of 0.8. Below, the corners have to survive: this is what
# takes the voxel staircase off a surface, and spent freely it also rounds every
# building in a city off into a lump. Half of what the feature can bear keeps
# both, and a corner stays a corner.
_ISO_BLUR = 0.2


@dataclass(frozen=True)
class Lattice:
    """A scene-sized occupancy grid in print millimetres."""

    occ: np.ndarray          # (nx, ny, nz) bool, True where there is material
    origin: np.ndarray       # (3,) mm at index (0, 0, 0)
    pitch: float             # mm per voxel

    def x_centres(self) -> np.ndarray:
        return self.origin[0] + np.arange(self.occ.shape[0]) * self.pitch

    def y_centres(self) -> np.ndarray:
        return self.origin[1] + np.arange(self.occ.shape[1]) * self.pitch

    def level(self, z_mm) -> np.ndarray:
        """The voxel index a millimetre height falls on."""
        return np.rint((np.asarray(z_mm) - self.origin[2]) / self.pitch).astype(np.int64)


def pitch_for(verts: np.ndarray, min_feature: float) -> float:
    """Voxel size: half a nozzle, unless the model is too big to afford it.

    Half a nozzle is the coarsest pitch at which the closing below still has a
    radius of a whole voxel to work with, and there is nothing finer worth
    resolving — the printer cannot lay it down. Where no minimum width is asked
    for at all the nozzle cannot set the scale, so the model's own longest edge
    does, finely enough that the shape survives resampling.
    """
    extent = verts.max(axis=0) - verts.min(axis=0)
    pitch = 0.5 * min_feature if min_feature > 0 else float(extent.max()) / 600.0
    if pitch <= 0:
        return 0.0
    while np.prod(np.ceil(extent / pitch) + 4) > MAX_VOXELS:
        pitch *= 1.25
    return pitch


def rasterize(verts: np.ndarray, faces: np.ndarray, pitch: float,
              margin: float = 0.0, bounds=None) -> Lattice | None:
    """Mark every voxel a triangle passes through.

    Triangles are covered by sampling rather than scan-converted: at this pitch
    a PLATEAU surface is a few points per triangle, and sampling costs no
    special cases at the edges where a scan-converter would need them.
    """
    if len(faces) == 0 or pitch <= 0:
        return None
    used = np.unique(faces)
    if bounds is None:
        lo = verts[used].min(axis=0) - margin
        hi = verts[used].max(axis=0) + margin
    else:
        lo, hi = (np.asarray(b, float) for b in bounds)
    shape = tuple(np.ceil((hi - lo) / pitch).astype(int) + 1)
    if min(shape) < 1:
        return None
    occ = np.zeros(shape, bool)
    limit = np.array(shape) - 1

    def mark(pts: np.ndarray) -> None:
        ijk = np.clip(((pts - lo) / pitch).round().astype(np.int32), 0, limit)
        occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True

    mark(verts[used])
    a = verts[faces[:, 0]]
    e1, e2 = verts[faces[:, 1]] - a, verts[faces[:, 2]] - a
    step = 0.7 * pitch                       # a hair under one voxel, so no gaps
    n = np.maximum(np.ceil(np.maximum(np.linalg.norm(e1, axis=1),
                                      np.linalg.norm(e2, axis=1)) / step), 1)
    n = n.astype(np.int64)
    for k in np.unique(n):
        rows = np.flatnonzero(n == k)
        t = np.linspace(0.0, 1.0, int(k) + 1)
        uu, vv = np.meshgrid(t, t)
        inside = (uu + vv) <= 1.0 + 1e-9
        uu, vv = uu[inside], vv[inside]
        per = max(1, _SAMPLE_BUDGET // max(len(uu) * 3, 1))
        for s in range(0, len(rows), per):
            j = rows[s:s + per]
            mark((a[j][:, None, :] + uu[None, :, None] * e1[j][:, None, :]
                  + vv[None, :, None] * e2[j][:, None, :]).reshape(-1, 3))
    return Lattice(occ, lo, pitch)


def solidify(lat: Lattice) -> Lattice:
    """Fill what the surface encloses, so a building is a solid and not a shell.

    A feature whose mesh has a hole in it stays hollow rather than flooding the
    scene: the fill only reaches cavities that are not connected to the outside,
    so a broken record costs its own interior and nothing else.
    """
    return Lattice(ndimage.binary_fill_holes(lat.occ), lat.origin, lat.pitch)


def _ball(r: float) -> np.ndarray:
    n = int(np.floor(r))
    g = np.mgrid[-n:n + 1, -n:n + 1, -n:n + 1]
    return np.linalg.norm(g, axis=0) <= r + 1e-9


def _dilate(occ: np.ndarray, r: float) -> np.ndarray:
    return ndimage.binary_dilation(occ, structure=_ball(r))


def _erode(occ: np.ndarray, r: float) -> np.ndarray:
    return ndimage.binary_erosion(occ, structure=_ball(r))


def mass(lat: Lattice, min_feature: float) -> Lattice:
    """Close at the nozzle radius, then thicken whatever is still too thin.

    The two halves answer different questions and neither one alone is enough.
    The closing decides what is *one* object: anything nearer than a nozzle
    width becomes a single body and stays one, which is how a city block forms
    out of its buildings and how a girder joins the deck it carries. Because it
    erodes back by the radius it dilated, a gap wider than the nozzle reopens at
    the width it really has, rather than printing as a crack.

    The thickening then decides what is *printable*. An opening at the same
    radius passes the parts a nozzle-wide ball fits inside; what it misses is
    material too thin to lay down, and dilating only that brings it up to width
    while leaving everything else exactly where the data put it.

    The opening is grown back by *twice* the radius before that comparison, and
    the factor is the whole difference between this working and not. An opening
    always rounds a convex corner off, so comparing against it directly reports
    every corner in the model as too thin — which sounds harmless and is not:
    thickening them pads every edge by half a nozzle, inflates a solid block by
    22% of its volume, and closes the streets from both sides. Grown back by 2r
    a solid block comes through at exactly its own size, and only material that
    is genuinely thinner than the nozzle is still uncovered.
    """
    if min_feature <= 0:
        return lat
    # A whole voxel is the floor: where the print is so large that the lattice
    # had to be coarsened past the nozzle (see `pitch_for`), the voxel itself is
    # already wider than the nozzle and becomes the minimum feature in its place.
    r = max(0.5 * min_feature / lat.pitch, 1.0)
    # Binary morphology rather than a thresholded distance transform: at this
    # pitch the ball is one voxel across and the two agree exactly, but the
    # transform costs twenty times as much for the same answer.
    closed = _erode(_dilate(lat.occ, r), r)
    thin = closed & ~_dilate(_erode(closed, r), 2.0 * r)
    return Lattice(closed | _dilate(thin, r), lat.origin, lat.pitch)


def clip_to(lat: Lattice, outline) -> Lattice:
    """Cut the lattice flush with the model outline (a polygon in print mm).

    Applied last, for the reason the 2D massing applied its clip last: the
    thickening above is exactly what pushes material past the model edge, and
    trimming any earlier would let it grow straight back out.
    """
    if outline is None or outline.is_empty:
        return lat
    nx, ny = lat.occ.shape[:2]
    img = Image.new("1", (nx, ny), 0)
    draw = ImageDraw.Draw(img)
    for poly in getattr(outline, "geoms", [outline]):
        if poly.is_empty:
            continue
        rings = [(poly.exterior, 1)] + [(r, 0) for r in poly.interiors]
        for ring, fill in rings:
            xy = np.asarray(ring.coords, dtype=np.float64)[:, :2]
            ij = (xy - lat.origin[:2]) / lat.pitch
            draw.polygon([tuple(p) for p in ij], fill=fill, outline=fill)
    keep = np.asarray(img, bool).T          # PIL is (row, col) = (y, x)
    return Lattice(lat.occ & keep[:, :, None], lat.origin, lat.pitch)


def stand_on(lat: Lattice, floor: np.ndarray, *, embed: float,
             spacing: float = 0.0, min_feature: float = 0.0) -> Lattice:
    """Sink what already reaches the ground into it; prop up what does not.

    `floor` is the terrain surface in print millimetres, one height per column
    of the lattice. Anything whose underside is already at that height is fused
    to it by extending `embed` below, so a building does not sit on the terrain
    as a separate shell. Anything left standing in mid-air — a deck whose piers
    PLATEAU never modelled — is given props down to the terrain at `spacing`,
    because a bridge floating over a valley cannot be printed at all.

    Props are drawn one voxel wide and left that way: `mass` runs after this and
    thickens them out to the nozzle by the same rule it thickens a tower leg,
    so there is no second notion of how wide a support has to be.
    """
    occ = lat.occ.copy()
    nz = occ.shape[2]
    k_floor = np.clip(lat.level(floor), 0, nz - 1)
    k_embed = max(int(round(embed / lat.pitch)), 1)
    kk = np.arange(nz)

    standing = occ.any(axis=2)
    low = np.where(standing, occ.argmax(axis=2), nz)
    # A column is footed if its lowest material is already at the terrain (one
    # voxel of slack for the rounding above).
    footed = standing & (low <= k_floor + 1)
    base = np.where(footed, np.maximum(k_floor - k_embed, 0), nz)
    occ |= (kk[None, None, :] >= base[:, :, None]) & (kk[None, None, :] < np.where(
        footed, np.maximum(low, k_floor), 0)[:, :, None])

    if spacing <= 0:
        return Lattice(occ, lat.origin, lat.pitch)

    # Whatever is not connected to something footed is floating.
    label, count = ndimage.label(occ, structure=np.ones((3, 3, 3), bool))
    if count == 0:
        return Lattice(occ, lat.origin, lat.pitch)
    fi, fj = np.nonzero(footed)
    grounded = np.zeros(count + 1, bool)
    grounded[label[fi, fj, base[fi, fj]]] = True
    grounded[0] = True

    step = max(int(round(spacing / lat.pitch)), 1)
    half = max(int(round(0.5 * min_feature / lat.pitch)), 0)
    floor_cube = max(int(round(min_feature / lat.pitch)), 1) ** 3
    for lab, box in enumerate(ndimage.find_objects(label), start=1):
        if grounded[lab] or box is None:
            continue
        part = label[box] == lab
        if part.sum() < floor_cube:
            continue                         # a speck in mid-air is not a bridge
        i0, j0, k0 = box[0].start, box[1].start, box[2].start
        plan = part.any(axis=2)
        rows, cols = np.nonzero(plan)
        pick = ((rows + i0) % step == 0) & ((cols + j0) % step == 0)
        if not pick.any():                   # smaller than one prop spacing
            pick = np.zeros(len(rows), bool)
            pick[len(rows) // 2] = True
        for i, j in zip(rows[pick], cols[pick]):
            top = k0 + int(part[i, j].argmax())
            gi, gj = i + i0, j + j0
            sl = (slice(max(gi - half, 0), gi + half + 1),
                  slice(max(gj - half, 0), gj + half + 1),
                  slice(max(int(k_floor[gi, gj]) - k_embed, 0), top))
            occ[sl] = True
    return Lattice(occ, lat.origin, lat.pitch)


def to_mesh(lat: Lattice, thinnest: float = 1.0) -> trimesh.Trimesh | None:
    """Read the lattice back as one watertight solid.

    Marching cubes over a *blurred* occupancy rather than the raw one: the
    iso-surface then lands between samples instead of on the voxel staircase,
    which is the difference between a model that looks massed and one that looks
    like it was built out of sugar cubes. `thinnest` is the narrowest feature
    the lattice still holds, in voxels, and it is what the blur is allowed to
    spend — see `_ISO_BLUR`.
    """
    blur = _ISO_BLUR * max(thinnest, 1.0)
    # Enough empty margin for the blur to fall away inside the array: material
    # bleeding to the border would leave the iso-surface open there, and an open
    # surface is not a solid.
    pad = int(np.ceil(3.0 * blur)) + 1
    occ = np.pad(lat.occ, pad)
    if not occ.any() or occ.all():
        return None
    field = 0.5 - ndimage.gaussian_filter(occ.astype(np.float32), blur)
    try:
        verts, faces, _, _ = measure.marching_cubes(
            field, level=0.0, spacing=(lat.pitch,) * 3)
    except (ValueError, RuntimeError):
        return None
    if len(faces) == 0:
        return None
    mesh = trimesh.Trimesh(verts + lat.origin - pad * lat.pitch, faces,
                           process=False)
    # Deliberately not welded. Marching cubes already shares its vertices, and
    # the only ones left coincident are where two solids meet at a single point;
    # welding those turns a watertight mesh into a non-manifold one.
    mesh.fix_normals()
    return mesh


def terrain_floor(lat: Lattice, proj) -> np.ndarray:
    """The terrain surface (print mm) under every column of a lattice."""
    xx, yy = np.meshgrid(lat.x_centres(), lat.y_centres(), indexing="ij")
    return proj.sample_z(proj.lon_of(xx), proj.lat_of(yy))


def _lowest_terrain(proj, lo, hi, n: int = 64) -> float:
    """The lowest the terrain gets under an xy extent, coarsely sampled."""
    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], n),
                         np.linspace(lo[1], hi[1], n), indexing="ij")
    return float(np.min(proj.sample_z(proj.lon_of(xx), proj.lat_of(yy))))


def solid_from(verts: np.ndarray, faces: np.ndarray, min_feature: float, *,
               proj=None, embed: float = 0.0, spacing: float = 0.0,
               outline=None) -> trimesh.Trimesh | None:
    """The whole pipeline: real geometry in print mm -> one printable solid.

    The order is the argument. Filling and standing come first, because they say
    what material is there; massing comes next, because it says how thin that
    material is allowed to be; the outline is cut last, because massing is
    exactly what pushes material past it.
    """
    if len(faces) == 0:
        return None
    used = verts[np.unique(faces)]
    pitch = pitch_for(used, min_feature)
    if pitch <= 0:
        return None
    margin = 2.0 * max(min_feature, pitch)
    lo, hi = used.min(axis=0) - margin, used.max(axis=0) + margin
    if proj is not None:
        # The lattice has to reach the ground it is going to stand on: a deck
        # modelled without piers is propped down to the terrain, and the terrain
        # can be far below anything the data itself put there.
        lo[2] = min(lo[2], _lowest_terrain(proj, lo, hi) - embed - margin)
    lat = rasterize(verts, faces, pitch, bounds=(lo, hi))
    if lat is None:
        return None
    lat = solidify(lat)
    if proj is not None:
        lat = stand_on(lat, terrain_floor(lat, proj), embed=embed,
                       spacing=spacing, min_feature=min_feature)
    lat = mass(lat, min_feature)
    lat = clip_to(lat, outline)
    # Massing leaves nothing narrower than the nozzle — two voxels — which is
    # what the iso-surface below is allowed to smooth against. Without it the
    # lattice is only guaranteed the one voxel the rasteriser marked.
    return to_mesh(lat, thinnest=2.0 if min_feature > 0 else 1.0)
