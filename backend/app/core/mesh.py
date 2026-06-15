"""Turn an elevation grid into a watertight, 3D-printable terrain solid (mm).

The `Projection` ties geographic coordinates, the elevation grid and the print
scaling together so that both the terrain mesh and the track overlay live in the
same local millimetre space (and sample the same surface).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from .export import Body
from .terrain import ElevationGrid

# Equirectangular metres-per-degree about the scene centre.
_M_PER_DEG_LAT = 110540.0
_M_PER_DEG_LON = 111320.0


@dataclass
class MeshParams:
    size_mm: float = 120.0          # longest horizontal edge of the print
    vertical_scale: float = 8.0     # vertical exaggeration applied to relief
    base_thickness_mm: float = 3.0  # solid base below the lowest terrain point


@dataclass
class Projection:
    """Maps (lon, lat, elevation) -> local print millimetres, and samples z."""

    lon0: float
    lat0: float
    lat_mid: float
    scale: float            # mm per metre (horizontal)
    emin: float             # min valid elevation (metres), the z=0 datum
    vertical_scale: float
    base_thickness_mm: float
    grid: ElevationGrid
    filled: np.ndarray      # elevation with no-data filled to emin (ny, nx)

    def x_of(self, lon):
        return (lon - self.lon0) * _M_PER_DEG_LON * np.cos(np.radians(self.lat_mid)) * self.scale

    def y_of(self, lat):
        return (lat - self.lat0) * _M_PER_DEG_LAT * self.scale

    def z_of(self, elev):
        return (elev - self.emin) * self.scale * self.vertical_scale

    def sample_z(self, lon, lat):
        """Bilinearly sample terrain z (mm) at arbitrary lon/lat."""
        col = np.interp(lon, self.grid.lons, np.arange(self.grid.lons.size))
        row = np.interp(lat, self.grid.lats, np.arange(self.grid.lats.size))
        c0 = np.clip(np.floor(col).astype(int), 0, self.grid.lons.size - 2)
        r0 = np.clip(np.floor(row).astype(int), 0, self.grid.lats.size - 2)
        fc, fr = col - c0, row - r0
        f = self.filled
        top = f[r0, c0] * (1 - fc) + f[r0, c0 + 1] * fc
        bot = f[r0 + 1, c0] * (1 - fc) + f[r0 + 1, c0 + 1] * fc
        elev = top * (1 - fr) + bot * fr
        return self.z_of(elev)


def make_projection(grid: ElevationGrid, params: MeshParams) -> Projection:
    elev = grid.elev.astype(np.float64)
    valid = np.isfinite(elev)
    if not valid.any():
        raise ValueError("no valid elevation data in this area")
    emin = float(elev[valid].min())
    filled = np.where(valid, elev, emin)

    lat_mid = float(np.mean(grid.lats))
    width = (grid.lons.max() - grid.lons.min()) * _M_PER_DEG_LON * np.cos(np.radians(lat_mid))
    height = (grid.lats.max() - grid.lats.min()) * _M_PER_DEG_LAT
    span = max(float(width), float(height), 1e-6)
    scale = params.size_mm / span

    return Projection(
        lon0=float(grid.lons.min()),
        lat0=float(grid.lats.min()),
        lat_mid=lat_mid,
        scale=scale,
        emin=emin,
        vertical_scale=params.vertical_scale,
        base_thickness_mm=params.base_thickness_mm,
        grid=grid,
        filled=filled,
    )


def terrain_solid(
    proj: Projection, category_grid: np.ndarray | None = None
) -> list[Body]:
    """Build the terrain as one watertight solid *per colour*, full-height.

    Rather than a single base slab carrying a thin coloured skin on top, each
    land-use colour becomes its own closed solid running from the terrain
    surface straight down to the flat base. Splitting by colour (stl_multi, or
    per-object 3MF) then yields independent, printable solids whose colour spans
    the whole height — which is what a multi-material printer needs.

    A colour's solid is the boundary of its columns: the top surface cells, the
    flat bottom cells, and vertical walls only on edges where the neighbour is a
    *different* colour or off-grid (shared edges between same-colour cells are
    interior, so no wall). Returns one `Body` per colour.
    """
    grid = proj.grid
    ny, nx = grid.elev.shape
    xs = proj.x_of(grid.lons)                    # (nx,)
    ys = proj.y_of(grid.lats)                    # (ny,)
    z = proj.z_of(proj.filled)                   # (ny,nx), min 0 at lowest point
    bottom_z = -proj.base_thickness_mm

    gx, gy = np.meshgrid(xs, ys)                 # (ny,nx)
    top = np.column_stack([gx.ravel(), gy.ravel(), z.ravel()])
    bot = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, bottom_z)])
    vertices = np.vstack([top, bot])
    n_top = ny * nx

    if category_grid is not None:
        cell_cat = np.asarray(category_grid)[:-1, :-1]   # one category per cell
    else:
        cell_cat = np.full((ny - 1, nx - 1), "terrain", dtype="<U8")

    bodies: list[Body] = []
    for color in np.unique(cell_cat):
        m = cell_cat == color                    # (ny-1, nx-1) cells of this colour
        rr, cc = np.nonzero(m)
        if rr.size == 0:
            continue
        tl = rr * nx + cc                        # cell corner -> top vertex index
        tr = rr * nx + cc + 1
        bl = (rr + 1) * nx + cc
        br = (rr + 1) * nx + cc + 1
        blocks = [
            np.column_stack([tl, bl, br]),       # top surface (2 tris/cell)
            np.column_stack([tl, br, tr]),
            np.column_stack([tl + n_top, br + n_top, bl + n_top]),   # flat bottom
            np.column_stack([tl + n_top, tr + n_top, br + n_top]),
        ]

        # Vertical walls on this colour's boundary edges only. Winding is fixed
        # afterwards by fix_normals (the per-colour solid is closed).
        def wall(sel: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
            ra, ca = np.nonzero(sel)
            if ra.size == 0:
                return
            ta = (ra + a[0]) * nx + (ca + a[1])
            tb = (ra + b[0]) * nx + (ca + b[1])
            blocks.append(np.column_stack([ta, tb, tb + n_top]))
            blocks.append(np.column_stack([ta, tb + n_top, ta + n_top]))

        east = m.copy();  east[:, :-1] &= ~m[:, 1:]    # right edge  (TR..BR)
        west = m.copy();  west[:, 1:] &= ~m[:, :-1]    # left edge   (BL..TL)
        north = m.copy(); north[1:, :] &= ~m[:-1, :]   # upper edge  (TL..TR)
        south = m.copy(); south[:-1, :] &= ~m[1:, :]   # lower edge  (BR..BL)
        wall(east,  (0, 1), (1, 1))
        wall(west,  (1, 0), (0, 0))
        wall(north, (0, 0), (0, 1))
        wall(south, (1, 1), (1, 0))

        faces = np.vstack(blocks)
        used = np.unique(faces)                  # compact to referenced vertices
        remap = np.empty(len(vertices), np.int64)
        remap[used] = np.arange(used.size)
        mesh = trimesh.Trimesh(
            vertices=vertices[used], faces=remap[faces], process=False
        )
        mesh.fix_normals()
        bodies.append(Body(mesh, str(color)))
    return bodies
