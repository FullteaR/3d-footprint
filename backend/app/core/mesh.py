"""Turn an elevation grid into a watertight, 3D-printable terrain solid (mm).

The `Projection` ties geographic coordinates, the elevation grid and the print
scaling together so that both the terrain mesh and the track overlay live in the
same local millimetre space (and sample the same surface).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

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
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Build a closed terrain mesh and per-face labels.

    Top faces are labelled by land-use category (when `category_grid` is given,
    a (ny, nx) array of category names); the bottom and walls are labelled
    "base". Returns (mesh, face_labels) with face order preserved.
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

    def tidx(r, c):
        return r * nx + c

    def bidx(r, c):
        return n_top + r * nx + c

    r = np.arange(ny - 1)[:, None]
    c = np.arange(nx - 1)[None, :]
    v00 = (r * nx + c).ravel()
    v01 = (r * nx + c + 1).ravel()
    v10 = ((r + 1) * nx + c).ravel()
    v11 = ((r + 1) * nx + c + 1).ravel()
    top_faces = np.concatenate(
        [np.column_stack([v00, v10, v11]), np.column_stack([v00, v11, v01])]
    )
    n_cells = (ny - 1) * (nx - 1)
    bot_faces = top_faces[:, ::-1] + n_top

    def wall(seq):
        out = []
        for (r0, c0), (r1, c1) in zip(seq[:-1], seq[1:]):
            ta, tb = tidx(r0, c0), tidx(r1, c1)
            ba, bb = bidx(r0, c0), bidx(r1, c1)
            out.append([ta, tb, bb])
            out.append([ta, bb, ba])
        return out

    border = (
        [(0, cc) for cc in range(nx)]
        + [(rr, nx - 1) for rr in range(1, ny)]
        + [(ny - 1, cc) for cc in range(nx - 2, -1, -1)]
        + [(rr, 0) for rr in range(ny - 2, -1, -1)]
    )
    wall_faces = np.array(wall(border), dtype=np.int64)
    faces = np.vstack([top_faces, bot_faces, wall_faces])

    # Per-face labels (same order as `faces`): top by category, rest "base".
    if category_grid is not None:
        cell_cat = category_grid[:-1, :-1].ravel()  # one category per grid cell
    else:
        cell_cat = np.full(n_cells, "terrain", dtype="<U8")
    top_labels = np.concatenate([cell_cat, cell_cat])  # two triangles per cell
    base_labels = np.full(len(bot_faces) + len(wall_faces), "base", dtype="<U8")
    labels = np.concatenate([top_labels, base_labels])

    # process=False keeps face order (vertices are already shared by index).
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.fix_normals()
    return mesh, labels
