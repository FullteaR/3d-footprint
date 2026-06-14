"""Turn an elevation grid into a watertight, 3D-printable terrain solid (mm)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from .terrain import ElevationGrid


@dataclass
class MeshParams:
    size_mm: float = 120.0          # longest horizontal edge of the print
    vertical_scale: float = 3.0     # vertical exaggeration applied to relief
    base_thickness_mm: float = 3.0  # solid base below the lowest terrain point


def _local_xy_meters(grid: ElevationGrid) -> tuple[np.ndarray, np.ndarray]:
    """Convert lon/lat to local planar meters (equirectangular about the center).

    Corrects E-W vs N-S distortion so the terrain is not stretched along longitude.
    """
    lat_mid = float(np.mean(grid.lats))
    x_m = (grid.lons - grid.lons.min()) * 111320.0 * np.cos(np.radians(lat_mid))
    y_m = (grid.lats - grid.lats.min()) * 110540.0
    return x_m, y_m


def terrain_solid(grid: ElevationGrid, params: MeshParams) -> trimesh.Trimesh:
    """Build a closed terrain mesh: draped top surface + flat bottom + side walls."""
    elev = grid.elev.astype(np.float64)
    ny, nx = elev.shape

    # Fill no-data with the minimum valid elevation (sea -> flat).
    valid = np.isfinite(elev)
    if not valid.any():
        raise ValueError("no valid elevation data in this area")
    emin = float(elev[valid].min())
    elev = np.where(valid, elev, emin)

    x_m, y_m = _local_xy_meters(grid)
    width = float(x_m.max() - x_m.min())
    height = float(y_m.max() - y_m.min())
    span = max(width, height, 1e-6)
    scale = params.size_mm / span  # mm per meter (horizontal)

    xs = x_m * scale                                    # (nx,)
    ys = y_m * scale                                    # (ny,)
    z = (elev - emin) * scale * params.vertical_scale   # (ny,nx), min 0 at lowest point
    bottom_z = -params.base_thickness_mm

    # --- vertices: top grid then bottom grid (same x,y, flat z) ---
    gx, gy = np.meshgrid(xs, ys)                        # (ny,nx)
    top = np.column_stack([gx.ravel(), gy.ravel(), z.ravel()])
    bot = np.column_stack(
        [gx.ravel(), gy.ravel(), np.full(gx.size, bottom_z)]
    )
    vertices = np.vstack([top, bot])
    n_top = ny * nx

    def tidx(r, c):  # top vertex index
        return r * nx + c

    def bidx(r, c):  # bottom vertex index
        return n_top + r * nx + c

    faces: list = []

    # Top surface (two triangles per cell, CCW seen from above).
    r = np.arange(ny - 1)[:, None]
    c = np.arange(nx - 1)[None, :]
    v00 = (r * nx + c).ravel()
    v01 = (r * nx + c + 1).ravel()
    v10 = ((r + 1) * nx + c).ravel()
    v11 = ((r + 1) * nx + c + 1).ravel()
    top_faces = np.concatenate(
        [np.column_stack([v00, v10, v11]), np.column_stack([v00, v11, v01])]
    )
    # Bottom surface (mirror, reversed winding, offset to bottom vertices).
    bot_faces = top_faces[:, ::-1] + n_top
    faces.append(top_faces)
    faces.append(bot_faces)

    # Side walls along the four borders (top perimeter -> bottom perimeter).
    def wall(seq):
        out = []
        for (r0, c0), (r1, c1) in zip(seq[:-1], seq[1:]):
            ta, tb = tidx(r0, c0), tidx(r1, c1)
            ba, bb = bidx(r0, c0), bidx(r1, c1)
            out.append([ta, tb, bb])
            out.append([ta, bb, ba])
        return out

    border = (
        [(0, c) for c in range(nx)]                       # south edge
        + [(r, nx - 1) for r in range(1, ny)]             # east edge
        + [(ny - 1, c) for c in range(nx - 2, -1, -1)]    # north edge
        + [(r, 0) for r in range(ny - 2, -1, -1)]         # west edge
    )
    faces.append(np.array(wall(border), dtype=np.int64))

    mesh = trimesh.Trimesh(
        vertices=vertices, faces=np.vstack(faces), process=True
    )
    mesh.fix_normals()  # make winding consistent + normals outward
    return mesh
