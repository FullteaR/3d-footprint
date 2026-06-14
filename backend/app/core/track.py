"""Build the GPX track as a raised convex line that hugs the terrain surface.

The track is its own watertight body (kept separate from the terrain, which is
what multi-colour printing wants). Its cross-section is a world-up rectangle so
the ridge stays vertical; the bottom is embedded slightly below the sampled
terrain so the ridge merges with the ground (no floating gap).
"""
from __future__ import annotations

import numpy as np
import trimesh

from .gpx import Track
from .mesh import Projection


def track_ridge(
    track: Track,
    proj: Projection,
    width_mm: float = 1.2,
    height_mm: float = 1.5,
    embed_mm: float = 0.6,
) -> trimesh.Trimesh:
    """Sweep a rectangular cross-section along the track at terrain height."""
    lon = np.asarray(track.lons, dtype=np.float64)
    lat = np.asarray(track.lats, dtype=np.float64)
    x = proj.x_of(lon)
    y = proj.y_of(lat)
    zt_surf = proj.sample_z(lon, lat)  # terrain z (mm) under each point

    # Drop consecutive (near-)duplicate points so tangents are well defined.
    p = np.column_stack([x, y])
    keep = np.concatenate([[True], np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-4])
    p = p[keep]
    zt_surf = zt_surf[keep]
    if len(p) < 2:
        raise ValueError("track has too few distinct points")

    # XY tangents (central differences) and left normals.
    t = np.gradient(p, axis=0)
    tn = np.linalg.norm(t, axis=1, keepdims=True)
    t = np.divide(t, tn, out=np.zeros_like(t), where=tn > 0)
    normal = np.column_stack([-t[:, 1], t[:, 0]])  # left-hand perpendicular

    w2 = width_mm / 2.0
    z_top = zt_surf + height_mm
    z_bot = zt_surf - embed_mm

    left = p + normal * w2
    right = p - normal * w2

    m = len(p)
    # 4 vertices per station: TL, TR, BR, BL.
    verts = np.empty((m * 4, 3))
    verts[0::4] = np.column_stack([left, z_top])    # TL
    verts[1::4] = np.column_stack([right, z_top])   # TR
    verts[2::4] = np.column_stack([right, z_bot])   # BR
    verts[3::4] = np.column_stack([left, z_bot])    # BL

    faces: list = []
    i = np.arange(m - 1)
    TLi, TRi, BRi, BLi = 4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3
    TLj, TRj, BRj, BLj = 4 * (i + 1), 4 * (i + 1) + 1, 4 * (i + 1) + 2, 4 * (i + 1) + 3

    def strip(a, b, c, d):  # quad (a,b on station i; c,d on station i+1) -> 2 tris
        return np.concatenate(
            [np.column_stack([a, b, c]), np.column_stack([a, c, d])]
        )

    faces.append(strip(TLi, TRi, TRj, TLj))  # top
    faces.append(strip(TRi, BRi, BRj, TRj))  # right side
    faces.append(strip(BRi, BLi, BLj, BRj))  # bottom
    faces.append(strip(BLi, TLi, TLj, BLj))  # left side

    # End caps.
    faces.append(np.array([[0, 1, 2], [0, 2, 3]]))                         # start
    last = 4 * (m - 1)
    faces.append(np.array([[last, last + 2, last + 1], [last, last + 3, last + 2]]))

    mesh = trimesh.Trimesh(vertices=verts, faces=np.vstack(faces), process=True)
    mesh.fix_normals()
    return mesh
