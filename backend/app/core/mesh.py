"""Turn an elevation grid into a watertight, 3D-printable terrain solid (mm).

The `Projection` ties geographic coordinates, the elevation grid and the print
scaling together so that both the terrain mesh and the track overlay live in the
same local millimetre space (and sample the same surface).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter

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

    def lon_of(self, x):
        """Inverse of x_of: print mm -> longitude (for sampling terrain at mm xy)."""
        return self.lon0 + x / (_M_PER_DEG_LON * np.cos(np.radians(self.lat_mid)) * self.scale)

    def lat_of(self, y):
        """Inverse of y_of: print mm -> latitude."""
        return self.lat0 + y / (_M_PER_DEG_LAT * self.scale)

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


def _douglas_peucker(pts: np.ndarray, eps: float) -> np.ndarray:
    """Boolean keep-mask: Ramer-Douglas-Peucker simplification of a polyline."""
    n = len(pts)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, ab = pts[i], pts[j] - pts[i]
        seg = pts[i + 1 : j] - a
        L = float(np.hypot(ab[0], ab[1]))
        if L < 1e-9:
            d = np.hypot(seg[:, 0], seg[:, 1])
        else:
            d = np.abs(seg[:, 0] * ab[1] - seg[:, 1] * ab[0]) / L
        k = int(np.argmax(d))
        if d[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    return keep


def _straighten_boundaries(pos, n0, cP, cQ, chords, eps) -> None:
    """Straighten near-straight colour borders in place (e.g. reclaimed coasts).

    The contour field renders a straight-but-grid-quantised border as gentle
    waves. We trace each border polyline (crossing vertices linked by the
    per-cell chords), Douglas-Peucker it, and slide every dropped crossing back
    onto the straight segment between the kept ones — but only *along its own
    grid edge*, so the mesh topology (and watertightness) is untouched. Genuine
    curves keep enough Douglas-Peucker vertices to stay curved; corners survive.
    """
    from collections import defaultdict

    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in chords:
        adj[a].append(b)
        adj[b].append(a)
    deg = {v: len(ns) for v, ns in adj.items()}

    def trace(start: int, first: int) -> list[int]:
        chain, prev, cur = [start, first], start, first
        while deg.get(cur, 0) == 2:
            a, b = adj[cur]
            nxt = b if a == prev else a
            if nxt == start:           # closed loop: stop
                break
            chain.append(nxt)
            prev, cur = cur, nxt
        return chain

    seen: set[tuple[int, int]] = set()
    chains = []
    for s in [v for v in adj if deg[v] != 2]:      # trace open chains from ends
        for nb in adj[s]:
            if (s, nb) in seen:
                continue
            ch = trace(s, nb)
            for p, q in zip(ch, ch[1:]):
                seen.add((p, q))
                seen.add((q, p))
            chains.append(ch)

    for ch in chains:
        if len(ch) < 3:
            continue
        pts = pos[ch, :2]
        ks = np.nonzero(_douglas_peucker(pts, eps))[0]
        for a, b in zip(ks, ks[1:]):
            if b <= a + 1:
                continue
            A, dAB = pos[ch[a], :2], pos[ch[b], :2] - pos[ch[a], :2]
            for mi in range(a + 1, b):
                v = ch[mi]
                Pn, Qn = pos[cP[v - n0], :2], pos[cQ[v - n0], :2]
                dE = Qn - Pn
                det = dE[0] * dAB[1] - dAB[0] * dE[1]
                if abs(det) < 1e-12:               # border parallel to its edge
                    continue
                rhs = Pn - A
                t = (dAB[0] * rhs[1] - dAB[1] * rhs[0]) / det
                t = min(0.92, max(0.08, t))         # stay on the edge interior
                zP, zQ = pos[cP[v - n0], 2], pos[cQ[v - n0], 2]
                pos[v] = [Pn[0] + t * dE[0], Pn[1] + t * dE[1], zP + t * (zQ - zP)]


def terrain_solid(
    proj: Projection,
    category_grid: np.ndarray | None = None,
    contour_sigma: float = 1.5,
    naturalize: bool = True,
) -> list[Body]:
    """Build the terrain as one watertight solid *per colour*, full-height.

    Each land-use colour becomes its own closed solid running from the terrain
    surface straight down to the flat base, so splitting by colour (stl_multi or
    per-object 3MF) yields independent printable solids whose colour spans the
    whole height — what a multi-material printer needs.

    Crucially, colour boundaries are **not** snapped to grid-cell edges (which
    looks like a staircase of squares). Cells whose four corner categories agree
    are emitted whole; boundary cells are cut along a marching-squares contour
    whose crossing on each grid edge is placed by a smooth (Gaussian) category
    field. The boundary therefore runs diagonally through cells as a smooth
    polyline — natural curves rather than axis-aligned blocks (`contour_sigma`
    sets the field blur). Borders are then Douglas-Peucker straightened, so a
    grid-quantised straight edge (reclaimed coastline, urban parcel) collapses
    back to a clean line while genuine curves stay curved. Per colour we keep
    its top triangles, mirror them to the base, and wall every *boundary* edge
    (an edge used by only one of that colour's triangles); shared interior edges
    get no wall. Returns one `Body` per colour.

    With ``naturalize=False`` (the smoothing slider at 0) the contour cut and
    straightening are skipped entirely: every cell is emitted whole under the
    majority of its four corners, so colours follow raw grid-cell edges (the
    plain axis-aligned "squares" look) — nothing is filtered.
    """
    grid = proj.grid
    ny, nx = grid.elev.shape
    xs = proj.x_of(grid.lons)
    ys = proj.y_of(grid.lats)
    ztop = proj.z_of(proj.filled)                # (ny,nx)
    bottom_z = -proj.base_thickness_mm
    gx, gy = np.meshgrid(xs, ys)

    if category_grid is not None:
        cats = sorted(set(np.asarray(category_grid).ravel().tolist()))
        lut = {c: i for i, c in enumerate(cats)}
        cat = np.asarray(category_grid)
        cat = np.vectorize(lut.__getitem__)(cat).astype(np.int64)
    else:
        cats = ["terrain"]
        cat = np.zeros((ny, nx), np.int64)
    K = len(cats)

    node_top = np.column_stack([gx.ravel(), gy.ravel(), ztop.ravel()])
    top_chunks = [node_top]
    base = ny * nx
    hidx = np.full((ny, nx - 1), -1, np.int64)   # crossing vtx on horizontal edges
    vidx = np.full((ny - 1, nx), -1, np.int64)   # crossing vtx on vertical edges
    cross_P: list[np.ndarray] = []               # grid-edge endpoints per crossing
    cross_Q: list[np.ndarray] = []               # (for the straightening pass)

    tris: list[list[np.ndarray]] = [[] for _ in range(K)]
    centers: list[list[float]] = []
    chords: list[tuple[int, int]] = []           # crossing pairs = border segments
    c00, c01 = cat[:-1, :-1], cat[:-1, 1:]
    c10, c11 = cat[1:, :-1], cat[1:, 1:]

    if K > 1 and naturalize:
        # Smooth per-category field; the A/B contour crossing on an edge sits
        # where the two categories' fields are equal (a sub-cell position).
        field = np.stack([
            gaussian_filter((cat == k).astype(np.float32), contour_sigma, mode="nearest")
            for k in range(K)
        ])

        def crossing_t(A, B, SP, SQ):
            """Interpolation factor (P->Q) where field[A]==field[B] on an edge."""
            gP = np.take_along_axis(SP, A[None], 0)[0] - np.take_along_axis(SP, B[None], 0)[0]
            gQ = np.take_along_axis(SQ, A[None], 0)[0] - np.take_along_axis(SQ, B[None], 0)[0]
            den = gP - gQ
            t = np.where(np.abs(den) > 1e-9, gP / np.where(den == 0, 1.0, den), 0.5)
            return np.clip(t, 0.08, 0.92)

        A, B = cat[:, :-1], cat[:, 1:]           # horizontal edges
        rr, cc = np.nonzero(A != B)
        if rr.size:
            t = crossing_t(A, B, field[:, :, :-1], field[:, :, 1:])[rr, cc]
            hidx[rr, cc] = base + np.arange(rr.size)
            base += rr.size
            cross_P.append(rr * nx + cc)
            cross_Q.append(rr * nx + cc + 1)
            top_chunks.append(np.column_stack([
                xs[cc] * (1 - t) + xs[cc + 1] * t, ys[rr],
                ztop[rr, cc] * (1 - t) + ztop[rr, cc + 1] * t]))

        A, B = cat[:-1, :], cat[1:, :]           # vertical edges
        rr, cc = np.nonzero(A != B)
        if rr.size:
            t = crossing_t(A, B, field[:, :-1, :], field[:, 1:, :])[rr, cc]
            vidx[rr, cc] = base + np.arange(rr.size)
            base += rr.size
            cross_P.append(rr * nx + cc)
            cross_Q.append((rr + 1) * nx + cc)
            top_chunks.append(np.column_stack([
                xs[cc], ys[rr] * (1 - t) + ys[rr + 1] * t,
                ztop[rr, cc] * (1 - t) + ztop[rr + 1, cc] * t]))

        # Uniform cells (all four corners agree): emit two triangles in bulk.
        uni = (c00 == c01) & (c00 == c10) & (c00 == c11)
        ur, uc = np.nonzero(uni)
        if ur.size:
            TL = ur * nx + uc
            # CCW (normal +z): (TL,TR,BR) and (TL,BR,BL), matching the CCW cell
            # perimeter used for boundary cells, so winding is globally consistent.
            t1 = np.column_stack([TL, TL + 1, TL + nx + 1])
            t2 = np.column_stack([TL, TL + nx + 1, TL + nx])
            col = c00[ur, uc]
            for k in range(K):
                mk = col == k
                if mk.any():
                    tris[k].extend((t1[mk], t2[mk]))

        # Boundary cells: cut along the contour.
        for r, c in zip(*(np.nonzero(~uni))):
            r, c = int(r), int(c)
            TL, TR = r * nx + c, r * nx + c + 1
            BR, BL = (r + 1) * nx + c + 1, (r + 1) * nx + c
            perim: list[tuple[int, int | None]] = [(TL, int(cat[r, c]))]
            if hidx[r, c] >= 0: perim.append((int(hidx[r, c]), None))
            perim.append((TR, int(cat[r, c + 1])))
            if vidx[r, c + 1] >= 0: perim.append((int(vidx[r, c + 1]), None))
            perim.append((BR, int(cat[r + 1, c + 1])))
            if hidx[r + 1, c] >= 0: perim.append((int(hidx[r + 1, c]), None))
            perim.append((BL, int(cat[r + 1, c])))
            if vidx[r, c] >= 0: perim.append((int(vidx[r, c]), None))

            cross = [i for i, (_, k) in enumerate(perim) if k is None]
            distinct = {cat[r, c], cat[r, c + 1], cat[r + 1, c + 1], cat[r + 1, c]}
            if len(cross) == 2 and len(distinct) == 2:
                i0, i1 = cross
                chords.append((perim[i0][0], perim[i1][0]))
                for arc in (perim[i0:i1 + 1], perim[i1:] + perim[:i0 + 1]):
                    k = next(kk for _, kk in arc if kk is not None)
                    poly = [v for v, _ in arc]
                    for a in range(1, len(poly) - 1):
                        tris[k].append(np.array([[poly[0], poly[a], poly[a + 1]]], np.int64))
            else:  # saddle / 3-4 categories: fan from the cell centre
                ci = base + len(centers)
                centers.append([
                    float(gx[r, c:c + 2].mean()), float(gy[r:r + 2, c].mean()),
                    float(ztop[r:r + 2, c:c + 2].mean())])
                m = len(perim)
                for a in range(m):
                    v0, k0 = perim[a]
                    v1, k1 = perim[(a + 1) % m]
                    k = k0 if k0 is not None else k1
                    tris[k].append(np.array([[ci, v0, v1]], np.int64))
    else:
        # Raw mode (smoothness 0, or single colour): no contour cut. Each cell is
        # emitted whole under the majority of its four corners, so colour borders
        # follow grid-cell edges — the plain axis-aligned "squares".
        if K > 1:
            counts = np.stack([
                (c00 == k).astype(np.int64) + (c01 == k) + (c10 == k) + (c11 == k)
                for k in range(K)
            ])
            cell_cat = np.argmax(counts, axis=0)
        else:
            cell_cat = np.zeros((ny - 1, nx - 1), np.int64)
        for k in range(K):
            rr, cc = np.nonzero(cell_cat == k)
            if rr.size:
                TL = rr * nx + cc
                tris[k].append(np.column_stack([TL, TL + 1, TL + nx + 1]))
                tris[k].append(np.column_stack([TL, TL + nx + 1, TL + nx]))

    top_all = np.vstack(top_chunks + ([np.array(centers)] if centers else []))
    n = len(top_all)

    # Straighten near-straight colour borders (reclaimed coasts, urban parcels)
    # by sliding crossings along their edges — topology, and thus the per-colour
    # watertightness, is preserved.
    if chords:
        cell_mm = 0.5 * (float(np.mean(np.diff(xs))) + float(np.mean(np.diff(ys))))
        _straighten_boundaries(
            top_all, ny * nx,
            np.concatenate(cross_P), np.concatenate(cross_Q),
            chords, eps=0.75 * cell_mm,
        )

    vertices = np.vstack([top_all, top_all * [1, 1, 0] + [0, 0, bottom_z]])

    bodies: list[Body] = []
    for k in range(K):
        if not tris[k]:
            continue
        F = np.vstack(tris[k])
        # Directed half-edges; a boundary edge is one whose reverse is absent
        # (interior edges appear as i->j and j->i and cancel). Top is CCW so the
        # boundary circulates with the interior on its left; the wall winding
        # below then faces outward — no costly fix_normals/fix_winding pass.
        di = np.concatenate([F[:, 0], F[:, 1], F[:, 2]]).astype(np.int64)
        dj = np.concatenate([F[:, 1], F[:, 2], F[:, 0]]).astype(np.int64)
        keys = di * n + dj
        bnd = ~np.isin(dj * n + di, keys)
        bi, bj = di[bnd], dj[bnd]
        faces = np.vstack([
            F,                                    # top surface (normal +z)
            F[:, ::-1] + n,                       # mirrored base (normal -z)
            np.column_stack([bi, bi + n, bj + n]),    # walls (outward)
            np.column_stack([bi, bj + n, bj]),
        ])
        used = np.unique(faces)
        remap = np.empty(len(vertices), np.int64)
        remap[used] = np.arange(used.size)
        mesh = trimesh.Trimesh(vertices=vertices[used], faces=remap[faces], process=False)
        bodies.append(Body(mesh, cats[k]))
    return bodies
