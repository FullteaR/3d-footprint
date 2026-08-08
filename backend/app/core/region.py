"""Model region: the print outline (rect / square / regular hexagon) on the map.

A region is an axis-aligned bbox (the *unrotated* shape extents), a shape kind
and a rotation about the bbox centre. Shape and rotation live in the local
equirectangular metre frame centred on the bbox — the same constants the print
projection uses — so a "square" or "regular hexagon" is regular in printed
millimetres, not in degrees.

`polygon_lonlat` is the rotated outline back in lon/lat: the clip region for
the terrain cells, the track, buildings and bridges. The DEM is fetched over
its axis-aligned bounds, and once every body is built `to_print_frame` rotates
the scene back so the outline prints axis-aligned starting at the origin.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Polygon

from .gpx import Track
from .mesh import _M_PER_DEG_LAT, _M_PER_DEG_LON, Projection

SHAPES = ("rect", "square", "hex")


def parse_shape_param(value: str) -> str:
    if value not in SHAPES:
        raise ValueError(f"shape must be one of: {', '.join(SHAPES)}")
    return value


def parse_rotation_param(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("rotation_deg must be a finite number")
    return float(value) % 360.0


def parse_lonlat_param(value: str) -> tuple[float, float]:
    """Parse a user-supplied "lon,lat" point."""
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError("point must be lon,lat")
    try:
        lon, lat = (float(p) for p in parts)
    except ValueError:
        raise ValueError("point values must be numbers")
    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        raise ValueError("point is out of range")
    return lon, lat


@dataclass(frozen=True)
class Region:
    bbox: tuple[float, float, float, float]  # unrotated extents (w, s, e, n)
    shape: str = "rect"
    rotation_deg: float = 0.0                # CCW in the local metre frame

    @property
    def is_plain(self) -> bool:
        """True when the region is exactly its bbox (the legacy fast path)."""
        return self.shape == "rect" and self.rotation_deg % 360.0 == 0.0

    @property
    def center(self) -> tuple[float, float]:
        w, s, e, n = self.bbox
        return ((w + e) / 2.0, (s + n) / 2.0)

    def _m_per_deg(self) -> tuple[float, float]:
        _, clat = self.center
        return _M_PER_DEG_LON * math.cos(math.radians(clat)), _M_PER_DEG_LAT

    @property
    def half_extents_m(self) -> tuple[float, float]:
        """Half width/height (metres) of the shape, inscribed in the bbox.

        The frontend keeps the bbox at the shape's exact aspect; inscribing here
        (min) just guarantees a true square / regular hexagon for any caller.
        """
        w, s, e, n = self.bbox
        m_lon, m_lat = self._m_per_deg()
        hw = (e - w) / 2.0 * m_lon
        hh = (n - s) / 2.0 * m_lat
        if self.shape == "square":
            a = min(hw, hh)
            return a, a
        if self.shape == "hex":
            r = min(hw, 2.0 * hh / math.sqrt(3.0))
            return r, math.sqrt(3.0) / 2.0 * r
        return hw, hh

    @property
    def span_m(self) -> float:
        """Longest edge of the shape (metres) — what size_mm applies to."""
        hw, hh = self.half_extents_m
        return 2.0 * max(hw, hh)

    def _outline_local(self) -> np.ndarray:
        """Unrotated outline vertices (metres about the centre), CCW."""
        hw, hh = self.half_extents_m
        if self.shape == "hex":  # flat-top regular hexagon, circumradius hw
            ang = np.radians(np.arange(6) * 60.0)
            return np.column_stack([hw * np.cos(ang), hw * np.sin(ang)])
        return np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])

    def polygon_lonlat(self) -> Polygon:
        """The rotated outline in lon/lat (the clip region)."""
        pts = self._outline_local()
        th = math.radians(self.rotation_deg)
        c, s = math.cos(th), math.sin(th)
        x = pts[:, 0] * c - pts[:, 1] * s
        y = pts[:, 0] * s + pts[:, 1] * c
        clon, clat = self.center
        m_lon, m_lat = self._m_per_deg()
        return Polygon(np.column_stack([clon + x / m_lon, clat + y / m_lat]))

    def polygon_mm(self, proj: Projection) -> Polygon:
        """The rotated outline in print millimetres (for terrain clipping)."""
        ext = np.asarray(self.polygon_lonlat().exterior.coords)
        return Polygon(np.column_stack([proj.x_of(ext[:, 0]), proj.y_of(ext[:, 1])]))

    def fetch_bbox(self) -> tuple[float, float, float, float]:
        """Axis-aligned lon/lat bounds of the rotated outline (the DEM extent)."""
        return self.polygon_lonlat().bounds

    def _print_frame(self, proj: Projection):
        """(centre, offset, cos, sin) of the map -> print frame transform."""
        clon, clat = self.center
        th = math.radians(self.rotation_deg)
        hw, hh = self.half_extents_m
        return (
            (float(proj.x_of(clon)), float(proj.y_of(clat))),
            (hw * proj.scale, hh * proj.scale),
            math.cos(th), math.sin(th),
        )

    def print_xy(self, x, y, proj: Projection):
        """Map-frame millimetres -> the axis-aligned print frame."""
        if self.is_plain:
            return x, y
        (cx, cy), (ox, oy), c, s = self._print_frame(proj)
        xr, yr = x - cx, y - cy
        return xr * c + yr * s + ox, -xr * s + yr * c + oy

    def map_xy(self, x, y, proj: Projection):
        """Inverse of `print_xy`: print frame -> map-frame millimetres.

        What the terrain has to be sampled through — the elevation grid still
        lives in the unrotated DEM frame after the scene has been rotated.
        """
        if self.is_plain:
            return x, y
        (cx, cy), (ox, oy), c, s = self._print_frame(proj)
        xr, yr = x - ox, y - oy
        return xr * c - yr * s + cx, xr * s + yr * c + cy

    def outline_print_mm(self, proj: Projection) -> Polygon:
        """The print outline in the print frame (mm) — where the model ends."""
        hw, hh = self.half_extents_m
        pts = self._outline_local() * proj.scale
        return Polygon(pts + np.array([hw * proj.scale, hh * proj.scale]))

    def to_print_frame(self, bodies, proj: Projection) -> None:
        """Rotate finished bodies so the outline prints axis-aligned at origin.

        The whole scene was built in the DEM's north-up frame; undoing the
        region rotation about the shape centre aligns the cut walls with the
        print axes, and the translation puts the shape's bounding box at
        [0, 2*hw] x [0, 2*hh] mm like the plain-rect path.
        """
        if self.is_plain:
            return
        for b in bodies:
            v = np.asarray(b.mesh.vertices, dtype=np.float64).copy()
            v[:, 0], v[:, 1] = self.print_xy(v[:, 0], v[:, 1], proj)
            b.mesh.vertices = v


def clip_track_to_polygon(track: Track, poly) -> list[Track]:
    """Polygon counterpart of gpx.clip_track: in-outline pieces as sub-tracks.

    Cuts land exactly on the outline, so a clipped ridge ends at the model edge
    instead of jumping straight across the part it left out.
    """
    if len(track.lons) < 2:
        return []
    inter = LineString(zip(track.lons, track.lats)).intersection(poly)
    out: list[Track] = []
    for g in getattr(inter, "geoms", [inter]):
        if isinstance(g, LineString) and len(g.coords) >= 2:
            lons, lats = zip(*((p[0], p[1]) for p in g.coords))
            out.append(Track(lats=list(lats), lons=list(lons)))
    return out
