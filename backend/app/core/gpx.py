"""GPX parsing: extract track points and compute a bounding box."""
from __future__ import annotations

from dataclasses import dataclass

from lxml import etree


@dataclass
class Track:
    # Parallel arrays of WGS84 coordinates, in track order.
    lats: list[float]
    lons: list[float]

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """(min_lon, min_lat, max_lon, max_lat)."""
        return (min(self.lons), min(self.lats), max(self.lons), max(self.lats))


def parse_gpx(data: bytes) -> Track:
    """Parse trackpoints from GPX bytes (namespace-agnostic).

    Falls back to <rtept>/<wpt> when no <trkpt> is present so that route- or
    waypoint-style files still produce a path.
    """
    root = etree.fromstring(data)

    for tag in ("trkpt", "rtept", "wpt"):
        pts = root.findall(f".//{{*}}{tag}")
        if pts:
            lats = [float(p.get("lat")) for p in pts]
            lons = [float(p.get("lon")) for p in pts]
            return Track(lats=lats, lons=lons)

    raise ValueError("GPX contains no trkpt/rtept/wpt points")


def expand_bbox(
    bbox: tuple[float, float, float, float], margin: float = 0.08
) -> tuple[float, float, float, float]:
    """Pad a bbox by `margin` fraction of its span (min span guard included)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    dlon = max(max_lon - min_lon, 1e-3)
    dlat = max(max_lat - min_lat, 1e-3)
    return (
        min_lon - dlon * margin,
        min_lat - dlat * margin,
        max_lon + dlon * margin,
        max_lat + dlat * margin,
    )


def parse_bbox_param(value: str) -> tuple[float, float, float, float]:
    """Parse a user-supplied "min_lon,min_lat,max_lon,max_lat" bbox string."""
    parts = value.split(",")
    if len(parts) != 4:
        raise ValueError("bbox must be min_lon,min_lat,max_lon,max_lat")
    try:
        min_lon, min_lat, max_lon, max_lat = (float(p) for p in parts)
    except ValueError:
        raise ValueError("bbox values must be numbers")
    if not (-180.0 <= min_lon < max_lon <= 180.0):
        raise ValueError("bbox longitudes must satisfy -180 <= min < max <= 180")
    if not (-90.0 <= min_lat < max_lat <= 90.0):
        raise ValueError("bbox latitudes must satisfy -90 <= min < max <= 90")
    if max_lon - min_lon < 1e-3 or max_lat - min_lat < 1e-3:
        raise ValueError("bbox too small: each side must span at least 0.001 deg")
    return (min_lon, min_lat, max_lon, max_lat)


def clip_track(
    track: Track, bbox: tuple[float, float, float, float]
) -> list[Track]:
    """Clip the track polyline to bbox (Liang-Barsky per segment).

    Returns the in-bbox pieces as separate sub-tracks; each cut lands exactly
    on the border, so a clipped ridge ends at the model edge instead of
    jumping straight across the part it left out.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    out: list[Track] = []
    cur_lons: list[float] = []
    cur_lats: list[float] = []

    def flush() -> None:
        nonlocal cur_lons, cur_lats
        if len(cur_lons) >= 2:
            out.append(Track(lats=cur_lats, lons=cur_lons))
        cur_lons, cur_lats = [], []

    for i in range(len(track.lons) - 1):
        x0, y0 = track.lons[i], track.lats[i]
        dx = track.lons[i + 1] - x0
        dy = track.lats[i + 1] - y0
        t0, t1 = 0.0, 1.0
        inside = True
        for p, q in (
            (-dx, x0 - min_lon), (dx, max_lon - x0),
            (-dy, y0 - min_lat), (dy, max_lat - y0),
        ):
            if p == 0.0:
                if q < 0.0:
                    inside = False
                    break
            else:
                r = q / p
                if p < 0.0:
                    t0 = max(t0, r)
                else:
                    t1 = min(t1, r)
        if not inside or t0 > t1:
            flush()
            continue
        if t0 > 0.0:  # (re-)entering: the previous piece ended outside
            flush()
        if not cur_lons:
            cur_lons.append(x0 + t0 * dx)
            cur_lats.append(y0 + t0 * dy)
        cur_lons.append(x0 + t1 * dx)
        cur_lats.append(y0 + t1 * dy)
        if t1 < 1.0:  # leaving: cut at the border
            flush()
    flush()
    return out
