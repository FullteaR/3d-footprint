"""GPX parsing: extract track points and compute a bounding box."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from lxml import etree


@dataclass
class Track:
    # Parallel arrays of WGS84 coordinates, in track order.
    lats: list[float]
    lons: list[float]
    # Epoch seconds per point, or None when the file carries no <time> at all
    # (route/waypoint exports often don't). Clipping drops it: a cut lands
    # between two logged points, and nothing downstream needs the timing.
    times: list[float] | None = None

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """(min_lon, min_lat, max_lon, max_lat)."""
        return (min(self.lons), min(self.lats), max(self.lons), max(self.lats))


def _point_time(pt) -> float | None:
    """Epoch seconds of a <time> child, or None if absent/unparsable."""
    el = pt.find("{*}time")
    if el is None or not el.text:
        return None
    try:
        dt = datetime.fromisoformat(el.text.strip())
    except ValueError:
        return None
    if dt.tzinfo is None:  # naive stamps are UTC per the GPX schema
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def parse_gpx(data: bytes) -> Track:
    """Parse trackpoints from GPX bytes (namespace-agnostic).

    Falls back to <rtept>/<wpt> when no <trkpt> is present so that route- or
    waypoint-style files still produce a path.
    """
    try:
        root = etree.fromstring(data)
    except etree.LxmlError as e:
        # Not a ValueError on its own (it derives from SyntaxError), so the
        # route would answer a mistyped upload with a 500 instead of the
        # message the user needs.
        raise ValueError(f"GPX could not be parsed as XML: {e}")

    for tag in ("trkpt", "rtept", "wpt"):
        pts = root.findall(f".//{{*}}{tag}")
        if pts:
            lats = [float(p.get("lat")) for p in pts]
            lons = [float(p.get("lon")) for p in pts]
            return Track(lats=lats, lons=lons, times=_track_times(pts))

    raise ValueError("GPX contains no trkpt/rtept/wpt points")


def _track_times(pts) -> list[float] | None:
    """Per-point epoch seconds, or None when no point is stamped.

    Points missing a <time> inherit the last stamped one (the leading ones the
    first), so a partially stamped file still trims as one continuous track.
    """
    times = [_point_time(p) for p in pts]
    known = [t for t in times if t is not None]
    if not known:
        return None
    last = known[0]
    filled = []
    for t in times:
        if t is not None:
            last = t
        filled.append(last)
    return filled


def parse_time_range_param(value: str) -> tuple[float, float]:
    """Parse a user-supplied "start,end" time range (epoch seconds)."""
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError("time_range must be start,end in epoch seconds")
    try:
        start, end = (float(p) for p in parts)
    except ValueError:
        raise ValueError("time_range values must be numbers")
    if not (math.isfinite(start) and math.isfinite(end)):
        raise ValueError("time_range values must be finite")
    if end <= start:
        raise ValueError("time_range must satisfy start < end")
    return start, end


def trim_track(track: Track, start: float, end: float) -> Track:
    """Keep only the points logged within [start, end] (epoch seconds)."""
    if track.times is None:
        raise ValueError("GPX has no timestamps: time_range cannot be applied")
    keep = [i for i, t in enumerate(track.times) if start <= t <= end]
    if len(keep) < 2:
        raise ValueError("time_range keeps fewer than 2 track points")
    return Track(
        lats=[track.lats[i] for i in keep],
        lons=[track.lons[i] for i in keep],
        times=[track.times[i] for i in keep],
    )


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
