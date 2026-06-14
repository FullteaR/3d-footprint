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
