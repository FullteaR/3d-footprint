"""PLATEAU tran (道路) -> the grooves that divide the city into blocks.

Below about 1:40,000 a road cannot print as itself. The building massing
(`buildings.py`) keeps whatever gap between blocks is at least one nozzle
wide, which is the honest answer — but in ground metres that threshold is
`min_feature_mm x 縮尺分母 / 1000`, so at 1:150,000 it asks for a 120 m road
and a whole low-rise ward comes out as one slab. 環七 is 0.2 m of a
millimetre there; no arrangement of true-to-scale geometry will show it.

So the roads worth keeping are drawn deliberately instead, at a printable
width rather than their own — the same bargain `height_scale` makes for
height. Two steps, and the first is exactly the closing in `massing.py` run
backwards:

  * **erode** the road surface by half the width threshold. Anything narrower
    vanishes (路地, 区画道路, driveways), and what survives is a ribbon
    running down the middle of each arterial — nearly its centreline where
    the road is just wide enough, proportionally fatter where it opens out
    into an interchange or a plaza.
  * **dilate** that ribbon by half `min_feature_mm`. Now every groove is at
    least one nozzle across whatever the scale, and the wide places stay
    wide.

The result is subtracted from the building blocks (in `routes.py`, the same
way the 銘板's footprint is), never from the terrain: the map keeps its real
surface, only the massing is cut.

Which roads is decided by width, and the width has to be measured off the
geometry — the erosion above is that measurement. PLATEAU carries no road
width: `uro:RoadStructureAttribute` is present but holds only `sectionType`
(一般部 / 交差点部) in every dataset checked, from three different producers,
and `uro:width` and `numberOfLanes` are never populated. Nor does the
classification help: `tran:function` tags 国道 reliably (43 of 大田区's 1,610)
but leaves 環七 — a 都道 — in the 1,566 coded 「その他」, and the 2024 横浜 set
carries no attributes at all.

Files are per 3rd-level mesh like `bldg`/`brid`, and each mesh's road surface
is unioned once at parse time and cached as WKB, so the request path only
unions a few hundred mesh-sized polygons.
"""
from __future__ import annotations

import hashlib

import numpy as np
import requests
import shapely
from lxml import etree
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import transform

from ..config import DATA_DIR
from .buildings import _mesh3_codes, _rings
from .massing import polygon_parts
from .net import atomic_savez, session
from .parallel import process_map
from .plateau import fetch_datacatalog_cities
from .mesh import Projection

_TRAN_NS = "http://www.opengis.net/citygml/transportation/2.0"
_GML_NS = "http://www.opengis.net/gml"
_ROAD_TAG = f"{{{_TRAN_NS}}}Road"
_POLYGON_TAG = f"{{{_GML_NS}}}Polygon"

# 都市計画の標準幅員 (m). A 幹線街路 — the roads a city plan draws as the arterial
# frame — starts at 22 m, which is what four lanes plus footways measures; the
# steps above it are the widths a 主要幹線 is planned at. Starting at 22 m is
# not a matter of taste, and it is why there is no slider for it.
ARTERIAL_WIDTHS_M = (22.0, 25.0, 30.0, 36.0, 40.0, 50.0)
# ...but a groove is one nozzle wide whatever ground that covers, so past about
# 1:150,000 even the arterial frame is more street than there is plate. Then the
# cut steps up the ladder and shows only the 主要幹線 it can afford.
MAX_STREET_SHARE = 1.0 / 3.0


def _road_surface(road: etree._Element):
    """One Road feature's LOD1 surface as shapely polygons in lon/lat.

    Roads are flat, so only the plan matters and the posList's height is
    dropped. PLATEAU surfaces are occasionally self-touching; `make_valid`
    turns those into something `union_all` can take rather than losing the
    road.
    """
    out = []
    for poly in road.iter(_POLYGON_TAG):
        ext, holes = _rings(poly)
        if len(ext) < 3:
            continue
        p = Polygon(ext[:, :2], [h[:, :2] for h in holes if len(h) >= 3])
        out.extend(polygon_parts(p if p.is_valid else shapely.make_valid(p)))
    return out


def _cache_path(mesh: str, url: str):
    key = hashlib.sha1(url.encode()).hexdigest()[:16]
    return DATA_DIR / "roads" / f"{mesh}_{key}.wkb.npz"


def _surface(mesh: str, url: str):
    """Cached road surface for one tran GML: one lon/lat geometry, or None.

    Module-level so `process_map` can ship it to parse workers by reference.
    """
    cache = _cache_path(mesh, url)
    if cache.is_file():
        return shapely.from_wkb(np.load(cache)["wkb"].tobytes())

    polys = []
    try:
        with session().get(url, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            resp.raw.decode_content = True
            for _, road in etree.iterparse(resp.raw, tag=_ROAD_TAG):
                polys.extend(_road_surface(road))
                road.clear()
    except (requests.RequestException, OSError, ValueError):
        return None

    # Union per mesh so the request path joins mesh-sized pieces, not millions
    # of lot-sized ones. Roads run continuously across a mesh border, so the
    # seam is closed there by the caller's union, not here.
    merged = shapely.union_all(polys) if polys else Polygon()
    atomic_savez(cache, wkb=np.frombuffer(shapely.to_wkb(merged), dtype=np.uint8))
    return merged


def _warm_surface(mesh: str, url: str) -> bool:
    """Parse-worker job: ensure one GML's cache exists (True on success)."""
    return _surface(mesh, url) is not None


class PlateauRoadProvider:
    """PLATEAU tran (road) provider. Covers PLATEAU cities only."""

    def _tran_urls(self, codes: list[str]) -> dict[str, list[str]]:
        """Map covered 8-digit mesh -> every tran GML URL (one per municipality)."""
        wanted = set(codes)
        out: dict[str, list[str]] = {}
        for city in fetch_datacatalog_cities(codes):
            for entry in city.get("files", {}).get("tran", []) or []:
                mesh, url = str(entry.get("code")), entry.get("url")
                if mesh in wanted and url and url not in out.setdefault(mesh, []):
                    out[mesh].append(url)
        return {m: u for m, u in out.items() if u}

    def road_cut(self, proj: Projection, min_feature_mm: float = 0.8,
                 outline=None):
        """The grooves to subtract from the building blocks (print mm), or None.

        Takes no threshold: a 幹線街路 is 22 m and up wherever it is, so that
        is where the cut starts. It only widens its mind when the plate cannot
        hold that many streets (`MAX_STREET_SHARE`), stepping up the standard
        widths until the arterial frame fits. `outline` is the printed model
        outline; the fetched grid rectangle by default.
        """
        grid = proj.grid
        bbox = (grid.lons.min(), grid.lats.min(), grid.lons.max(), grid.lats.max())
        urls = self._tran_urls(_mesh3_codes(bbox))
        if not urls:
            return None

        pairs = [(m, u) for m, us in urls.items() for u in us]
        fresh = [p for p in pairs if not _cache_path(*p).is_file()]
        failed = {p for p, ok in zip(fresh, process_map(_warm_surface, fresh)) if not ok}

        parts, seen = [], set()
        for mesh, mesh_urls in urls.items():
            for url in mesh_urls:
                if (mesh, url) in failed:
                    continue
                geom = _surface(mesh, url)
                if geom is None or geom.is_empty:
                    continue
                # A border mesh's files are either city-partitioned or the same
                # mesh-wide content duplicated per city; render identical once.
                digest = hashlib.sha1(shapely.to_wkb(geom)).digest()
                if digest in seen:
                    continue
                seen.add(digest)
                parts.append(geom)
        if not parts:
            return None

        roads = transform(lambda x, y: (proj.x_of(x), proj.y_of(y)),
                          shapely.union_all(parts))
        if outline is None:
            outline = box(
                float(proj.x_of(grid.lons.min())), float(proj.y_of(grid.lats.min())),
                float(proj.x_of(grid.lons.max())), float(proj.y_of(grid.lats.max())),
            )
        cut = None
        for width_m in ARTERIAL_WIDTHS_M:
            wider = grooves(roads, min_feature_mm, width_m * proj.scale)
            if wider is None:
                break            # nothing is that wide; keep the last that was
            cut = wider
            if cut.area <= MAX_STREET_SHARE * outline.area:
                break
        return cut


def grooves(roads, min_feature: float, min_road: float):
    """Road surface (print mm) -> printable grooves, or None if none qualify.

    Erode to drop everything narrower than `min_road` and leave the arterials
    as centre ribbons, then dilate that back out to one nozzle. A road that
    pinches below `min_road` anywhere is severed there, which is honest: at
    that point it is not wide enough to divide anything.
    """
    core = roads.buffer(-0.5 * min_road)
    if core.is_empty:
        return None
    # Round joins, unlike the mitred ones the buildings use: a mitre spikes
    # outward at every junction corner, and a road network is nearly all
    # junctions.
    cut = core.buffer(0.5 * min_feature)
    # A road that only just clears the threshold leaves a crumb of a core, and
    # dilating a crumb gives a disc — a dot punched in a block rather than a
    # street through it. Half the perimeter is a piece's length, so anything
    # not a few nozzles long is one of those.
    parts = [p for p in polygon_parts(cut)
             if 0.5 * p.exterior.length >= 3.0 * min_feature]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)
