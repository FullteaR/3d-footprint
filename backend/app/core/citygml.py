"""Reading one PLATEAU CityGML file: the parts every feature layer shares.

bldg (buildings), brid (bridges) and tran (roads) are the same document
format carrying different features, so the mechanics of getting geometry out
of one are the same in all three: stream the file rather than hold it, read a
gml:Polygon's rings out of its posLists, and triangulate a planar 3D ring set
into something a mesh can be built from. Only the feature namespace and what
is done with the result differ, and those stay in the layer's own module.

Which files to read is a separate question, answered by `plateau.py`.
"""
from __future__ import annotations

import mapbox_earcut as earcut
import numpy as np
from lxml import etree

from . import safexml
from .mesh import _M_PER_DEG_LAT, _M_PER_DEG_LON
from .net import session

GML_NS = "http://www.opengis.net/gml"
_NS = {"gml": GML_NS}
POLYGON_TAG = f"{{{GML_NS}}}Polygon"


def stream_features(url, tag: str):
    """Yield each `tag` feature of a CityGML file as it arrives.

    A city's GML is tens of megabytes and the geometry it holds is a fraction
    of that, so the document is parsed off the socket and each feature is
    cleared once the caller is done with it — nothing but the extracted arrays
    is ever held. Network and parse errors propagate to the caller, which is
    what decides whether a missing file is fatal.
    """
    with session().get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        resp.raw.decode_content = True
        for _, el in safexml.iterparse(resp.raw, tag):
            yield el
            el.clear()


def mesh_lat_mid(mesh: str) -> float:
    """A rough mid-latitude for a 3rd-level mesh code.

    Only ever used as the metric basis a triangulation is projected onto, so
    the mesh's own half-degree band is precise enough.
    """
    return (int(mesh[:2]) / 1.5) + 0.5


def _poslist(ring: etree._Element) -> np.ndarray:
    """LinearRing -> (n,3) lon,lat,height (dropping the repeated closing point)."""
    vals = ring.findtext("gml:posList", namespaces=_NS)
    if not vals:
        return np.empty((0, 3))
    a = np.array(vals.split(), dtype=float).reshape(-1, 3)
    if len(a) > 1 and np.allclose(a[0], a[-1]):
        a = a[:-1]
    return a[:, [1, 0, 2]]  # posList is lat lon h -> store lon lat h


def rings(poly: etree._Element) -> tuple[np.ndarray, list[np.ndarray]]:
    """A gml:Polygon's exterior ring and its interior rings (holes)."""
    ext_el = poly.find("gml:exterior/gml:LinearRing", _NS)
    ext = _poslist(ext_el) if ext_el is not None else np.empty((0, 3))
    holes = [_poslist(r) for r in poly.findall("gml:interior/gml:LinearRing", _NS)]
    return ext, [h for h in holes if len(h) >= 3]


def triangulate(ext: np.ndarray, holes: list[np.ndarray], lat_mid: float):
    """Triangulate a planar 3D polygon; return (points (k,3), faces (t,3))."""
    all_rings = [ext] + holes
    pts = np.vstack(all_rings)
    if len(pts) < 3:
        return None
    # Project to a local metric plane, drop the axis most aligned with the
    # polygon normal, and earcut the remaining two coordinates.
    klon = _M_PER_DEG_LON * np.cos(np.radians(lat_mid))
    metric = pts * np.array([klon, _M_PER_DEG_LAT, 1.0])
    x, y, z = metric[: len(ext)].T
    nx_ = np.sum((y - np.roll(y, -1)) * (z + np.roll(z, -1)))
    ny_ = np.sum((z - np.roll(z, -1)) * (x + np.roll(x, -1)))
    nz_ = np.sum((x - np.roll(x, -1)) * (y + np.roll(y, -1)))
    drop = int(np.argmax(np.abs([nx_, ny_, nz_])))
    keep = [i for i in range(3) if i != drop]
    verts2d = np.ascontiguousarray(metric[:, keep], dtype=np.float64)
    ring_ends = np.cumsum([len(r) for r in all_rings]).astype(np.uint32)
    try:
        idx = earcut.triangulate_float64(verts2d, ring_ends)
    except Exception:
        return None
    if len(idx) < 3:
        return None
    return pts, np.asarray(idx, dtype=np.int64).reshape(-1, 3)
