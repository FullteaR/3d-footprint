"""Terrain coloring: PLATEAU land use (luse), painted as-is.

One source, no post-processing. The PLATEAU data catalog maps the DEM grid's
bbox to per-city luse CityGML files (one 2nd-level JIS mesh each); every file's
LandUse parcels (都市計画基礎調査の土地利用現況) are stream-parsed once into a
compact npz of polygon rings + class code, then rasterized straight onto the
DEM grid in document order. The official class codes
(codelists/Common_landUseType.xml) map to a small print palette.

Parcels of unknown class (231 不明) and cells no parcel covers fall through
to the JAXA HRLULC fallback (jaxa.py) — strictly gap-fill, painted just as
literally: it never overrides a PLATEAU-classified cell. Cells neither source
classifies keep the "terrain" label, i.e. the plain user-picked terrain
colour. No smoothing, no sea stamping, no DEM-based corrections anywhere.
"""
from __future__ import annotations

import hashlib

import numpy as np
import requests
from lxml import etree
from PIL import Image, ImageDraw

from ..config import DATA_DIR
from . import jaxa
from .plateau import fetch_datacatalog_cities
from .terrain import ElevationGrid

# Common_landUseType class code -> print category (= colour layer in export).
LUSE_CATEGORY: dict[int, str] = {
    204: "water",   # 水面（河川、湖沼、ため池、用水路、濠、運河）
    203: "forest",  # 山林（樹林地）
    217: "forest",  # 公共空地（公園・緑地、広場、運動場、墓園）
    220: "forest",  # その他①（ゴルフ場）
    201: "field",   # 田（水田）
    202: "field",   # 畑（樹園地、採草地等）
    219: "field",   # 農林漁業施設用地
    260: "field",   # 農地（田・畑の区分なし）
    211: "urban",   # 住宅用地
    212: "urban",   # 商業用地
    213: "urban",   # 工業用地
    214: "urban",   # 公益施設用地
    216: "urban",   # 交通施設用地（運輸倉庫施設）
    218: "urban",   # その他公的施設用地（防衛施設）
    222: "urban",   # その他③（平面駐車場）
    251: "urban",   # 可住地
    261: "urban",   # 宅地（住宅・商業等の区分なし）
    215: "road",    # 道路用地（道路、駅前広場等）
    262: "road",    # 道路・鉄軌道敷
    205: "bare",    # その他自然地（原野、荒れ地、低湿地、河川敷、海浜等）
    221: "bare",    # その他②（太陽光発電）
    223: "bare",    # その他④（建物跡地、資材置場、造成中の土地、法面）
    224: "bare",    # 低未利用土地（空地、空家等）
    252: "bare",    # 非可住地
    263: "bare",    # 空地（その他①～④の区分なし）
    # 231 不明 and anything unlisted stay unclassified.
}

UNCLASSIFIED = "terrain"  # cells luse doesn't classify = plain terrain colour
_PALETTE = [UNCLASSIFIED, "water", "forest", "field", "urban", "road", "bare"]
_IDX = {name: i for i, name in enumerate(_PALETTE)}
_HOLE = 1  # ring-code sentinel for interior rings (real classes are >= 201)

_LANDUSE_TAG = "{http://www.opengis.net/citygml/landuse/2.0}LandUse"
_CLASS_TAG = "{http://www.opengis.net/citygml/landuse/2.0}class"
_POLYGON_TAG = "{http://www.opengis.net/gml}Polygon"
_EXTERIOR_TAG = "{http://www.opengis.net/gml}exterior"
_INTERIOR_TAG = "{http://www.opengis.net/gml}interior"
_POSLIST_TAG = "{http://www.opengis.net/gml}posList"


def _mesh2_codes(bbox: tuple[float, float, float, float]) -> list[str]:
    """2nd-level (6-digit) JIS mesh codes covering a bbox."""
    min_lon, min_lat, max_lon, max_lat = bbox
    codes = []
    for i in range(int(min_lat * 12), int(max_lat * 12) + 1):    # lat in 1/12 deg
        for j in range(int(min_lon * 8), int(max_lon * 8) + 1):  # lon in 1/8 deg
            p, q = divmod(i, 8)
            u, v = divmod(j, 8)
            codes.append(f"{p:02d}{u - 100:02d}{q}{v}")
    return codes


def _luse_files(bbox: tuple[float, float, float, float]) -> list[tuple[str, str]]:
    """(mesh_code, url) of every luse CityGML covering the bbox.

    A city overlapping any queried mesh returns its whole file list, so entries
    are filtered back to the wanted meshes (8-digit 3rd-level codes match by
    their 6-digit prefix). Same-mesh files from different cities usually hold
    identical mesh-wide content; painting them all is redundant but harmless,
    and stays correct if a city's file only covers its own territory.
    """
    codes = _mesh2_codes(bbox)
    wanted = set(codes)
    files: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for city in fetch_datacatalog_cities(codes):
        for entry in city.get("files", {}).get("luse", []) or []:
            code, url = str(entry.get("code")), entry.get("url")
            if url and (code in wanted or code[:6] in wanted) and (code, url) not in seen:
                seen.add((code, url))
                files.append((code, url))
    return files


def _parse_luse(stream) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stream-parse LandUse parcels from GML.

    Returns ``(coords, starts, codes, feats)``: ring vertices (N, 2) as
    lon/lat, ring offsets into it (R+1,), per-ring code (R,) — the parcel's
    class on exterior rings, the `_HOLE` sentinel on interior rings — and
    per-parcel ring offsets (F+1,). Parcels that have holes are emitted before
    simple ones, so a parcel sitting inside another's hole is painted after
    the hole punched it and survives the painter's algorithm.
    """
    with_holes: list[list[tuple[int, np.ndarray]]] = []
    simple: list[list[tuple[int, np.ndarray]]] = []
    for _, el in etree.iterparse(stream, tag=_LANDUSE_TAG):
        cl = el.find(_CLASS_TAG)
        try:
            code = int(cl.text) if cl is not None and cl.text else 0
        except ValueError:
            code = 0
        rings: list[tuple[int, np.ndarray]] = []
        for poly in el.iter(_POLYGON_TAG):
            for wrap_tag, value in ((_EXTERIOR_TAG, code), (_INTERIOR_TAG, _HOLE)):
                for wrap in poly.findall(wrap_tag):
                    for pos in wrap.iter(_POSLIST_TAG):
                        vals = np.asarray((pos.text or "").split(), dtype=np.float64)
                        # PLATEAU posLists are (lat lon height) tuples; the
                        # modulo is a guard for the rare 2-D export.
                        step = 3 if vals.size % 3 == 0 else 2
                        lat, lon = vals[0::step], vals[1::step]
                        m = min(lat.size, lon.size)
                        if m >= 3:
                            rings.append((value, np.column_stack([lon[:m], lat[:m]])))
        el.clear()
        if rings:
            (with_holes if any(v == _HOLE for v, _ in rings) else simple).append(rings)

    parcels = with_holes + simple
    if not parcels:
        return (np.empty((0, 2)), np.zeros(1, np.int64),
                np.empty(0, np.uint16), np.zeros(1, np.int64))
    flat = [ring for parcel in parcels for ring in parcel]
    coords = np.vstack([r for _, r in flat])
    starts = np.concatenate([[0], np.cumsum([len(r) for _, r in flat])]).astype(np.int64)
    codes = np.array([v for v, _ in flat], np.uint16)
    feats = np.concatenate([[0], np.cumsum([len(p) for p in parcels])]).astype(np.int64)
    return coords, starts, codes, feats


def _load_rings(code: str, url: str):
    """Parsed rings of one luse file, cached as npz (the GML parse is the slow bit)."""
    key = hashlib.sha1(url.encode()).hexdigest()[:16]
    cache = DATA_DIR / "plateau_luse" / f"{code}_{key}.npz"
    if cache.is_file():
        z = np.load(cache)
        return z["coords"], z["starts"], z["codes"], z["feats"]
    try:
        with requests.get(
            url, headers={"User-Agent": "3d-footprint/0.1"}, stream=True, timeout=300
        ) as resp:
            resp.raise_for_status()
            resp.raw.decode_content = True
            coords, starts, codes, feats = _parse_luse(resp.raw)
    except (requests.RequestException, OSError, ValueError, etree.LxmlError):
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, coords=coords, starts=starts, codes=codes, feats=feats)
    return coords, starts, codes, feats


# Salt the painted-grid memo with the code->category mapping so editing the
# mapping (or the palette) invalidates memoized grids but not the parsed rings.
_MAP_SALT = hashlib.sha1(repr((sorted(LUSE_CATEGORY.items()), _PALETTE)).encode()).hexdigest()[:8]


def _paint(grid: ElevationGrid, files: list[tuple[str, str]]) -> tuple[np.ndarray, bool]:
    """Rasterize the files' parcels onto the grid -> (palette-index grid, all_loaded)."""
    nx, ny = grid.lons.size, grid.lats.size
    # Pixel (col j, row i) == grid node (lons[j], lats[i]); rows follow the
    # ascending-lat grid order and PIL clips anything outside the canvas.
    sx = (nx - 1) / (grid.lons[-1] - grid.lons[0])
    sy = (ny - 1) / (grid.lats[-1] - grid.lats[0])
    img = Image.new("L", (nx, ny), 0)
    draw = ImageDraw.Draw(img)

    all_loaded = True
    seen: set[bytes] = set()
    for code, url in files:
        rings = _load_rings(code, url)
        if rings is None:
            all_loaded = False
            continue
        coords, starts, codes, feats = rings
        if not len(codes):
            continue
        # Each city's dataset usually repeats the identical mesh-wide file;
        # painting byte-identical parcel sets again would only cost time.
        digest = hashlib.sha1(coords.tobytes() + codes.tobytes()).digest()
        if digest in seen:
            continue
        seen.add(digest)
        xs = (coords[:, 0] - grid.lons[0]) * sx
        ys = (coords[:, 1] - grid.lats[0]) * sy
        for f in range(len(feats) - 1):
            r0, r1 = int(feats[f]), int(feats[f + 1])
            idx = _IDX[LUSE_CATEGORY.get(int(codes[r0]), UNCLASSIFIED)]
            if idx == 0:
                continue  # 不明 (231) / unmapped class: leave unclassified
            for r in range(r0, r1):
                a, b = int(starts[r]), int(starts[r + 1])
                draw.polygon(
                    list(zip(xs[a:b], ys[a:b])),
                    fill=0 if codes[r] == _HOLE else idx,
                )
    return np.asarray(img), all_loaded


def _plateau_index_grid(grid: ElevationGrid) -> np.ndarray | None:
    """(ny, nx) palette-index grid from PLATEAU luse (0 = unclassified).

    None when no luse file covers the bbox at all. The painted grid is
    memoized per (bbox, shape, file set), so the preview loop repaints only
    when the area or resolution actually changes; a partial paint (some file
    failed to download) is returned but not memoized, so a later request can
    heal the gap.
    """
    bbox = (float(grid.lons[0]), float(grid.lats[0]),
            float(grid.lons[-1]), float(grid.lats[-1]))
    files = _luse_files(bbox)
    if not files:
        return None

    ny, nx = grid.lats.size, grid.lons.size
    key = hashlib.sha1("|".join(
        [",".join(f"{v:.9f}" for v in bbox), f"{ny}x{nx}", _MAP_SALT]
        + sorted(url for _, url in files)
    ).encode()).hexdigest()[:16]
    memo = DATA_DIR / "plateau_luse" / f"grid_{key}.npz"
    if memo.is_file():
        return np.load(memo)["classes"]
    idx, all_loaded = _paint(grid, files)
    if not all_loaded:
        return idx if idx.any() else None
    memo.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(memo, classes=idx)
    return idx


def category_grid(grid: ElevationGrid) -> np.ndarray | None:
    """(ny, nx) print-category labels for the DEM grid.

    PLATEAU luse wherever it classifies a cell; JAXA HRLULC fills *only* the
    cells PLATEAU leaves unclassified (it never overrides PLATEAU). Cells
    neither source classifies keep the "terrain" label; None when neither
    source covers the bbox at all (plain single-colour terrain).
    """
    nx, ny = grid.lons.size, grid.lats.size
    if nx < 2 or ny < 2:
        return None
    idx = _plateau_index_grid(grid)
    if idx is None or (idx == 0).any():
        classes = jaxa.class_grid(grid)
        if classes is not None:
            lut = np.zeros(256, np.uint8)
            for cls, cat in jaxa.CLASS_CATEGORY.items():
                lut[cls] = _IDX[cat]
            fill = lut[classes]
            idx = fill if idx is None else np.where(idx == 0, fill, idx)
    if idx is None:
        return None
    return np.asarray(_PALETTE, dtype="<U8")[idx]
