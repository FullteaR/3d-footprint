"""Which PLATEAU CityGML files cover an area, and reading them once each.

The datacatalog API maps JIS mesh codes to each covering city's CityGML file
URLs, grouped by package (luse / bldg / brid / tran):
https://api.plateauview.mlit.go.jp/datacatalog/citygml/m:{codes}

`mesh3_codes` turns a bbox into the codes to ask about, `file_urls` into the
files to read, and `distinct_files` parses them — in parallel, cached on disk,
and each distinct content only once however many cities publish it. What is
*in* one of those files is `citygml.py`'s side of the job.
"""
from __future__ import annotations

from . import cache as disk_cache, report

import hashlib
import threading
import time
from pathlib import Path
from typing import Callable

import requests

from ..config import DATA_DIR
from .net import session
from .parallel import process_map, thread_map

DATACATALOG_URL = "https://api.plateauview.mlit.go.jp/datacatalog/citygml/m:{codes}"
# The datacatalog rejects more than 30 mesh codes per request ("too many
# bounds"). 3rd-level (bldg) meshes are ~1 km, so a multi-km route easily
# exceeds this; query in chunks and merge.
DATACATALOG_MAX_CODES = 30
CATALOG_WORKERS = 8      # concurrent chunk queries
CATALOG_TTL_S = 3600.0   # memo lifetime (the catalog changes ~annually)
_MEMO_MAX = 64

# Buildings, bridges and land use all resolve the same area within one
# /generate call (bldg and brid even share the exact code list), and the
# preview loop repeats the same area across calls — memoize per code list.
_memo: dict[tuple[str, ...], tuple[float, list[dict]]] = {}
_memo_guard = threading.Lock()


def _fetch_chunk(chunk: tuple[str, ...]) -> list[dict] | None:
    """One catalog query; None on failure (vs. an honest empty result)."""
    try:
        resp = session().get(DATACATALOG_URL.format(codes=",".join(chunk)), timeout=60)
    except requests.RequestException:
        report.warn("plateau_catalog")
        return None
    if resp.status_code != 200:
        report.warn("plateau_catalog")
        return None
    try:
        cities = resp.json().get("cities", [])
        if not isinstance(cities, list):
            raise ValueError("invalid catalog")
        return cities
    except (ValueError, AttributeError):
        report.warn("plateau_catalog", "parse_failed")
        return None


def fetch_datacatalog_cities(codes: list[str]) -> list[dict]:
    """Return the merged ``cities`` list for ``codes``, chunked under the API cap.

    Chunks are fetched in parallel and merged in chunk order (painting order
    matters downstream). A fully successful result is memoized for
    CATALOG_TTL_S; a partial one (some chunk failed) is returned but not
    memoized, so a later request can heal it.
    """
    key = tuple(codes)
    now = time.monotonic()
    with _memo_guard:
        hit = _memo.get(key)
        if hit is not None and now - hit[0] < CATALOG_TTL_S:
            return hit[1]

    chunks = [
        tuple(codes[i : i + DATACATALOG_MAX_CODES])
        for i in range(0, len(codes), DATACATALOG_MAX_CODES)
    ]
    parts = thread_map(_fetch_chunk, [(c,) for c in chunks], CATALOG_WORKERS)
    cities: list[dict] = []
    for part in parts:
        if part:
            cities.extend(part)
    if all(part is not None for part in parts):
        with _memo_guard:
            if len(_memo) >= _MEMO_MAX:  # drop the stalest entry
                del _memo[min(_memo, key=lambda k: _memo[k][0])]
            _memo[key] = (now, cities)
    return cities


MESH3_DLAT = 1.0 / 120.0  # 3rd-level mesh latitude span (30 arc-sec)
MESH3_DLON = 1.0 / 80.0   # 3rd-level mesh longitude span (45 arc-sec)


def mesh3_codes(bbox: tuple[float, float, float, float]) -> list[str]:
    """3rd-level (8-digit) JIS mesh codes covering a bbox."""
    min_lon, min_lat, max_lon, max_lat = bbox

    def code(lat: float, lon: float) -> str:
        p, u = int(lat * 1.5), int(lon) - 100
        lat1, lon1 = p / 1.5, u + 100
        q = int((lat - lat1) / (1.0 / 12.0))            # 2nd mesh row (0..7)
        v = int((lon - lon1) / (1.0 / 8.0))             # 2nd mesh col (0..7)
        r = int((lat - lat1 - q / 12.0) / MESH3_DLAT)   # 3rd mesh row (0..9)
        w = int((lon - lon1 - v / 8.0) / MESH3_DLON)    # 3rd mesh col (0..9)
        return f"{p:02d}{u:02d}{q}{v}{r}{w}"

    codes = set()
    lat = min_lat
    while lat <= max_lat + MESH3_DLAT:
        lon = min_lon
        while lon <= max_lon + MESH3_DLON:
            codes.add(code(lat, lon))
            lon += MESH3_DLON
        lat += MESH3_DLAT
    return sorted(codes)


def file_urls(package: str, codes: list[str]) -> dict[str, list[str]]:
    """Map each covered mesh -> every `package` GML URL (one per municipality).

    A mesh straddling a city border appears in each city's dataset, and the
    files are either partitioned (each city carries only its own features) or
    the identical mesh-wide content duplicated per city (the 2025 pref sets),
    dataset-dependent. So all of them are needed — keeping just the first
    would drop the other side of a border — and the duplicates are settled by
    content in `distinct_files`. A city spanning several query chunks is
    returned once per chunk, hence the URL dedupe here.
    """
    wanted = set(codes)
    out: dict[str, list[str]] = {}
    for city in fetch_datacatalog_cities(codes):
        for entry in city.get("files", {}).get(package, []) or []:
            mesh, url = str(entry.get("code")), entry.get("url")
            if mesh in wanted and url and url not in out.setdefault(mesh, []):
                out[mesh].append(url)
    return {m: u for m, u in out.items() if u}


def cache_file(kind: str, mesh: str, url: str, suffix: str = ".npz") -> Path:
    """Where one parsed GML is kept. The URL is what identifies it: the same
    mesh is published by every city covering it, at a different URL each."""
    key = hashlib.sha1(url.encode()).hexdigest()[:16]
    return DATA_DIR / kind / f"{mesh}_{key}{suffix}"


def _warm(load: Callable, mesh: str, url: str) -> bool:
    """Parse-worker job: ensure one GML's cache exists (True on success).

    Module-level, and taking `load` as an argument, so `process_map` can ship
    both to a spawned worker by reference. Only the flag comes back: the
    parsed arrays stay in the worker's cache file, which is the whole point of
    caching them there rather than returning them.
    """
    return load(mesh, url) is not None


def distinct_files(urls: dict[str, list[str]], *, cache_path, load, key):
    """Yield each distinct file's parsed content once, in catalog order.

    `cache_path(mesh, url)` says where a file's parse is kept, `load` returns
    it (parsing on a miss), and `key` is the content digest that decides what
    is a duplicate — or None for a file holding nothing to render.

    First use of an area parses every uncached file on the process pool: they
    are independent and CPU-heavy, and the caller then reads the caches back
    in order. A file that fails to parse stays uncached and is skipped, rather
    than being retried here — a second download attempt is what the *next*
    request is for.
    """
    pairs = [(mesh, url) for mesh, mesh_urls in urls.items() for url in mesh_urls]
    fresh = [p for p in pairs if not cache_path(*p).is_file()]
    warmed = process_map(_warm, [(load, mesh, url) for mesh, url in fresh])
    failed = {p for p, ok in zip(fresh, warmed) if not ok}

    seen: set[bytes] = set()
    for pair in pairs:
        if pair in failed:
            continue
        content = load(*pair)
        if content is None:
            continue
        digest = key(content)
        if digest is None or digest in seen:
            continue
        seen.add(digest)
        yield content
