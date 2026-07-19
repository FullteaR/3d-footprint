"""PLATEAU CityGML data-catalog client (shared by coloring / buildings / bridges).

The datacatalog API maps JIS mesh codes to each covering city's CityGML file
URLs, grouped by package (luse / bldg / brid / ...):
https://api.plateauview.mlit.go.jp/datacatalog/citygml/m:{codes}
"""
from __future__ import annotations

import threading
import time

import requests

from .net import session
from .parallel import thread_map

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
        return None
    if resp.status_code != 200:
        return None
    return resp.json().get("cities", [])


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
