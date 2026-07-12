"""PLATEAU CityGML data-catalog client (shared by coloring / buildings / bridges).

The datacatalog API maps JIS mesh codes to each covering city's CityGML file
URLs, grouped by package (luse / bldg / brid / ...):
https://api.plateauview.mlit.go.jp/datacatalog/citygml/m:{codes}
"""
from __future__ import annotations

import requests

DATACATALOG_URL = "https://api.plateauview.mlit.go.jp/datacatalog/citygml/m:{codes}"
# The datacatalog rejects more than 30 mesh codes per request ("too many
# bounds"). 3rd-level (bldg) meshes are ~1 km, so a multi-km route easily
# exceeds this; query in chunks and merge.
DATACATALOG_MAX_CODES = 30


def fetch_datacatalog_cities(codes: list[str]) -> list[dict]:
    """Return the merged ``cities`` list for ``codes``, chunked under the API cap."""
    cities: list[dict] = []
    for i in range(0, len(codes), DATACATALOG_MAX_CODES):
        chunk = codes[i : i + DATACATALOG_MAX_CODES]
        try:
            resp = requests.get(
                DATACATALOG_URL.format(codes=",".join(chunk)),
                headers={"User-Agent": "3d-footprint/0.1"},
                timeout=60,
            )
        except requests.RequestException:
            continue
        if resp.status_code == 200:
            cities.extend(resp.json().get("cities", []))
    return cities
