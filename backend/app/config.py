"""Runtime configuration, sourced from environment variables."""
from __future__ import annotations

import os
from pathlib import Path

# Directory holding the built frontend (Vite output). In the Docker image the
# frontend build is copied here; in local dev it may not exist (use Vite dev server).
STATIC_DIR = Path(os.environ.get("STATIC_DIR", "/app/static"))

# Persistent cache for DEM / PLATEAU data (mounted as a Docker volume).
DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))

# Upload ceilings, in bytes. Both files are read whole into memory and neither
# is large in honest use: a GPX is text, and even a multi-day outing at one
# point a second is a few megabytes; a nameplate SVG is line art, and one that
# needs more than this is past what `nameplate._MAX_SUBPATHS` will draw anyway.
# MAX_REQUEST_BYTES is the ceiling on the whole multipart body, enforced before
# it is parsed — the per-file numbers cannot protect anything on their own,
# because by the time a route reads an UploadFile it has already been spooled
# to disk. Overridable so an operator can tighten them without a rebuild.
MAX_GPX_BYTES = int(os.environ.get("MAX_GPX_BYTES", 20 * 1024 * 1024))
MAX_SVG_BYTES = int(os.environ.get("MAX_SVG_BYTES", 5 * 1024 * 1024))
MAX_REQUEST_BYTES = int(os.environ.get("MAX_REQUEST_BYTES", 32 * 1024 * 1024))

# CORS origins for local dev (Vite dev server). Comma-separated.
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if o.strip()
]
