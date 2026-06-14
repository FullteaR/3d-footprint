"""Runtime configuration, sourced from environment variables."""
from __future__ import annotations

import os
from pathlib import Path

# Directory holding the built frontend (Vite output). In the Docker image the
# frontend build is copied here; in local dev it may not exist (use Vite dev server).
STATIC_DIR = Path(os.environ.get("STATIC_DIR", "/app/static"))

# Persistent cache for DEM / PLATEAU data (mounted as a Docker volume).
DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))

# CORS origins for local dev (Vite dev server). Comma-separated.
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if o.strip()
]
