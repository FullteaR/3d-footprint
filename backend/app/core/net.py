"""Shared fetch plumbing: keep-alive HTTP session, atomic cache writes, key locks.

One `requests.Session` per process so the hundreds of small tile/catalog
requests reuse TCP+TLS connections instead of re-handshaking every time
(urllib3's connection pool is thread-safe; no cookies or other session state
are used). Cache files are written via temp file + atomic rename so parallel
workers — threads here, parse processes elsewhere — can never expose a torn
file to a concurrent reader, and `keyed_lock` serializes work on one cache key
so concurrent workers don't download the same tile twice.
"""
from __future__ import annotations

import os
import threading
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import requests
from requests.adapters import HTTPAdapter

USER_AGENT = "3d-footprint/0.1"

_session: requests.Session | None = None
_guard = threading.Lock()


def session() -> requests.Session:
    """Process-wide keep-alive session (lazy; shared across threads)."""
    global _session
    with _guard:
        if _session is None:
            s = requests.Session()
            s.headers["User-Agent"] = USER_AGENT
            adapter = HTTPAdapter(pool_connections=8, pool_maxsize=32)
            s.mount("https://", adapter)
            s.mount("http://", adapter)
            _session = s
        return _session


def _tmp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes via temp + rename so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    """np.savez_compressed via temp + rename (same torn-file guarantee)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


_key_locks: dict[object, threading.Lock] = defaultdict(threading.Lock)


@contextmanager
def keyed_lock(key: object):
    """Serialize one cache key's fetch across threads (dedupes downloads)."""
    with _guard:
        lock = _key_locks[key]
    with lock:
        yield
