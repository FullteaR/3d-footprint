"""Bounded disposable data cache. Only downloaded/derived data belongs here."""
from __future__ import annotations

import fcntl
import io
import os
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ..config import (
    DATA_DIR, CACHE_MAX_BYTES, CACHE_TTL_SECONDS, ABSENT_TTL_SECONDS, JOB_TIMEOUT_SECONDS,
)


def read_bytes(path: Path) -> bytes | None:
    try:
        ttl = ABSENT_TTL_SECONDS if path.suffix == ".absent" else CACHE_TTL_SECONDS
        if time.time() - path.stat().st_mtime > ttl:
            return None
        # Open/read in one operation. Eviction after open is safe on POSIX;
        # eviction before open is an ordinary miss.
        data = path.read_bytes()
        try:
            stat = path.stat()
            os.utime(path, (time.time(), stat.st_mtime))
        except FileNotFoundError:
            pass
        return data
    except FileNotFoundError:
        return None


def load_npz(path: Path) -> dict | None:
    data = read_bytes(path)
    if data is None:
        return None
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            return {key: z[key] for key in z.files}
    except (ValueError, OSError, EOFError, zipfile.BadZipFile):
        return None


@contextmanager
def write_budget(path: Path, size: int):
    """Serialize admission/eviction across processes before atomic replacement.

    Reads do not lock: an opened inode stays valid during unlink/replace.
    Temporary files and the lock are excluded; jobs live in a separate root.
    An entry larger than the entire budget is usable but is not cached.
    """
    if not path.is_relative_to(DATA_DIR):
        yield True
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (DATA_DIR / ".cache.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if size > CACHE_MAX_BYTES:
            yield False
            return
        now = time.time()
        files = []
        for parent, dirs, names in os.walk(DATA_DIR):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for name in names:
                p = Path(parent) / name
                if p.is_symlink():
                    continue
                try:
                    s = p.stat()
                    if name.startswith("."):
                        # SIGKILL can interrupt an atomic writer before its
                        # finally block runs. Only reap abandoned temp files;
                        # live jobs are bounded by JOB_TIMEOUT_SECONDS.
                        if name.endswith(".tmp") and now - s.st_mtime > 2 * JOB_TIMEOUT_SECONDS:
                            p.unlink(missing_ok=True)
                        continue
                    ttl = ABSENT_TTL_SECONDS if p.suffix == ".absent" else CACHE_TTL_SECONDS
                    if now - s.st_mtime > ttl:
                        p.unlink(missing_ok=True)
                    elif p != path:
                        files.append((s.st_atime, p, s.st_size))
                except FileNotFoundError:
                    continue
        total = sum(s for _, _, s in files) + size
        count = len(files) + 1
        for _, p, n in sorted(files):
            if total <= CACHE_MAX_BYTES and count <= 100_000:
                break
            p.unlink(missing_ok=True)
            total -= n
            count -= 1
        yield True


def prune() -> None:
    with write_budget(DATA_DIR / ".maintenance", 0):
        pass
