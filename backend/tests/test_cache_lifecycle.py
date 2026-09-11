import os
import threading
import time

import numpy as np

from app.core import cache, plateau, report
from app.core.net import atomic_write_bytes, atomic_savez, _key_locks, keyed_lock


def test_oldest_cache_is_evicted_before_admitting_new_data(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cache, "CACHE_MAX_BYTES", 10)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    atomic_write_bytes(a, b"123456")
    atomic_write_bytes(b, b"abcdef")
    assert not a.exists()
    assert b.read_bytes() == b"abcdef"


def test_oversized_entry_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cache, "CACHE_MAX_BYTES", 3)
    path = tmp_path / "large.png"
    atomic_write_bytes(path, b"1234")
    assert not path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_negative_cache_expires_even_when_repeatedly_read(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "ABSENT_TTL_SECONDS", 10)
    path = tmp_path / "tile.absent"
    path.write_bytes(b"")
    assert cache.read_bytes(path) == b""
    old = time.time() - 20
    os.utime(path, (old, old))
    assert cache.read_bytes(path) is None


def test_concurrent_npz_reader_never_sees_partial_or_mixed_arrays(tmp_path):
    path = tmp_path / "grid.npz"
    failures = []
    def writer():
        for i in range(20):
            atomic_savez(path, a=np.full(1000, i), b=np.full(1000, i))
    thread = threading.Thread(target=writer)
    thread.start()
    while thread.is_alive():
        data = cache.load_npz(path)
        if data is not None and not np.array_equal(data["a"], data["b"]):
            failures.append(data)
    thread.join()
    assert not failures
    assert cache.load_npz(path)["a"][0] == 19


def test_evicted_and_corrupt_npz_are_cache_misses(tmp_path):
    path = tmp_path / "grid.npz"
    assert cache.load_npz(path) is None
    path.write_bytes(b"broken zip")
    assert cache.load_npz(path) is None


def test_key_locks_are_released_after_last_user():
    key = ("test", object())
    with keyed_lock(key):
        assert key in _key_locks
    assert key not in _key_locks


def test_catalog_failure_is_reported_but_empty_coverage_is_not_a_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("JOB_REPORT_DIR", str(tmp_path))
    class Response:
        status_code = 503
        def json(self):
            return {"cities": []}
    response = Response()
    class Session:
        def get(self, *args, **kwargs):
            return response
    monkeypatch.setattr(plateau, "session", Session)
    assert plateau._fetch_chunk(("test",)) is None
    assert (tmp_path / "warnings" / "plateau_catalog-fetch_failed.json").is_file()
    response.status_code = 200
    assert plateau._fetch_chunk(("empty",)) == []


def test_failed_fetch_is_not_also_reported_as_no_coverage(tmp_path, monkeypatch):
    monkeypatch.setenv("JOB_REPORT_DIR", str(tmp_path))
    report.warn("jaxa", "no_coverage")
    report.warn("jaxa")
    assert not (tmp_path / "warnings" / "jaxa-no_coverage.json").exists()
    report.warn("plateau_catalog")
    report.warn("landuse", "no_coverage")
    assert not (tmp_path / "warnings" / "landuse-no_coverage.json").exists()


def test_only_abandoned_atomic_write_temps_are_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cache, "JOB_TIMEOUT_SECONDS", 10)
    abandoned, active = tmp_path / ".old.tmp", tmp_path / ".active.tmp"
    abandoned.write_bytes(b"partial")
    active.write_bytes(b"partial")
    old = time.time() - 30
    os.utime(abandoned, (old, old))
    cache.prune()
    assert not abandoned.exists()
    assert active.exists()
