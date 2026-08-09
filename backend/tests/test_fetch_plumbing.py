"""Cache writes and worker pools — the plumbing under every data fetcher.

Nothing here goes near the network: what matters is that a cache file is never
readable half-written, that one key is only fetched once, and that the pools
keep job order and stay strictly an optimization.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from app.core import parallel
from app.core.net import atomic_savez, atomic_write_bytes, keyed_lock, session
from app.core.parallel import process_map, thread_map


# ---- atomic cache writes ---------------------------------------------------

def test_bytes_land_whole_and_leave_no_litter(tmp_path):
    path = tmp_path / "sub" / "tile.png"
    atomic_write_bytes(path, b"\x89PNG...")
    assert path.read_bytes() == b"\x89PNG..."
    assert [p.name for p in path.parent.iterdir()] == ["tile.png"]


def test_a_rewrite_replaces_the_file_in_one_step(tmp_path):
    path = tmp_path / "tile.png"
    atomic_write_bytes(path, b"old")
    atomic_write_bytes(path, b"new")
    assert path.read_bytes() == b"new"


def test_arrays_round_trip_through_the_npz_cache(tmp_path):
    path = tmp_path / "rings.npz"
    atomic_savez(path, coords=np.arange(6.0).reshape(3, 2),
                 codes=np.array([204, 211], np.uint16))
    z = np.load(path)
    assert z["coords"].shape == (3, 2)
    assert z["codes"].tolist() == [204, 211]
    assert [p.name for p in path.parent.iterdir()] == ["rings.npz"]


def test_a_failed_write_does_not_leave_a_temp_file(tmp_path, monkeypatch):
    path = tmp_path / "rings.npz"
    monkeypatch.setattr(np, "savez_compressed",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        atomic_savez(path, a=np.zeros(3))
    assert list(path.parent.iterdir()) == []


# ---- key locks -------------------------------------------------------------

def test_one_key_is_only_worked_on_once_at_a_time():
    """2x2 z15 tiles share a z14 fallback; without this they all fetch it."""
    overlaps = []
    live = 0
    guard = threading.Lock()

    def worker():
        nonlocal live
        with keyed_lock(("dem", 15, 1, 2)):
            with guard:
                live += 1
                overlaps.append(live)
            with guard:
                live -= 1

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert max(overlaps) == 1


def test_different_keys_do_not_block_each_other():
    with keyed_lock(("dem", 15, 1, 2)):
        with keyed_lock(("dem", 15, 9, 9)):
            pass


def test_the_session_is_shared_and_identifies_itself():
    assert session() is session()
    assert session().headers["User-Agent"].startswith("3d-footprint")


# ---- pools -----------------------------------------------------------------

def double(x):
    return x * 2


def test_thread_map_keeps_job_order():
    jobs = [(i,) for i in range(20)]
    assert thread_map(double, jobs, workers=8) == [i * 2 for i in range(20)]


def test_thread_map_runs_a_lone_job_in_place():
    """No pool for one tile — spinning up threads would cost more than it saves."""
    assert thread_map(double, [(3,)], workers=8) == [6]
    assert thread_map(double, [], workers=8) == []


def test_thread_map_passes_every_argument():
    assert thread_map(lambda a, b: a + b, [(1, 2), (3, 4)], workers=2) == [3, 7]


def test_a_job_exception_surfaces(monkeypatch):
    def boom(x):
        raise RuntimeError("parse failed")

    with pytest.raises(RuntimeError, match="parse failed"):
        thread_map(boom, [(1,), (2,)], workers=2)


def test_process_map_keeps_job_order_without_a_pool(monkeypatch):
    """PARSE_PROCS=1 disables the pool; the result must be identical."""
    monkeypatch.setattr(parallel, "PARSE_PROCS", 1)
    assert process_map(double, [(i,) for i in range(5)]) == [0, 2, 4, 6, 8]
    assert process_map(double, []) == []


def test_a_parse_job_exception_propagates_as_itself(monkeypatch):
    """Only pool *infrastructure* failures fall back to sequential execution;
    a job's own error is the caller's to handle."""
    monkeypatch.setattr(parallel, "PARSE_PROCS", 1)

    def boom(x):
        raise ValueError("bad gml")

    with pytest.raises(ValueError, match="bad gml"):
        process_map(boom, [(1,), (2,)])


@pytest.mark.slow
def test_process_map_really_uses_the_pool():
    """Spawned workers, order preserved — `os.getpid` is picklable by name."""
    import os

    if parallel.PARSE_PROCS < 2:
        pytest.skip("pool disabled on this box")
    pids = process_map(os.getpid, [(), (), (), ()])
    assert len(pids) == 4
    assert os.getpid() not in pids
