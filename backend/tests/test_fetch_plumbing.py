"""Cache writes, worker pools and file dedupe — the plumbing under every fetcher.

Nothing here goes near the network: what matters is that a cache file is never
readable half-written, that one key is only fetched once, that the pools keep
job order and stay strictly an optimization, and that a PLATEAU mesh published
by several cities is still rendered once.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from app.core import parallel, plateau
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


# ---- one PLATEAU file per distinct content ---------------------------------

def _distinct(monkeypatch, tmp_path, urls, contents, cached=(), fails=()):
    """Drive `plateau.distinct_files` over an in-memory set of files.

    Returns (what it yielded, what the parse workers were asked to warm). A
    content of "" stands for a file that parsed to nothing.
    """
    warmed: list[tuple[str, str]] = []

    def fake_process_map(fn, jobs):
        warmed.extend((mesh, url) for _load, mesh, url in jobs)
        return [(mesh, url) not in fails for _load, mesh, url in jobs]

    monkeypatch.setattr(plateau, "process_map", fake_process_map)
    for mesh, url in cached:
        (tmp_path / f"{mesh}_{url}").write_bytes(b"")
    out = list(plateau.distinct_files(
        urls,
        cache_path=lambda mesh, url: tmp_path / f"{mesh}_{url}",
        load=lambda mesh, url: contents.get((mesh, url)),
        key=lambda content: content.encode() or None,
    ))
    return out, warmed


def test_the_same_content_from_two_cities_is_read_once(monkeypatch, tmp_path):
    """A mesh on a city border is published by every city covering it, and in
    some datasets each copy is the whole mesh: rendering both draws it twice."""
    out, _ = _distinct(
        monkeypatch, tmp_path,
        {"53393599": ["from-ota", "from-shinagawa"], "53393590": ["west"]},
        {("53393599", "from-ota"): "east", ("53393599", "from-shinagawa"): "east",
         ("53393590", "west"): "west"},
    )
    assert out == ["east", "west"]      # catalog order, duplicate dropped


def test_a_file_that_would_not_parse_is_left_out(monkeypatch, tmp_path):
    """A download that failed leaves no cache, and is skipped rather than
    retried here — the next request is the second attempt."""
    out, _ = _distinct(
        monkeypatch, tmp_path,
        {"53393599": ["good", "broken"]},
        {("53393599", "good"): "kept", ("53393599", "broken"): "never read"},
        fails={("53393599", "broken")},
    )
    assert out == ["kept"]


def test_only_the_uncached_files_go_to_the_parse_workers(monkeypatch, tmp_path):
    """The pool exists to pay the parse once; a cached mesh is already paid."""
    _, warmed = _distinct(
        monkeypatch, tmp_path,
        {"53393599": ["cached", "fresh"]},
        {("53393599", "cached"): "a", ("53393599", "fresh"): "b"},
        cached={("53393599", "cached")},
    )
    assert warmed == [("53393599", "fresh")]


def test_a_file_holding_nothing_does_not_stand_in_for_the_next_one(
        monkeypatch, tmp_path):
    """An empty parse is not content, so it must not claim a digest — two of
    them in a row would otherwise hide whatever came after."""
    out, _ = _distinct(
        monkeypatch, tmp_path,
        {"53393599": ["empty", "also-empty", "real"]},
        {("53393599", "empty"): "", ("53393599", "also-empty"): "",
         ("53393599", "real"): "buildings"},
    )
    assert out == ["buildings"]
