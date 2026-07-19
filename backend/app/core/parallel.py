"""Worker pools shared by the data fetchers.

`thread_map` fans small I/O-bound jobs (tile / catalog HTTP GETs) over a
short-lived thread pool. `process_map` fans heavy CPU-bound jobs (CityGML
download + stream-parse + triangulation) over a persistent process pool so
parsing uses multiple cores; workers are spawned (not forked — forking under
uvicorn's threads is unsafe) and kept alive across requests, so their
interpreter/import startup cost is paid once. Both preserve job order, and
`process_map` degrades to in-process execution when the pool cannot be used,
so parallelism is strictly an optimization.

`PARSE_PROCS` (env) overrides the parse-worker count; 1 disables the pool.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Callable, Iterable

PARSE_PROCS = max(1, int(os.environ.get("PARSE_PROCS", min(os.cpu_count() or 1, 6))))

_pool: ProcessPoolExecutor | None = None
_pool_guard = threading.Lock()


def thread_map(fn: Callable, jobs: Iterable[tuple], workers: int) -> list:
    """`[fn(*job) for job in jobs]`, jobs running on up to `workers` threads."""
    jobs = list(jobs)
    if len(jobs) <= 1:
        return [fn(*job) for job in jobs]
    ex = ThreadPoolExecutor(max_workers=min(workers, len(jobs)))
    try:
        return list(ex.map(lambda job: fn(*job), jobs))
    finally:
        ex.shutdown(wait=True, cancel_futures=True)


def _call(item: tuple[Callable, tuple]):
    fn, job = item
    return fn(*job)


def _get_pool() -> ProcessPoolExecutor:
    global _pool
    with _pool_guard:
        if _pool is None:
            _pool = ProcessPoolExecutor(
                max_workers=PARSE_PROCS, mp_context=mp.get_context("spawn")
            )
        return _pool


def process_map(fn: Callable, jobs: Iterable[tuple]) -> list:
    """`[fn(*job) for job in jobs]` on the process pool (in-process fallback).

    `fn` must be a module-level function (pickled by reference). A job's own
    exception propagates exactly as it would in-process; only pool
    *infrastructure* failures (spawn refused, worker killed) fall back to
    sequential execution.
    """
    jobs = list(jobs)
    if not jobs:
        return []
    if PARSE_PROCS > 1 and len(jobs) > 1:
        global _pool
        try:
            pool = _get_pool()
        except OSError:
            return [fn(*job) for job in jobs]
        try:
            return list(pool.map(_call, [(fn, job) for job in jobs]))
        except BrokenProcessPool:
            with _pool_guard:
                if _pool is not None:
                    _pool.shutdown(wait=False, cancel_futures=True)
                    _pool = None
    return [fn(*job) for job in jobs]
