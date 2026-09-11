"""Single-host job supervisor. One API process owns this directory.

Each generation runs in its own process group so deadlines and cancellation
also stop native calculations and PLATEAU parse children. Finished artifacts
are immutable and expire; GPX/SVG inputs are removed as soon as the job ends.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import HTTPException

from . import config
from .core.report import write_json

log = logging.getLogger(__name__)
TERMINAL = {"succeeded", "failed", "cancelled", "expired"}
FORMATS = {"glb": ("model/gltf-binary", "glb"), "3mf": ("model/3mf", "3mf"),
           "stl": ("model/stl", "stl"), "stl_multi": ("application/zip", "zip")}


class JobManager:
    def __init__(self, root: Path = config.JOBS_DIR, *, workers=config.JOB_WORKERS,
                 capacity=config.MAX_JOBS, timeout=config.JOB_TIMEOUT_SECONDS,
                 ttl=config.JOB_TTL_SECONDS, max_bytes=config.MAX_JOB_BYTES,
                 command=None):
        self.root, self.workers, self.capacity = root, workers, capacity
        self.timeout, self.ttl, self.max_bytes = timeout, ttl, max_bytes
        self.command = command or [sys.executable, "-m", "app.job_worker"]
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.jobs: dict[str, dict] = {}
        self.cancelled: dict[str, float] = {}
        self.running: dict[str, tuple[subprocess.Popen, float]] = {}
        self.thread = None
        self.owner = None

    def start(self):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.owner = (self.root / ".owner.lock").open("a")
        try:
            fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.owner.close()
            raise RuntimeError("JOBS_DIR already has an owner; run uvicorn with one worker")
        for directory in self.root.iterdir():
            if not directory.is_dir() or not self.valid_id(directory.name):
                continue
            try:
                state = json.loads((directory / "state.json").read_text())
                self.jobs[directory.name] = state
                if state["status"] not in TERMINAL:
                    self._finish(directory.name, "failed", "サーバーの再起動で生成が中断されました。再生成してください。")
            except (OSError, ValueError, KeyError):
                self.jobs.pop(directory.name, None)
                shutil.rmtree(directory)
        self.stop.clear()
        self.thread = threading.Thread(target=self._loop, name="generation-supervisor", daemon=True)
        self.thread.start()

    @staticmethod
    def valid_id(value):
        try:
            return uuid.UUID(value).hex == value
        except (ValueError, AttributeError):
            return False

    def healthy(self):
        return self.thread is not None and self.thread.is_alive() and not self.stop.is_set()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=10)
        with self.lock:
            for key, (process, _) in list(self.running.items()):
                self._kill(process)
                self._finish(key, "failed", "サーバーの停止で生成が中断されました。")
            self.running.clear()
            for key, state in list(self.jobs.items()):
                if state["status"] == "queued":
                    self._finish(key, "cancelled")
        if self.owner:
            self.owner.close()

    def _save(self, key):
        write_json(self.root / key / "state.json", self.jobs[key])

    def _cleanup(self):
        self.cancelled = {key: until for key, until in self.cancelled.items() if until > time.time()}
        for key, state in list(self.jobs.items()):
            if state["status"] in TERMINAL and time.time() >= state["expires_at"]:
                shutil.rmtree(self.root / key, ignore_errors=True)
                del self.jobs[key]

    def submit(self, params, gpx: bytes, svg: bytes, key: str | None = None):
        key = key or uuid.uuid4().hex
        if not self.valid_id(key):
            raise HTTPException(400, "Idempotency-Key must be a UUID without hyphens")
        with self.lock:
            if not self.healthy():
                raise HTTPException(503, "生成サービスを起動中です。")
            self._cleanup()
            if key in self.cancelled:
                raise HTTPException(409, "この生成はキャンセルされました。")
            if key in self.jobs:
                return self.snapshot(key)
            # Failed/cancelled jobs hold no useful artifact. Reclaim their
            # metadata before refusing a new submission, so cancelling a few
            # jobs cannot occupy every generation slot for the full TTL.
            for old_key, state in list(self.jobs.items()):
                if len(self.jobs) < self.capacity:
                    break
                if state["status"] in {"failed", "cancelled"}:
                    shutil.rmtree(self.root / old_key, ignore_errors=True)
                    del self.jobs[old_key]
            if len(self.jobs) >= self.capacity:
                raise HTTPException(429, "生成・保存枠が満杯です。不要な結果を削除するか、しばらく待ってください。",
                                    headers={"Retry-After": "60"})
            directory = self.root / key
            directory.mkdir(mode=0o700)
            try:
                (directory / "input.gpx").write_bytes(gpx)
                if svg:
                    (directory / "input.svg").write_bytes(svg)
                write_json(directory / "params.json", params)
                self.jobs[key] = {"id": key, "status": "queued", "stage": "queued",
                                  "created_at": time.time(), "expires_at": None, "warnings": []}
                self._save(key)
            except Exception:
                self.jobs.pop(key, None)
                shutil.rmtree(directory, ignore_errors=True)
                raise
            return self.snapshot(key)

    def snapshot(self, key):
        with self.lock:
            self._cleanup()
            if key not in self.jobs:
                raise HTTPException(404, "生成結果が見つからないか、保存期限が切れています。再生成してください。")
            state = dict(self.jobs[key])
            directory = self.root / key
            if state["status"] == "running":
                try:
                    state.update(json.loads((directory / "progress.json").read_text()))
                except (OSError, ValueError):
                    pass
            warnings = []
            for path in sorted((directory / "warnings").glob("*.json")):
                try:
                    warnings.append(json.loads(path.read_text()))
                except (OSError, ValueError):
                    pass
            state["warnings"] = warnings
            return state

    def cancel(self, key):
        with self.lock:
            if not self.valid_id(key):
                raise HTTPException(404, "job not found")
            self._cleanup()
            # A browser can cancel while its upload is still being parsed.
            # Remember that id so a late submission cannot start orphaned work.
            if len(self.cancelled) >= self.capacity * 4:
                del self.cancelled[min(self.cancelled, key=self.cancelled.get)]
            self.cancelled[key] = time.time() + self.timeout
            if key not in self.jobs:
                return
            if key in self.running:
                self._kill(self.running.pop(key)[0])
            if self.jobs[key]["status"] not in TERMINAL:
                self._finish(key, "cancelled")
            else:
                # DELETE also releases completed artifact storage immediately.
                shutil.rmtree(self.root / key, ignore_errors=True)
                del self.jobs[key]

    def open_artifact(self, key, fmt):
        if fmt not in FORMATS:
            raise HTTPException(400, "unsupported format")
        with self.lock:
            if self.snapshot(key)["status"] != "succeeded":
                raise HTTPException(409, "生成が完了していません。")
            try:
                return (self.root / key / f"model.{FORMATS[fmt][1]}").open("rb")
            except FileNotFoundError:
                raise HTTPException(410, "生成結果の保存期限が切れています。再生成してください。")

    def _finish(self, key, status, detail=None):
        state = self.jobs[key]
        state.update(status=status, stage=status, expires_at=time.time() + self.ttl)
        if detail:
            state["detail"] = detail
        directory = self.root / key
        for name in ("input.gpx", "input.svg", "params.json"):
            (directory / name).unlink(missing_ok=True)
        if status != "succeeded":
            for path in directory.glob("model.*"):
                path.unlink(missing_ok=True)
        self._save(key)
        log.info("generation finished status=%s", status)

    @staticmethod
    def _kill(process):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)

    def _tick(self):
        with self.lock:
            self._cleanup()
            for key, (process, started) in list(self.running.items()):
                directory = self.root / key
                size = sum(p.stat().st_size for p in directory.glob("model.*"))
                if time.monotonic() - started >= self.timeout or size > self.max_bytes:
                    self._kill(process)
                    self._finish(key, "failed", "生成時間またはファイル容量の上限を超えました。範囲や解像度を小さくしてください。")
                    del self.running[key]
                elif process.poll() is not None:
                    self._kill(process)  # also reap any surviving parse descendants
                    try:
                        outcome = json.loads((directory / "outcome.json").read_text())
                    except (OSError, ValueError):
                        outcome = {"status": "failed", "detail": "生成処理が停止しました。範囲や解像度を小さくして再試行してください。"}
                    complete = all((directory / f"model.{ext}").is_file() for _, ext in FORMATS.values())
                    if outcome.get("status") == "succeeded" and (process.returncode != 0 or not complete):
                        outcome = {"status": "failed", "detail": "生成ファイルを保存できませんでした。"}
                    self._finish(key, outcome["status"], outcome.get("detail"))
                    del self.running[key]
            for key, state in self.jobs.items():
                if state["status"] != "queued":
                    continue
                if time.time() - state["created_at"] >= self.timeout:
                    self._finish(key, "failed", "待ち時間の上限を超えました。時間をおいて再試行してください。")
                    continue
                if len(self.running) >= self.workers:
                    continue
                directory = self.root / key
                env = {**os.environ, "JOB_REPORT_DIR": str(directory),
                       "MAX_JOB_BYTES": str(self.max_bytes)}
                try:
                    process = subprocess.Popen([*self.command, str(directory)], env=env,
                                               start_new_session=True, stdin=subprocess.DEVNULL)
                except OSError:
                    self._finish(key, "failed", "生成処理を開始できませんでした。")
                    continue
                self.running[key] = (process, time.monotonic() - (time.time() - state["created_at"]))
                state.update(status="running", stage="starting")
                self._save(key)

    def _loop(self):
        while not self.stop.is_set():
            try:
                self._tick()
            except Exception:
                log.exception("generation supervisor failed")
                self.stop.set()  # health becomes unhealthy
                # A failed supervisor must not leave native calculations
                # running with nobody left to enforce their deadline.
                with self.lock:
                    for key, (process, _) in list(self.running.items()):
                        try:
                            self._kill(process)
                            del self.running[key]
                            self._finish(key, "failed", "生成サービスが停止しました。再試行してください。")
                        except Exception:
                            log.exception("failed to clean up interrupted generation")
            self.stop.wait(0.25)


manager = JobManager()
