import json
import sys
import time
import uuid
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import jobs
from app.main import app

FIXTURE = Path(__file__).with_name("worker_fixture.py")
GPX = b'<gpx><trk><trkseg><trkpt lat="35.003" lon="139.003"/><trkpt lat="35.007" lon="139.007"/></trkseg></trk></gpx>'


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("PARSE_PROCS", "1")
    # Needed by the subprocess in local runs as well as the Docker test stage.
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1]))
    manager = jobs.JobManager(tmp_path / "jobs", timeout=20, ttl=60,
                              command=[sys.executable, str(FIXTURE)])
    monkeypatch.setattr(jobs, "manager", manager)
    return manager


def wait(manager, key, terminal=True):
    end = time.monotonic() + 25
    while time.monotonic() < end:
        state = manager.snapshot(key)
        if (state["status"] in jobs.TERMINAL) if terminal else (state["status"] == "running"):
            return state
        time.sleep(0.03)
    raise AssertionError("job did not advance")


def test_submit_poll_and_download_all_formats_without_rebuilding(manager):
    with TestClient(app) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        key = uuid.uuid4().hex
        response = client.post("/api/generate", files={"file": ("route.gpx", GPX)},
                               headers={"Idempotency-Key": key})
        assert response.status_code == 202
        assert response.headers["location"] == f"/api/jobs/{key}"
        repeat = client.post("/api/generate", files={"file": ("route.gpx", GPX)},
                             headers={"Idempotency-Key": key})
        assert repeat.json()["id"] == key
        assert len(manager.jobs) == 1
        state = wait(manager, key)
        assert state["status"] == "succeeded", state
        assert state["warnings"] == [{"source": "buildings", "reason": "fetch_failed"}]
        assert not (manager.root / key / "input.gpx").exists()
        assert not (manager.root / key / "params.json").exists()
        for fmt, (content_type, ext) in jobs.FORMATS.items():
            response = client.get(f"/api/jobs/{key}/files/{fmt}")
            assert response.status_code == 200
            assert response.headers["content-type"] == content_type
            assert int(response.headers["content-length"]) == len(response.content)
            assert response.headers["cache-control"] == "private, no-store"
            assert response.content == (manager.root / key / f"model.{ext}").read_bytes()
        assert client.get(f"/api/jobs/{key}/files/obj").status_code == 400
        assert client.get("/api/jobs/not-a-job").status_code == 404
        client.delete(f"/api/jobs/{key}")
        assert not (manager.root / key).exists()


def test_deadline_kills_worker_and_descendants_and_cleans_inputs(manager):
    manager.timeout = 1.5
    with TestClient(app):
        key = manager.submit({}, b"sleep-child", b"private svg")["id"]
        state = wait(manager, key)
        assert state["status"] == "failed"
        assert "上限" in state["detail"]
        directory = manager.root / key
        child = int((directory / "child.pid").read_text())
        stat = Path(f"/proc/{child}/stat")
        if stat.exists():
            assert stat.read_text().split()[2] == "Z"  # exited, awaiting init's reap
        assert not (directory / "input.gpx").exists()
        assert not (directory / "input.svg").exists()


def test_cancel_queued_and_running_jobs(manager):
    with TestClient(app) as client:
        first = manager.submit({}, b"sleep", b"")["id"]
        wait(manager, first, terminal=False)
        second = manager.submit({}, b"sleep", b"")["id"]
        for key in (second, first):
            assert client.delete(f"/api/jobs/{key}").status_code == 204
            assert manager.snapshot(key)["status"] == "cancelled"
            assert not (manager.root / key / "input.gpx").exists()
        assert not manager.running


def test_capacity_expiry_and_process_failure(manager):
    manager.capacity = 1
    with TestClient(app):
        key = manager.submit({}, b"crash", b"")["id"]
        with pytest.raises(HTTPException) as exc:
            manager.submit({}, b"sleep", b"")
        assert exc.value.status_code == 429
        assert wait(manager, key)["status"] == "failed"
        manager.jobs[key]["expires_at"] = time.time() - 1
        with pytest.raises(HTTPException):
            manager.snapshot(key)
        assert not (manager.root / key).exists()


def test_restart_marks_unfinished_jobs_failed_and_removes_uploads(manager):
    key = uuid.uuid4().hex
    directory = manager.root / key
    directory.mkdir(parents=True)
    (directory / "input.gpx").write_bytes(b"private")
    (directory / "state.json").write_text(json.dumps({"id": key, "status": "running"}))
    with TestClient(app):
        assert manager.snapshot(key)["status"] == "failed"
        assert not (directory / "input.gpx").exists()


def test_a_second_supervisor_cannot_share_the_same_directory(manager):
    with TestClient(app):
        other = jobs.JobManager(manager.root)
        with pytest.raises(RuntimeError, match="one worker"):
            other.start()


def test_unhealthy_supervisor_is_reported(manager):
    with TestClient(app) as client:
        manager.stop.set()
        assert client.get("/api/health").status_code == 503


def test_supervisor_failure_stops_running_calculation(manager, monkeypatch):
    with TestClient(app) as client:
        key = manager.submit({}, b"sleep", b"")["id"]
        wait(manager, key, terminal=False)
        process = manager.running[key][0]

        def broken_tick():
            raise OSError("simulated supervisor failure")

        monkeypatch.setattr(manager, "_tick", broken_tick)
        assert wait(manager, key)["status"] == "failed"
        assert process.poll() is not None
        assert not (manager.root / key / "input.gpx").exists()
        assert client.get("/api/health").status_code == 503


def test_cancel_before_upload_arrives_prevents_orphaned_work(manager):
    with TestClient(app) as client:
        key = uuid.uuid4().hex
        assert client.delete(f"/api/jobs/{key}").status_code == 204
        response = client.post("/api/generate", files={"file": ("route.gpx", GPX)},
                               headers={"Idempotency-Key": key})
        assert response.status_code == 409
        assert not (manager.root / key).exists()


def test_cancelled_jobs_do_not_exhaust_capacity(manager):
    manager.capacity = 1
    with TestClient(app):
        old = manager.submit({}, b"sleep", b"")["id"]
        manager.cancel(old)
        new = manager.submit({}, b"sleep", b"")["id"]
        assert new != old
        assert not (manager.root / old).exists()


def test_queued_job_deadline_does_not_wait_for_a_worker_slot(manager):
    with TestClient(app):
        running = manager.submit({}, b"sleep", b"")["id"]
        wait(manager, running, terminal=False)
        queued = manager.submit({}, b"sleep", b"")["id"]
        with manager.lock:
            manager.jobs[queued]["created_at"] = time.time() - manager.timeout - 1
            manager._tick()
        assert manager.snapshot(queued)["status"] == "failed"
        assert manager.snapshot(running)["status"] == "running"
