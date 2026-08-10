"""What the service accepts from outside: how big an upload may be.

Both uploads are read whole into memory, and multipart is spooled to disk on
the way in, so the ceilings are the difference between a service and a way to
fill someone's disk. Two layers are checked here: the body cap that runs before
anything is parsed, and the per-file cap that tells a caller which file it was.
"""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from app.api import routes
from app.config import MAX_GPX_BYTES, MAX_SVG_BYTES
from app.limits import LimitRequestBody
from app.main import app

from conftest import DEG, LAT0, LON0


@pytest.fixture
def client():
    return TestClient(app)


# ---- the body cap, ahead of the parser -------------------------------------

def run_middleware(limit, headers, chunks):
    """Drive `LimitRequestBody` over a fake request; returns (sent, reached)."""
    reached = []

    async def inner(scope, receive, send):
        reached.append(True)
        while True:                                   # drain, as a parser would
            message = await receive()
            if message["type"] != "http.request" or not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"built the model"})

    queue = [{"type": "http.request", "body": c, "more_body": i < len(chunks) - 1}
             for i, c in enumerate(chunks)]
    sent = []

    async def receive():
        return queue.pop(0)

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "headers": headers}
    asyncio.run(LimitRequestBody(inner, limit)(scope, receive, send))
    return sent, bool(reached)


def status_of(sent):
    return next(m["status"] for m in sent if m["type"] == "http.response.start")


def body_of(sent):
    return b"".join(m.get("body", b"") for m in sent
                    if m["type"] == "http.response.body")


def test_a_body_that_declares_itself_too_big_is_refused_unread():
    """The point of the whole thing: nothing downstream is allowed to run, so
    no part of the upload is ever spooled to disk."""
    sent, reached = run_middleware(100, [(b"content-length", b"999999")], [b"x"])
    assert status_of(sent) == 413
    assert not reached
    assert "上限" in json.loads(body_of(sent))["detail"]


def test_a_body_with_no_declared_length_is_counted_as_it_arrives():
    """Chunked uploads carry no Content-Length, so the header check cannot see
    them; they are cut off at the same ceiling instead."""
    chunks = [b"x" * 60, b"x" * 60]
    sent, _ = run_middleware(100, [], chunks)
    assert status_of(sent) == 413


def test_what_the_app_said_before_the_cut_off_does_not_escape():
    """Once over the ceiling the answer is 413 and nothing else — a half-built
    response must not reach the client alongside it."""
    sent, _ = run_middleware(100, [], [b"x" * 200])
    assert [m["type"] for m in sent] == ["http.response.start", "http.response.body"]
    assert b"built the model" not in body_of(sent)


def test_a_body_inside_the_ceiling_is_passed_straight_through():
    sent, reached = run_middleware(1000, [(b"content-length", b"12")], [b"x" * 12])
    assert status_of(sent) == 200
    assert reached
    assert body_of(sent) == b"built the model"


# ---- the per-file caps ------------------------------------------------------

def test_an_oversized_gpx_is_refused_by_name(client, monkeypatch, offline_uploads):
    monkeypatch.setattr(routes, "MAX_GPX_BYTES", 512)
    resp = client.post(
        "/api/generate",
        files={"file": ("track.gpx", b"<gpx>" + b" " * 600 + b"</gpx>",
                        "application/gpx+xml")},
        data={"fmt": "3mf"},
    )
    assert resp.status_code == 413
    assert "GPX" in resp.json()["detail"]


def test_an_oversized_nameplate_is_refused_by_name(client, monkeypatch,
                                                   offline_uploads, make_gpx):
    monkeypatch.setattr(routes, "MAX_SVG_BYTES", 512)
    gpx = make_gpx([(LAT0 + 0.3 * DEG, LON0 + 0.3 * DEG),
                    (LAT0 + 0.6 * DEG, LON0 + 0.6 * DEG)])
    resp = client.post(
        "/api/generate",
        files={"file": ("track.gpx", gpx, "application/gpx+xml"),
               "plate_svg": ("plate.svg", b"<svg>" + b" " * 600 + b"</svg>",
                             "image/svg+xml")},
        data={"fmt": "3mf", "plate_center": f"{LON0 + 0.5 * DEG},{LAT0 + 0.5 * DEG}"},
    )
    assert resp.status_code == 413
    assert "銘板" in resp.json()["detail"]


def test_the_shipped_ceilings_leave_room_for_real_files():
    """A long outing and a drawn nameplate both have to fit, or the limit is
    just a bug with a message."""
    assert MAX_GPX_BYTES >= 10 * 1024 * 1024
    assert MAX_SVG_BYTES >= 1024 * 1024


@pytest.fixture
def offline_uploads(monkeypatch, hill_grid):
    """The DEM/land-use fetchers patched out; these tests never get that far."""
    monkeypatch.setattr(routes, "fetch_elevation_grid",
                        lambda bbox, zoom, grid_max: hill_grid)
    monkeypatch.setattr(routes, "category_grid", lambda grid: None)


# ---- the cap under a real server --------------------------------------------

def capped_app(limit):
    """The app's middleware shape in miniature: the cap, wrapped in CORS."""
    tiny = FastAPI()
    tiny.add_middleware(LimitRequestBody, limit=limit)
    tiny.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                        allow_headers=["*"])

    @tiny.post("/up")
    def up(f: UploadFile = File(...)) -> dict:
        return {"got": len(f.file.read())}

    return tiny


def test_a_real_multipart_upload_over_the_cap_comes_back_413():
    """Through Starlette's own parser, which is what has to be stopped before
    it spools the part to a temporary file."""
    with TestClient(capped_app(2048)) as c:
        resp = c.post("/up", files={"f": ("big.gpx", b"x" * 8192)})
    assert resp.status_code == 413
    assert "上限" in resp.json()["detail"]


def test_the_refusal_carries_its_cors_headers():
    """CORS sits outside the cap on purpose: without these a browser reports a
    413 as a CORS failure and the reason never reaches the user."""
    with TestClient(capped_app(2048)) as c:
        resp = c.post("/up", files={"f": ("big.gpx", b"x" * 8192)},
                      headers={"Origin": "http://localhost:5173"})
    assert resp.status_code == 413
    assert resp.headers.get("access-control-allow-origin") == "*"


def test_an_upload_under_the_cap_still_reaches_the_route():
    with TestClient(capped_app(65536)) as c:
        resp = c.post("/up", files={"f": ("small.gpx", b"x" * 1000)})
    assert resp.status_code == 200
    assert resp.json() == {"got": 1000}
