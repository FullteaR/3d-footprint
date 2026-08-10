"""The /api/generate endpoint, end to end — without the network.

The route's two outside inputs are the DEM mosaic and the land-use raster, and
both are plain arrays, so patching the two fetchers runs the whole real
pipeline (projection, terrain, track, 出典 stamp, nameplate, export) on
synthetic ground.
"""
from __future__ import annotations

import io
import struct
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import Point, box

from app.api import routes
from app.main import app

from conftest import DEG, LAT0, LON0, N, make_grid

_3MF = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}

SVG = (b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 40">'
       b'<path d="M10,10 H90 V30 H10 Z" fill="#000"/></svg>')


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def offline(monkeypatch, hill_grid):
    """The pipeline with its two fetchers replaced. Returns a knob for the
    land-use grid so a test can pick plain terrain or two colours."""
    state = {"cats": None}
    monkeypatch.setattr(routes, "fetch_elevation_grid",
                        lambda bbox, zoom, grid_max: hill_grid)
    monkeypatch.setattr(routes, "category_grid", lambda grid: state["cats"])
    return state


@pytest.fixture
def gpx(make_gpx):
    """A short track across the middle of the synthetic grid."""
    lats = np.linspace(LAT0 + 0.3 * DEG, LAT0 + 0.7 * DEG, 8)
    lons = np.linspace(LON0 + 0.2 * DEG, LON0 + 0.8 * DEG, 8)
    return make_gpx(list(zip(lats, lons)))


def post(client, gpx, **form):
    return client.post(
        "/api/generate",
        files={"file": ("track.gpx", gpx, "application/gpx+xml")},
        data={"fmt": "3mf", **form},
    )


def model_xml(data: bytes):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return ET.fromstring(z.read("3D/3dmodel.model"))


def metadata(data: bytes) -> dict:
    root = model_xml(data)
    return {m.get("name"): m.text for m in root.findall("m:metadata", _3MF)}


def object_names(data: bytes) -> list[str]:
    return [o.get("name") for o in model_xml(data).findall(".//m:object", _3MF)]


# ---- health ----------------------------------------------------------------

def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


# ---- the happy paths -------------------------------------------------------

def test_a_plain_track_generates_a_3mf(client, gpx, offline):
    resp = post(client, gpx)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "model/3mf"
    assert "footprint.3mf" in resp.headers["content-disposition"]
    assert set(object_names(resp.content)) == {"terrain", "track"}


@pytest.mark.parametrize("fmt,ctype,ext", [
    ("stl", "model/stl", "stl"),
    ("stl_multi", "application/zip", "zip"),
    ("glb", "model/gltf-binary", "glb"),
    ("3mf", "model/3mf", "3mf"),
])
def test_every_offered_format_comes_back(client, gpx, offline, fmt, ctype, ext):
    resp = post(client, gpx, fmt=fmt)
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == ctype
    assert f"footprint.{ext}" in resp.headers["content-disposition"]
    assert len(resp.content) > 1000


def test_land_use_colours_become_their_own_objects(client, gpx, offline,
                                                   split_cats):
    offline["cats"] = split_cats
    resp = post(client, gpx)
    assert resp.status_code == 200
    assert {"water", "forest"} <= set(object_names(resp.content))


def test_turning_the_track_off_leaves_it_out(client, gpx, offline):
    resp = post(client, gpx, include_track="false")
    assert object_names(resp.content) == ["terrain"]


@pytest.mark.parametrize("shape,rotation", [
    ("rect", 0.0), ("square", 0.0), ("hex", 0.0), ("rect", 30.0), ("hex", 17.0),
])
def test_every_outline_shape_and_angle_generates(client, gpx, offline,
                                                 shape, rotation):
    resp = post(client, gpx, shape=shape, rotation_deg=rotation, fmt="stl")
    assert resp.status_code == 200, resp.text
    assert struct.unpack("<I", resp.content[80:84])[0] > 0


def test_an_explicit_bbox_overrides_the_automatic_extent(client, gpx, offline):
    bbox = f"{LON0},{LAT0},{LON0 + DEG},{LAT0 + DEG}"
    assert post(client, gpx, bbox=bbox).status_code == 200


def test_a_time_range_trims_the_track(client, make_gpx, offline):
    stamped = make_gpx([
        (LAT0 + 0.3 * DEG, LON0 + 0.3 * DEG, "2026-05-01T00:00:00Z"),
        (LAT0 + 0.5 * DEG, LON0 + 0.5 * DEG, "2026-05-01T00:10:00Z"),
        (LAT0 + 0.7 * DEG, LON0 + 0.7 * DEG, "2026-05-01T00:20:00Z"),
    ])
    whole = post(client, stamped, fmt="stl")
    trimmed = post(client, stamped, fmt="stl",
                   time_range="1777593600,1777594260")   # first 11 minutes
    assert whole.status_code == trimmed.status_code == 200
    assert len(trimmed.content) < len(whole.content)


# ---- attribution -----------------------------------------------------------

def test_the_credit_rides_in_the_file(client, gpx, offline):
    """It is not a UI decoration: the licenses require it to travel with the
    model, so every export carries the sentence."""
    meta = metadata(post(client, gpx).content)
    assert meta["Copyright"].startswith("出典: 国土地理院")
    assert meta["Copyright"] == meta["Description"]


def test_the_credit_names_land_use_only_when_it_was_used(client, gpx, offline,
                                                         split_cats):
    plain = metadata(post(client, gpx).content)["Copyright"]
    assert "JAXA" not in plain and "PLATEAU" not in plain

    offline["cats"] = split_cats
    coloured = metadata(post(client, gpx).content)["Copyright"]
    assert "JAXA" in coloured and "PLATEAU" in coloured


def test_the_stl_header_carries_the_credit(client, gpx, offline):
    data = post(client, gpx, fmt="stl").content
    assert data[:80].rstrip().startswith(b"Source: GSI")
    assert not data.startswith(b"solid")


# ---- the nameplate ---------------------------------------------------------

def test_an_uploaded_svg_becomes_a_nameplate(client, gpx, offline):
    resp = client.post(
        "/api/generate",
        files={"file": ("track.gpx", gpx, "application/gpx+xml"),
               "plate_svg": ("plate.svg", SVG, "image/svg+xml")},
        data={"fmt": "3mf",
              "plate_center": f"{LON0 + 0.5 * DEG},{LAT0 + 0.5 * DEG}"},
    )
    assert resp.status_code == 200, resp.text
    assert "nameplate" in object_names(resp.content)


@pytest.mark.parametrize("shape", [{}, {"rotation_deg": "20"}],
                         ids=["plain-rect", "rotated"])
def test_nothing_is_built_on_top_of_the_nameplate(client, gpx, offline, monkeypatch,
                                                  shape):
    """PLATEAU blocks were being extruded over the plaque.

    The map under the plate is routed out into a pocket, so anything left
    standing there rises out of the recess and buries the artwork — and the
    plaque is the one thing on the model that has to stay readable. Both
    structure providers must be handed an outline with the plate cut out.
    """
    seen = {}

    class _Recorder:
        """Stands in for a PLATEAU provider, keeping the outline it was given."""

        def __init__(self, key):
            self.key = key

        def building_body(self, proj, height_scale=1.0, min_feature_mm=0.8, clip=None):
            seen[self.key] = (proj, clip)
            return None

        def bridge_body(self, proj, min_feature_mm=0.8, clip=None):
            seen[self.key] = (proj, clip)
            return None

    monkeypatch.setattr(routes, "PlateauBuildingProvider", lambda: _Recorder("bldg"))
    monkeypatch.setattr(routes, "PlateauBridgeProvider", lambda: _Recorder("brid"))

    plon, plat = LON0 + 0.5 * DEG, LAT0 + 0.5 * DEG
    resp = client.post(
        "/api/generate",
        files={"file": ("track.gpx", gpx, "application/gpx+xml"),
               "plate_svg": ("plate.svg", SVG, "image/svg+xml")},
        data={"fmt": "3mf", "include_buildings": "true",
              "plate_center": f"{plon},{plat}", **shape},
    )
    assert resp.status_code == 200, resp.text
    assert set(seen) == {"bldg", "brid"}
    for proj, clip in seen.values():
        assert clip is not None          # a plain rect gets a real outline now
        def at(lon, lat):
            return Point(float(proj.x_of(lon)), float(proj.y_of(lat)))
        assert not clip.contains(at(plon, plat))                    # the plaque
        assert clip.contains(at(plon, LAT0 + 0.3 * DEG))            # the map


def test_without_a_nameplate_the_structure_outline_is_left_alone(flat_grid, flat_proj):
    """No plate, no hole — a plain rect keeps handing the providers None so
    they fall back to the fetched grid rectangle themselves."""
    assert routes._structure_clip(None, None, flat_proj, flat_grid) is None
    outline = box(0.0, 0.0, 10.0, 10.0)
    assert routes._structure_clip(outline, None, flat_proj, flat_grid) is outline


# ---- the nameplate, continued ----------------------------------------------

def test_a_nameplate_without_a_position_is_a_400(client, gpx, offline):
    resp = client.post(
        "/api/generate",
        files={"file": ("track.gpx", gpx, "application/gpx+xml"),
               "plate_svg": ("plate.svg", SVG, "image/svg+xml")},
        data={"fmt": "3mf"},
    )
    assert resp.status_code == 400
    assert "位置" in resp.json()["detail"]


# ---- bad input -------------------------------------------------------------

def test_a_file_that_is_not_gpx_is_a_400(client, offline):
    resp = post(client, b"hello, I am not XML")
    assert resp.status_code == 400


def test_a_gpx_without_points_is_a_400(client, offline):
    resp = post(client, b'<?xml version="1.0"?><gpx version="1.1"><trk/></gpx>')
    assert resp.status_code == 400
    assert "trkpt" in resp.json()["detail"]


@pytest.mark.parametrize("field,value", [
    ("bbox", "1,2,3"),
    ("bbox", "139.1,35.0,139.0,35.1"),
    ("shape", "triangle"),
    ("time_range", "200,100"),
    ("fmt", "obj"),
])
def test_bad_parameters_are_400_not_500(client, gpx, offline, field, value):
    resp = post(client, gpx, **{field: value})
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"]


def test_a_time_range_on_an_unstamped_gpx_is_a_400(client, gpx, offline):
    resp = post(client, gpx, time_range="0,100")
    assert resp.status_code == 400
    assert "timestamps" in resp.json()["detail"]
