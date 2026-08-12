"""Serialization: the 3MF palette, the STL set, and the attribution that has
to survive into every one of them.

The 出典 is not optional — the data licenses require it — so each format's
attribution slot is asserted here rather than left to the UI.
"""
from __future__ import annotations

import io
import json
import struct
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pytest
import trimesh

from app.core.export import (
    DEFAULT_COLORS, Body, _XML_ROWS_PER_PASS, _part_names, _rows_to_xml,
    export_bodies,
)

_3MF = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
CREDIT = "出典: 国土地理院「地理院タイル（標高タイル）」 を加工して作成"
ASCII_CREDIT = "Source: GSI (modified) - 3d-footprint"


def cube(label, size=1.0, at=(0.0, 0.0, 0.0)) -> Body:
    mesh = trimesh.creation.box(extents=(size, size, size))
    mesh.apply_translation(at)
    return Body(mesh, label)


def two_tone_cube() -> Body:
    """One body carrying two colour layers, like the nameplate's tile + ink."""
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    labels = np.array(["plate"] * len(mesh.faces), dtype="<U8")
    labels[: len(labels) // 2] = "label"
    return Body(mesh, labels)


def read_3mf(data: bytes) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(data))


# ---- palette ---------------------------------------------------------------

def test_every_label_the_pipeline_emits_has_a_colour():
    """A missing entry would silently export as the #999999 fallback."""
    from app.core.coloring import _PALETTE
    for label in (*_PALETTE, "track", "building", "plate", "label"):
        assert label in DEFAULT_COLORS
        assert DEFAULT_COLORS[label].startswith("#")
        assert len(DEFAULT_COLORS[label]) == 7


def test_an_unknown_format_is_rejected():
    with pytest.raises(ValueError, match="unsupported format"):
        export_bodies([cube("terrain")], "obj")


# ---- part names ------------------------------------------------------------

def test_a_body_is_named_for_its_colour_layer():
    assert _part_names([cube("forest"), cube("water")]) == ["forest", "water"]


def test_opaque_internal_names_get_a_readable_one():
    assert _part_names([cube("bare"), cube("other"), cube("plate")]) == [
        "bare-ground", "other-landuse", "nameplate-base"]


def test_a_repeated_layer_is_numbered_so_names_stay_unique():
    """Buildings and bridges share the structure layer; a slicer's object list
    must still tell them apart."""
    assert _part_names([cube("building"), cube("building")]) == [
        "building", "building (2)"]


def test_a_multi_layer_body_is_named_for_the_whole():
    assert _part_names([two_tone_cube()]) == ["nameplate"]


# ---- 3MF -------------------------------------------------------------------

def test_3mf_is_a_valid_package():
    data, ctype, ext = export_bodies([cube("forest")], "3mf")
    assert (ctype, ext) == ("model/3mf", "3mf")
    names = set(read_3mf(data).namelist())
    assert {"[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model",
            "Metadata/Slic3r_PE_model.config"} <= names


def test_3mf_palette_covers_the_labels_used():
    data, _, _ = export_bodies([cube("water"), cube("forest")], "3mf")
    root = ET.fromstring(read_3mf(data).read("3D/3dmodel.model"))
    bases = root.findall(".//m:basematerials/m:base", _3MF)
    assert [b.get("name") for b in bases] == ["forest", "water"]   # sorted
    colour = {b.get("name"): b.get("displaycolor")[:7].lower() for b in bases}
    assert colour["water"] == DEFAULT_COLORS["water"]


def test_3mf_triangles_reference_their_own_material():
    """Per-face colour is the whole reason 3MF is written by hand here."""
    data, _, _ = export_bodies([two_tone_cube()], "3mf")
    root = ET.fromstring(read_3mf(data).read("3D/3dmodel.model"))
    names = [b.get("name") for b in root.findall(".//m:basematerials/m:base", _3MF)]
    tris = root.findall(".//m:object/m:mesh/m:triangles/m:triangle", _3MF)
    used = {names[int(t.get("p1"))] for t in tris}
    assert used == {"label", "plate"}
    assert {t.get("pid") for t in tris} == {"1"}


def test_3mf_names_every_object_in_both_dialects():
    """Cura reads the core attribute, the PrusaSlicer family reads the config;
    without one of them the object list is just "Object 1..N"."""
    data, _, _ = export_bodies([cube("forest"), cube("water")], "3mf")
    z = read_3mf(data)
    root = ET.fromstring(z.read("3D/3dmodel.model"))
    assert [o.get("name") for o in root.findall(".//m:object", _3MF)] == [
        "forest", "water"]
    config = ET.fromstring(z.read("Metadata/Slic3r_PE_model.config"))
    assert [m.get("value") for m in config.iterfind(
        "./object/metadata[@key='name']")] == ["forest", "water"]


def test_3mf_carries_the_credit_in_its_metadata():
    data, _, _ = export_bodies([cube("terrain")], "3mf", credit_full=CREDIT)
    root = ET.fromstring(read_3mf(data).read("3D/3dmodel.model"))
    meta = {m.get("name"): m.text for m in root.findall("m:metadata", _3MF)}
    assert meta["Copyright"] == CREDIT
    assert meta["Description"] == CREDIT


def test_3mf_keeps_one_object_per_body():
    """Merging them would destroy the per-body watertightness."""
    data, _, _ = export_bodies([cube("forest"), cube("water")], "3mf")
    root = ET.fromstring(read_3mf(data).read("3D/3dmodel.model"))
    assert len(root.findall(".//m:object", _3MF)) == 2
    assert len(root.findall(".//m:build/m:item", _3MF)) == 2


# ---- STL -------------------------------------------------------------------

def test_stl_puts_the_credit_in_its_header():
    data, ctype, ext = export_bodies([cube("terrain")], "stl",
                                     credit_ascii=ASCII_CREDIT)
    assert (ctype, ext) == ("model/stl", "stl")
    assert data[:80].rstrip() == ASCII_CREDIT.encode()
    # "solid" as the first word is the ASCII-STL sniff; this must stay binary.
    assert not data.startswith(b"solid")
    assert struct.unpack("<I", data[80:84])[0] == len(cube("terrain").mesh.faces)


def test_stl_without_a_credit_is_still_a_valid_binary_stl():
    data, _, _ = export_bodies([cube("terrain")], "stl")
    assert struct.unpack("<I", data[80:84])[0] == 12


def test_stl_multi_writes_one_file_per_colour():
    data, ctype, ext = export_bodies(
        [cube("water"), cube("forest"), two_tone_cube()], "stl_multi",
        credit_full=CREDIT, credit_ascii=ASCII_CREDIT,
    )
    assert (ctype, ext) == ("application/zip", "zip")
    z = zipfile.ZipFile(io.BytesIO(data))
    assert set(z.namelist()) == {
        "water.stl", "forest.stl", "plate.stl", "label.stl", "README.txt"}
    readme = z.read("README.txt").decode()
    assert CREDIT in readme
    assert DEFAULT_COLORS["water"] in readme
    assert z.read("water.stl")[:80].rstrip() == ASCII_CREDIT.encode()


def test_stl_multi_parts_keep_a_shared_origin():
    """They only line up in the slicer if nothing is re-centred on the way out."""
    data, _, _ = export_bodies(
        [cube("water", at=(0.0, 0.0, 0.0)), cube("forest", at=(10.0, 0.0, 0.0))],
        "stl_multi")
    z = zipfile.ZipFile(io.BytesIO(data))
    load = lambda n: trimesh.load(io.BytesIO(z.read(n)), file_type="stl")
    assert load("water.stl").bounds[0][0] == pytest.approx(-0.5)
    assert load("forest.stl").bounds[0][0] == pytest.approx(9.5)


# ---- GLB (preview) ---------------------------------------------------------

def glb_json(data: bytes) -> dict:
    length = struct.unpack("<I", data[12:16])[0]
    return json.loads(data[20 : 20 + length])


def test_glb_carries_the_credit_in_the_standard_slot():
    data, ctype, ext = export_bodies([cube("terrain")], "glb",
                                     credit_full=CREDIT)
    assert (ctype, ext) == ("model/gltf-binary", "glb")
    assert glb_json(data)["asset"]["copyright"] == CREDIT


def test_glb_writes_normals_for_a_multi_colour_body():
    """The exporter only writes NORMAL when it is already cached, and a glTF
    without it renders unlit — the plaque came out solid black."""
    doc = glb_json(export_bodies([two_tone_cube()], "glb")[0])
    for mesh in doc["meshes"]:
        for prim in mesh["primitives"]:
            assert "NORMAL" in prim["attributes"]


def test_glb_paints_one_colour_per_body():
    doc = glb_json(export_bodies([cube("water"), cube("forest")], "glb")[0])
    assert len(doc["meshes"]) == 2


# ---- bulk XML formatting ---------------------------------------------------

def per_row(rows, template):
    """The obvious way, which is what the fast one has to agree with."""
    return "".join(template % tuple(r) for r in rows.tolist())


def test_rows_are_formatted_exactly_as_one_at_a_time_would():
    rows = np.column_stack([np.arange(2000), np.arange(2000) * 3, np.arange(2000),
                            np.arange(2000) % 5]).astype(np.int64)
    tmpl = '<triangle v1="%d" v2="%d" v3="%d" pid="1" p1="%d"/>'
    assert _rows_to_xml(rows, tmpl) == per_row(rows, tmpl)


def test_float_rows_keep_every_digit_they_had():
    rng = np.random.default_rng(0)
    rows = rng.random((1500, 3)) * 200.0
    tmpl = '<vertex x="%.6f" y="%.6f" z="%.6f"/>'
    assert _rows_to_xml(rows, tmpl) == per_row(rows, tmpl)


def test_the_chunk_boundary_does_not_drop_or_repeat_a_row():
    """The rows are formatted a hundred thousand at a time so the repeated
    template stays small; the seam is the one place that could lose one."""
    n = _XML_ROWS_PER_PASS * 2 + 7
    rows = np.column_stack([np.arange(n), np.arange(n)]).astype(np.int64)
    out = _rows_to_xml(rows, "%d:%d;")
    assert out.count(";") == n
    assert out.startswith("0:0;")
    assert out.endswith(f"{n - 1}:{n - 1};")
    assert f"{_XML_ROWS_PER_PASS - 1}:{_XML_ROWS_PER_PASS - 1};" \
           f"{_XML_ROWS_PER_PASS}:{_XML_ROWS_PER_PASS};" in out


def test_no_rows_at_all_is_an_empty_string():
    assert _rows_to_xml(np.empty((0, 3), np.int64), "%d%d%d") == ""
