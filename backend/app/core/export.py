"""Export labelled bodies to a 3D-printable format.

Each face carries a *label* (a colour-layer name: a land-use category, "track",
"base", ...). Labels across all bodies form a small palette. The heavy geometry
is shared; only the serializer differs:
  - 3mf : one object with <basematerials> palette + per-triangle material ref
          (so slicers map colour -> filament). Written directly (trimesh's 3MF
          export does not preserve per-face colour).
  - glb : per-face vertex colours (preview).
  - stl : geometry only (no colour).
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass

import numpy as np
import trimesh

# Default colour per label (hex). User overrides merge over this.
DEFAULT_COLORS: dict[str, str] = {
    "water": "#4a80c0",
    "forest": "#3f7d3a",
    "field": "#c9d17a",
    "urban": "#b0b0b0",
    "bare": "#cdbb8f",
    "other": "#9a8f80",
    "terrain": "#c2b280",
    "base": "#8a7f6f",
    "track": "#dc4628",
    "roof": "#b5651d",
    "wall": "#e6ddcb",
}

_FORMATS = {
    "stl": ("model/stl", "stl"),
    "stl_multi": ("application/zip", "zip"),
    "3mf": ("model/3mf", "3mf"),
    "glb": ("model/gltf-binary", "glb"),
}


@dataclass
class Body:
    mesh: trimesh.Trimesh
    labels: np.ndarray | str  # per-face label array, or a single label for all faces

    def face_labels(self) -> np.ndarray:
        if isinstance(self.labels, str):
            return np.full(len(self.mesh.faces), self.labels, dtype="<U8")
        return self.labels


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    h = value.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def export_bodies(
    bodies: list[Body], fmt: str, colors: dict[str, str] | None = None
) -> tuple[bytes, str, str]:
    if fmt not in _FORMATS:
        raise ValueError(f"unsupported format: {fmt}")
    content_type, ext = _FORMATS[fmt]
    color_map = {**DEFAULT_COLORS, **(colors or {})}

    # Global palette over all labels actually used.
    used = sorted({lbl for b in bodies for lbl in np.unique(b.face_labels())})
    palette = [(lbl, _hex_to_rgb(color_map.get(lbl, "#999999"))) for lbl in used]
    index_of = {lbl: i for i, (lbl, _) in enumerate(palette)}

    if fmt == "stl":
        merged = trimesh.util.concatenate([b.mesh for b in bodies])
        data = merged.export(file_type="stl")
    elif fmt == "stl_multi":
        data = _write_stl_multi(bodies, used, color_map)
    elif fmt == "glb":
        scene = trimesh.Scene()
        for i, b in enumerate(bodies):
            m = b.mesh.copy()
            m.unmerge_vertices()  # crisp per-face colors (no boundary averaging)
            rgb = np.array([palette[index_of[l]][1] for l in b.face_labels()], dtype=np.uint8)
            m.visual.face_colors = np.column_stack([rgb, np.full(len(rgb), 255, np.uint8)])
            scene.add_geometry(m, geom_name=f"body{i}")
        data = scene.export(file_type="glb")
    else:  # 3mf
        data = _write_3mf(bodies, palette, index_of)

    if isinstance(data, str):
        data = data.encode()
    return data, content_type, ext


def _write_stl_multi(bodies, used, color_map) -> bytes:
    """One STL per colour label, bundled in a zip (STL can't carry colour).

    Every STL shares the same coordinate space, so loading them all into a
    slicer as a single multi-part object lines them up exactly; the slicer
    slices their union (the original solid) and a filament is assigned per
    part. A README maps each file to its intended colour.
    """
    buf = io.BytesIO()
    lines = ["3d-footprint multi-colour STL set", "",
             "Load every .stl into your slicer as ONE object's parts (they",
             "share the same origin and line up). Assign a filament per part:",
             ""]
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for label in used:
            parts = []
            for b in bodies:
                idx = np.nonzero(b.face_labels() == label)[0]
                if len(idx):
                    parts.append(b.mesh.submesh([idx], append=True, repair=False))
            if not parts:
                continue
            mesh = parts[0] if len(parts) == 1 else trimesh.util.concatenate(parts)
            z.writestr(f"{label}.stl", mesh.export(file_type="stl"))
            lines.append(f"  {label}.stl  -> {color_map.get(label, '#999999')}")
        z.writestr("README.txt", "\n".join(lines) + "\n")
    return buf.getvalue()


def _write_3mf(bodies, palette, index_of) -> bytes:
    """3MF with a basematerials palette and one watertight object per body.

    Each triangle references its material via pid/p1 (per-face colour). Keeping
    bodies as separate objects preserves watertightness; slicers union the
    overlapping parts and map each material colour to a filament.
    """
    bases = "".join(
        f'<base name="{name}" displaycolor="#{r:02X}{g:02X}{b:02X}FF"/>'
        for name, (r, g, b) in palette
    )

    objects: list[str] = []
    items: list[str] = []
    for i, body in enumerate(bodies):
        oid = i + 2  # id 1 is the basematerials group
        verts = "".join(
            f'<vertex x="{x:.6f}" y="{y:.6f}" z="{z:.6f}"/>'
            for x, y, z in body.mesh.vertices
        )
        lbl = body.face_labels()
        tris = "".join(
            f'<triangle v1="{a}" v2="{b}" v3="{c}" pid="1" p1="{index_of[l]}"/>'
            for (a, b, c), l in zip(body.mesh.faces, lbl)
        )
        objects.append(
            f'<object id="{oid}" type="model"><mesh>'
            f"<vertices>{verts}</vertices><triangles>{tris}</triangles>"
            "</mesh></object>"
        )
        items.append(f'<item objectid="{oid}"/>')

    model = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<model unit="millimeter" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
        "<resources>"
        f'<basematerials id="1">{bases}</basematerials>'
        f'{"".join(objects)}'
        "</resources>"
        f'<build>{"".join(items)}</build>'
        "</model>"
    )

    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
        'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
        "</Relationships>"
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("3D/3dmodel.model", model)
    return buf.getvalue()
