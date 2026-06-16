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
    "road": "#6f6f6f",
    "bare": "#cdbb8f",
    "other": "#9a8f80",
    "terrain": "#c2b280",
    "base": "#8a7f6f",
    "track": "#dc4628",
    "building": "#b0b0b0",  # buildings + bridges (same structure colour layer)
}

_FORMATS = {
    "stl": ("model/stl", "stl"),
    "stl_multi": ("application/zip", "zip"),
    "3mf": ("model/3mf", "3mf"),
    "glb": ("model/gltf-binary", "glb"),
}

# Preview-only: dihedral threshold (deg) below which adjacent faces share a
# smooth normal. Gentle terrain slopes fall under it (smooth-shaded relief);
# walls, top rims and building corners (~90 deg) stay above it (crisp).
_GLB_CREASE_DEG = 40.0


def _creased_normals(mesh: trimesh.Trimesh, angle_deg: float) -> trimesh.Trimesh:
    """Return a copy whose normals are smoothed only across sub-`angle` edges.

    The terrain solid bakes flat per-face normals (every triangle reads as a
    hard facet) — fine for printing, ugly in the live preview. We re-split the
    mesh by *smooth group*: faces joined through edges gentler than `angle` form
    one group and share averaged normals; sharper edges keep the two sides
    separate (flat). Same idea as three's `toCreasedNormals`, done server-side.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    faces = mesh.faces
    nf = len(faces)
    adj = mesh.face_adjacency
    smooth = mesh.face_adjacency_angles < np.radians(angle_deg)
    e = adj[smooth]
    graph = sp.coo_matrix(
        (np.ones(len(e), bool), (e[:, 0], e[:, 1])), shape=(nf, nf)
    )
    _, group = connected_components(graph, directed=False)

    # Each (original vertex, smooth group) becomes one output vertex.
    fv = faces.reshape(-1)               # 3*nf corner -> original vertex id
    fg = np.repeat(group, 3)             # 3*nf corner -> smooth group id
    key, inv = np.unique(np.stack([fv, fg], 1), axis=0, return_inverse=True)
    inv = inv.ravel()
    out = trimesh.Trimesh(
        vertices=mesh.vertices[key[:, 0]], faces=inv.reshape(-1, 3), process=False
    )
    vn = np.zeros((len(key), 3))
    np.add.at(vn, inv, np.repeat(mesh.face_normals, 3, axis=0))
    norm = np.linalg.norm(vn, axis=1, keepdims=True)
    out.vertex_normals = vn / np.where(norm > 0, norm, 1.0)
    return out


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
            labels = b.face_labels()
            uniq = np.unique(labels)
            if uniq.size == 1:
                # Single-colour body: crease-smooth so the relief shades as a
                # surface, not facets, then paint one uniform colour. Boundaries
                # stay crisp because each colour is its own mesh.
                m = _creased_normals(b.mesh, _GLB_CREASE_DEG)
                rgb = palette[index_of[uniq[0]]][1]
                m.visual.vertex_colors = np.tile(
                    np.array([*rgb, 255], np.uint8), (len(m.vertices), 1)
                )
            else:
                # Multi-colour body: unmerge for crisp per-face colour (flat).
                m = b.mesh.copy()
                m.unmerge_vertices()
                rgb = np.array([palette[index_of[l]][1] for l in labels], np.uint8)
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
