"""Export labelled bodies to a 3D-printable format.

Each face carries a *label* (a colour-layer name: a land-use category, "track",
"base", ...). Labels across all bodies form a small palette. The heavy geometry
is shared; only the serializer differs:
  - 3mf : one named object per body + <basematerials> palette + per-triangle
          material ref (so slicers map colour -> filament). Written directly
          (trimesh's 3MF export does not preserve per-face colour).
  - glb : per-face vertex colours (preview).
  - stl : geometry only (no colour).
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from xml.sax.saxutils import escape

import numpy as np
import trimesh
from trimesh.exchange import gltf

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
    "plate": "#c2b280",     # nameplate tile (routes maps it to the terrain colour)
    "label": "#333333",     # the nameplate artwork raised on it
}

# Slicer-facing part names for the labels whose internal name isn't obvious in
# an object list; every other label is already its own best name.
_PART_NAMES: dict[str, str] = {
    "bare": "bare-ground",
    "other": "other-landuse",
    "plate": "nameplate-base",
    "label": "nameplate-text",
}

# Same, for the bodies that carry more than one colour layer at once.
_MIXED_NAMES: dict[frozenset[str], str] = {
    frozenset({"plate", "label"}): "nameplate",
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

    # Each (original vertex, smooth group) becomes one output vertex. The pair
    # is packed into one integer rather than uniqued as a two-column array:
    # `np.unique(..., axis=0)` lexsorts eight million rows as records, which on
    # a city is most of the whole preview export. With the vertex id as the
    # high digit the packed key sorts identically, so the output is unchanged.
    fv = faces.reshape(-1)               # 3*nf corner -> original vertex id
    fg = np.repeat(group, 3)             # 3*nf corner -> smooth group id
    ngroups = int(group.max()) + 1 if nf else 1
    key, inv = np.unique(fv.astype(np.int64) * ngroups + fg, return_inverse=True)
    inv = inv.ravel()
    out = trimesh.Trimesh(
        vertices=mesh.vertices[key // ngroups], faces=inv.reshape(-1, 3),
        process=False,
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


def _part_names(bodies) -> list[str]:
    """One name per body: what the part is.

    A slicer's object list is the only place the user meets these parts, and
    "Object 3" says nothing about which one it is — so each carries its layer's
    name ("forest"). A body that wants more than one filament (the nameplate:
    tile plus artwork) is named for the whole. Repeats (buildings and bridges
    share the structure layer) get numbered so every name stays unique.
    """
    names: list[str] = []
    seen: dict[str, int] = {}
    for b in bodies:
        uniq = np.unique(b.face_labels())
        if uniq.size == 1:
            name = _PART_NAMES.get(str(uniq[0]), str(uniq[0]))
        else:
            name = _MIXED_NAMES.get(frozenset(str(u) for u in uniq), "mixed")
        seen[name] = seen.get(name, 0) + 1
        names.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return names


def export_bodies(
    bodies: list[Body],
    fmt: str,
    colors: dict[str, str] | None = None,
    credit_full: str | None = None,
    credit_ascii: str | None = None,
) -> tuple[bytes, str, str]:
    """Serialize `bodies` to `fmt`.

    `credit_full` (UTF-8 出典 sentence) and `credit_ascii` (80-byte-safe
    variant) ride along in the file itself — 3MF metadata, the binary STL
    header, the multi-STL zip's README, the glb's asset.copyright — so any
    exported model keeps the attribution its data licenses require.
    """
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
        data = _stl_with_header(merged, credit_ascii)
    elif fmt == "stl_multi":
        data = _write_stl_multi(bodies, used, color_map, credit_full, credit_ascii)
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
                # Assign the (now per-face) normals explicitly: the exporter
                # only writes NORMAL when it is already there, and a glTF
                # without it renders unlit — the plaque came out solid black.
                m.vertex_normals = np.repeat(m.face_normals, 3, axis=0)
            scene.add_geometry(m, geom_name=f"body{i}")
        post = None
        if credit_full:
            # glTF's standard attribution slot; survives re-export by most tools.
            def post(tree, _c=credit_full):
                tree["asset"]["copyright"] = _c
        data = gltf.export_glb(scene, tree_postprocessor=post)
    else:  # 3mf
        data = _write_3mf(bodies, palette, index_of, credit_full)

    if isinstance(data, str):
        data = data.encode()
    return data, content_type, ext


def _stl_with_header(mesh: trimesh.Trimesh, credit_ascii: str | None) -> bytes:
    """Binary STL with the attribution in its 80-byte header.

    The header must not begin with "solid" (that's the ASCII-STL sniff);
    "Source: ..." is safely different.
    """
    data = mesh.export(file_type="stl")
    if not credit_ascii:
        return data
    hdr = credit_ascii.encode("ascii", "ignore")[:80].ljust(80, b" ")
    return hdr + data[80:]


def _write_stl_multi(bodies, used, color_map, credit_full=None, credit_ascii=None) -> bytes:
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
            z.writestr(f"{label}.stl", _stl_with_header(mesh, credit_ascii))
            lines.append(f"  {label}.stl  -> {color_map.get(label, '#999999')}")
        if credit_full:
            lines += ["", credit_full]
        z.writestr("README.txt", "\n".join(lines) + "\n")
    return buf.getvalue()


_XML_ROWS_PER_PASS = 100_000   # bounded so the repeated template stays small


def _rows_to_xml(rows: np.ndarray, template: str) -> str:
    """Format an (n, k) array into one string, `template` applied per row.

    A single `%` over a repeated template does the formatting in C; writing an
    f-string per row runs the interpreter a few million times instead, which
    on a city is several seconds of a download. Chunked only so the repeated
    template does not itself grow to hundreds of megabytes. The bytes produced
    are identical either way.
    """
    n, k = rows.shape
    flat = rows.ravel().tolist()
    if n <= _XML_ROWS_PER_PASS:
        return (template * n) % tuple(flat)
    out = []
    for i in range(0, n, _XML_ROWS_PER_PASS):
        part = flat[i * k:(i + _XML_ROWS_PER_PASS) * k]
        out.append((template * (len(part) // k)) % tuple(part))
    return "".join(out)


def _write_3mf(bodies, palette, index_of, credit_full=None) -> bytes:
    """3MF with a basematerials palette and one watertight object per body.

    Each triangle references its material via pid/p1 (per-face colour). Keeping
    bodies as separate objects preserves watertightness; slicers union the
    overlapping parts and map each material colour to a filament. The 出典
    sentence goes into the spec's model metadata (Copyright + Description —
    the fields slicers actually surface).

    Every object is named ("forest") both in the core spec's `name` attribute
    and in a PrusaSlicer-style `Slic3r_PE_model.config`, because the two halves
    of the slicer world read different ones: Cura / 3D Builder / most viewers
    take the attribute, the PrusaSlicer family (Orca, Bambu, SuperSlicer) takes
    the config. Without a name the object list is just "Object 1..N".
    """
    meta = ""
    if credit_full:
        esc = escape(credit_full)
        meta = (
            f'<metadata name="Copyright">{esc}</metadata>'
            f'<metadata name="Description">{esc}</metadata>'
            '<metadata name="Application">3d-footprint</metadata>'
        )
    bases = "".join(
        f'<base name="{name}" displaycolor="#{r:02X}{g:02X}{b:02X}FF"/>'
        for name, (r, g, b) in palette
    )

    names = _part_names(bodies)
    objects: list[str] = []
    items: list[str] = []
    configs: list[str] = []
    for i, body in enumerate(bodies):
        oid = i + 2  # id 1 is the basematerials group
        name = escape(names[i], {'"': "&quot;"})
        verts = _rows_to_xml(np.asarray(body.mesh.vertices, dtype=np.float64),
                             '<vertex x="%.6f" y="%.6f" z="%.6f"/>')
        # One dict lookup per distinct label rather than per triangle.
        lbl = body.face_labels()
        distinct, back = np.unique(lbl, return_inverse=True)
        pid = np.array([index_of[l] for l in distinct], np.int64)[back.ravel()]
        tris = _rows_to_xml(
            np.column_stack([np.asarray(body.mesh.faces, np.int64), pid]),
            '<triangle v1="%d" v2="%d" v3="%d" pid="1" p1="%d"/>')
        objects.append(
            f'<object id="{oid}" type="model" name="{name}"><mesh>'
            f"<vertices>{verts}</vertices><triangles>{tris}</triangles>"
            "</mesh></object>"
        )
        items.append(f'<item objectid="{oid}"/>')
        # The object's single volume spans all its triangles.
        configs.append(
            f'<object id="{oid}" instances_count="1">'
            f'<metadata type="object" key="name" value="{name}"/>'
            f'<volume firstid="0" lastid="{len(body.mesh.faces) - 1}">'
            f'<metadata type="volume" key="name" value="{name}"/>'
            f'<metadata type="volume" key="volume_type" value="ModelPart"/>'
            '<mesh edges_fixed="0" degenerate_facets="0" facets_removed="0" '
            'facets_reversed="0" backwards_edges="0"/>'
            "</volume></object>"
        )

    model = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<model unit="millimeter" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
        f"{meta}"
        "<resources>"
        f'<basematerials id="1">{bases}</basematerials>'
        f'{"".join(objects)}'
        "</resources>"
        f'<build>{"".join(items)}</build>'
        "</model>"
    )

    model_config = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<config>{"".join(configs)}</config>'
    )

    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
        # Declared so the package stays OPC-valid with the slicer sidecar in it
        # (PrusaSlicer omits this; readers that check would fault on it).
        '<Default Extension="config" ContentType="text/xml"/>'
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
        z.writestr("Metadata/Slic3r_PE_model.config", model_config)
    return buf.getvalue()
