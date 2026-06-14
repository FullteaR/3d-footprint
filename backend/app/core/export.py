"""Export named bodies to a 3D-printable format. Pluggable per format.

Bodies are kept separate (one per colour layer). The heavy work (building the
geometry) is shared; only the serializer differs:
  - 3mf / glb : a Scene with one named object per body (multi-colour ready).
  - stl       : bodies concatenated into a single file (STL carries no colour).
"""
from __future__ import annotations

import trimesh

Body = tuple[str, trimesh.Trimesh]

# Default per-layer colours (RGBA), used for GLB preview and 3MF objects.
LAYER_COLORS: dict[str, list[int]] = {
    "terrain": [194, 178, 128, 255],
    "track": [220, 70, 40, 255],
    "base": [70, 70, 70, 255],
}

_FORMATS = {
    "stl": ("model/stl", "stl"),
    "3mf": ("model/3mf", "3mf"),
    "glb": ("model/gltf-binary", "glb"),
}


def _colorize(name: str, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    color = LAYER_COLORS.get(name)
    if color is not None:
        mesh.visual.face_colors = color
    return mesh


def export_bodies(bodies: list[Body], fmt: str) -> tuple[bytes, str, str]:
    """Return (data, content_type, extension) for the given format."""
    if fmt not in _FORMATS:
        raise ValueError(f"unsupported format: {fmt}")
    content_type, ext = _FORMATS[fmt]

    bodies = [(name, _colorize(name, m)) for name, m in bodies]

    if fmt == "stl":
        merged = trimesh.util.concatenate([m for _, m in bodies])
        data = merged.export(file_type="stl")
    else:
        scene = trimesh.Scene()
        for name, m in bodies:
            scene.add_geometry(m, geom_name=name, node_name=name)
        data = scene.export(file_type=fmt)

    if isinstance(data, str):
        data = data.encode()
    return data, content_type, ext
