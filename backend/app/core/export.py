"""Export a mesh to a 3D-printable format. Pluggable per format.

M2 supports single-body STL / 3MF / GLB. Multi-color (per-layer bodies ->
3MF materials, color-split STL zip) lands in M4 on top of the same body list.
"""
from __future__ import annotations

import trimesh

# (content_type, file extension) per format.
_FORMATS = {
    "stl": ("model/stl", "stl"),
    "3mf": ("model/3mf", "3mf"),
    "glb": ("model/gltf-binary", "glb"),
}


def export_mesh(mesh: trimesh.Trimesh, fmt: str) -> tuple[bytes, str, str]:
    """Return (data, content_type, extension) for the given format."""
    if fmt not in _FORMATS:
        raise ValueError(f"unsupported format: {fmt}")
    content_type, ext = _FORMATS[fmt]

    if fmt == "glb":
        data = trimesh.Scene(mesh).export(file_type="glb")
    else:
        data = mesh.export(file_type=fmt)

    if isinstance(data, str):
        data = data.encode()
    return data, content_type, ext
