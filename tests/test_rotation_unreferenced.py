"""Rotated mesh with unreferenced extreme vertices: bounds must hug real geometry."""

import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest
from glb_builder import BinBuilder, write_glb

from glbmeasure import measure

INCH = 1.0 / 0.0254


def _rotated_triangle_glb(path: Path) -> None:
    # 4 vertices; vertex 3 is an extreme point never referenced by indices.
    positions = struct.pack(
        "<12f",
        1.0, 0.0, 0.0,   # v0
        0.0, 1.0, 0.0,   # v1
        0.0, 0.0, 1.0,   # v2
        1000.0, 1000.0, 1000.0,  # v3: unreferenced extreme
    )
    indices = struct.pack("<3H", 0, 1, 2)

    builder = BinBuilder()
    pos_view = builder.add(positions)
    idx_view = builder.add(indices)

    half = math.sqrt(0.5)  # 90 degrees about Z
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"mesh": 0, "rotation": [0.0, 0.0, half, half]},
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                        "mode": 4,
                    }
                ]
            }
        ],
        "bufferViews": builder.buffer_views,
        "accessors": [
            {
                "bufferView": pos_view,
                "componentType": 5126,
                "count": 4,
                "type": "VEC3",
            },
            {
                "bufferView": idx_view,
                "componentType": 5123,
                "count": 3,
                "type": "SCALAR",
            },
        ],
        "buffers": [{"byteLength": len(builder.data)}],
    }
    write_glb(path, gltf, bytes(builder.data))


def test_rotated_mesh_excludes_unreferenced_extremes(tmp_path):
    glb_path = tmp_path / "rotated.glb"
    _rotated_triangle_glb(glb_path)

    result = measure(str(glb_path))

    assert result.triangle_count == 1
    # 90 deg about Z maps (x, y, z) -> (-y, x, z):
    # v0 -> (0, 1, 0), v1 -> (-1, 0, 0), v2 -> (0, 0, 1)
    expected_min = (-1.0 * INCH, 0.0, 0.0)
    expected_max = (0.0, 1.0 * INCH, 1.0 * INCH)
    assert result.bounds is not None
    for got, want in zip(result.bounds[0], expected_min):
        assert got == pytest.approx(want, abs=1e-4)
    for got, want in zip(result.bounds[1], expected_max):
        assert got == pytest.approx(want, abs=1e-4)
