"""Absolute error must stay under 1e-4 inch at world coords up to 10000 m."""

import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest
from glb_builder import BinBuilder, write_glb

from glbmeasure import measure

INCH = 1.0 / 0.0254


def test_large_world_coordinates_precision(tmp_path):
    builder = BinBuilder()
    # A unit-ish triangle far from the origin.
    positions = struct.pack(
        "<9f",
        0.0, 0.0, 0.0,
        1.5, 0.25, 0.0,
        0.0, 0.25, 2.5,
    )
    pos_view = builder.add(positions)

    # Rotate 30 deg about Y and translate to (8000, -9000, 10000).
    angle = math.radians(30)
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {
                "mesh": 0,
                "translation": [8000.0, -9000.0, 10000.0],
                "rotation": [0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2)],
            }
        ],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0}}]}
        ],
        "bufferViews": builder.buffer_views,
        "accessors": [
            {"bufferView": pos_view, "componentType": 5126, "count": 3,
             "type": "VEC3"},
        ],
        "buffers": [{"byteLength": len(builder.data)}],
    }
    write_glb(tmp_path / "far.glb", gltf, bytes(builder.data))

    result = measure(str(tmp_path / "far.glb"))

    # Reference computation in Python floats (rotation about Y):
    # x' = x*cos + z*sin ; z' = -x*sin + z*cos
    c, s = math.cos(angle), math.sin(angle)
    pts = [(0.0, 0.0, 0.0), (1.5, 0.25, 0.0), (0.0, 0.25, 2.5)]
    world = [
        (x * c + z * s + 8000.0, y - 9000.0, -x * s + z * c + 10000.0)
        for x, y, z in pts
    ]
    want_min = tuple(min(p[i] for p in world) * INCH for i in range(3))
    want_max = tuple(max(p[i] for p in world) * INCH for i in range(3))

    assert result.triangle_count == 1
    for got, want in zip(result.bounds[0], want_min):
        assert got == pytest.approx(want, abs=1e-4)
    for got, want in zip(result.bounds[1], want_max):
        assert got == pytest.approx(want, abs=1e-4)
