"""One mesh referenced by several nodes under different ancestor transforms."""

import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest
from glb_builder import BinBuilder, write_glb

from glbmeasure import measure

INCH = 1.0 / 0.0254


def _instanced_glb(path: Path) -> None:
    # Single triangle: (0,0,0) (1,0,0) (0,1,0), no indices.
    positions = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    builder = BinBuilder()
    pos_view = builder.add(positions)

    # Node 0: plain instance at origin (TRS defaults).
    # Node 1: translated by (10, 0, 0), non-uniform scale (2, 3, 1).
    # Node 2: child of node 1, negative scale x=-1, rotation 90 deg about Z,
    #         so its world transform composes both ancestors.
    # Node 3: uses an explicit matrix (translation (0, 0, 5) + scale 2),
    #         stored column-major per glTF.
    half = math.sqrt(0.5)
    matrix = [
        2, 0, 0, 0,   # column 0
        0, 2, 0, 0,   # column 1
        0, 0, 2, 0,   # column 2
        0, 0, 5, 1,   # column 3 (translation)
    ]
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1, 3]}],
        "nodes": [
            {"mesh": 0},
            {"mesh": 0, "translation": [10, 0, 0], "scale": [2, 3, 1],
             "children": [2]},
            {"mesh": 0, "rotation": [0, 0, half, half], "scale": [-1, 1, 1]},
            {"mesh": 0, "matrix": matrix},
        ],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0}, "mode": 4}]}
        ],
        "bufferViews": builder.buffer_views,
        "accessors": [
            {"bufferView": pos_view, "componentType": 5126, "count": 3,
             "type": "VEC3"},
        ],
        "buffers": [{"byteLength": len(builder.data)}],
    }
    write_glb(path, gltf, bytes(builder.data))


def test_multiple_instances_counted_separately(tmp_path):
    glb_path = tmp_path / "instanced.glb"
    _instanced_glb(glb_path)

    result = measure(str(glb_path))

    # 4 nodes referencing the same 1-triangle mesh.
    assert result.triangle_count == 4

    # Instance world-space vertices:
    # node0: (0,0,0) (1,0,0) (0,1,0)
    # node1: (10,0,0) (12,0,0) (10,3,0)
    # node2 (world = node1 * local): local maps (x,y,z) -> rot90z(scale(x,y,z))
    #   = (-y, -x, z) with scale -1 on x after rotation... compute explicitly:
    #   local linear = R90z * diag(-1,1,1); v=(0,0,0)->(0,0,0); (1,0,0)->(0,-1,0);
    #   (0,1,0)->(-1,0,0). Then node1: scale (2,3,1) + translate (10,0,0):
    #   -> (10,0,0), (10,-3,0), (8,0,0)
    # node3: (0,0,5) (2,0,5) (0,2,5)
    xs = [0, 1, 0, 10, 12, 10, 10, 10, 8, 0, 2, 0]
    ys = [0, 0, 1, 0, 0, 3, 0, -3, 0, 0, 0, 2]
    zs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5]
    expected_min = (min(xs) * INCH, min(ys) * INCH, min(zs) * INCH)
    expected_max = (max(xs) * INCH, max(ys) * INCH, max(zs) * INCH)

    assert result.bounds is not None
    for got, want in zip(result.bounds[0], expected_min):
        assert got == pytest.approx(want, abs=1e-4)
    for got, want in zip(result.bounds[1], expected_max):
        assert got == pytest.approx(want, abs=1e-4)
