"""Scene selection, unreferenced content, empty results, default mode."""

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest
from glb_builder import BinBuilder, write_glb

from glbmeasure import measure

INCH = 1.0 / 0.0254


def _two_scene_glb(path: Path, with_scene_key: bool) -> None:
    """Scene 0 has a huge triangle; scene 1 a small one. Mesh 2 unreferenced."""
    builder = BinBuilder()
    big = builder.add(struct.pack("<9f", 0, 0, 0, 100, 0, 0, 0, 100, 0))
    small = builder.add(struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0))
    stray = builder.add(struct.pack("<9f", 0, 0, 0, 9999, 0, 0, 0, 9999, 0))

    gltf = {
        "asset": {"version": "2.0"},
        "scenes": [{"nodes": [0]}, {"nodes": [1]}],
        "nodes": [{"mesh": 0}, {"mesh": 1}, {"mesh": 2}],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0}}]},  # mode omitted
            {"primitives": [{"attributes": {"POSITION": 1}}]},
            {"primitives": [{"attributes": {"POSITION": 2}}]},
        ],
        "bufferViews": builder.buffer_views,
        "accessors": [
            {"bufferView": big, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": small, "componentType": 5126, "count": 3,
             "type": "VEC3"},
            {"bufferView": stray, "componentType": 5126, "count": 3,
             "type": "VEC3"},
        ],
        "buffers": [{"byteLength": len(builder.data)}],
    }
    if with_scene_key:
        gltf["scene"] = 1
    write_glb(path, gltf, bytes(builder.data))


def test_default_scene_key_respected(tmp_path):
    _two_scene_glb(tmp_path / "s.glb", with_scene_key=True)
    result = measure(str(tmp_path / "s.glb"))
    assert result.triangle_count == 1
    # Scene 1: small triangle only.
    for got, want in zip(result.bounds[1], (INCH, INCH, 0.0)):
        assert got == pytest.approx(want, abs=1e-4)


def test_first_scene_when_unspecified(tmp_path):
    _two_scene_glb(tmp_path / "s.glb", with_scene_key=False)
    result = measure(str(tmp_path / "s.glb"))
    assert result.triangle_count == 1
    # Scene 0: big triangle only; stray mesh 2 must not leak in.
    for got, want in zip(result.bounds[1], (100 * INCH, 100 * INCH, 0.0)):
        assert got == pytest.approx(want, abs=1e-4)


def test_no_triangles_gives_empty_bounds(tmp_path):
    builder = BinBuilder()
    pts = builder.add(struct.pack("<6f", 5, 5, 5, 6, 6, 6))
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {"primitives": [
                {"attributes": {"POSITION": 0}, "mode": 1},  # LINES
                {"attributes": {"POSITION": 0}, "mode": 0},  # POINTS
            ]}
        ],
        "bufferViews": builder.buffer_views,
        "accessors": [
            {"bufferView": pts, "componentType": 5126, "count": 2,
             "type": "VEC3"},
        ],
        "buffers": [{"byteLength": len(builder.data)}],
    }
    write_glb(tmp_path / "empty.glb", gltf, bytes(builder.data))

    result = measure(str(tmp_path / "empty.glb"))
    assert result.triangle_count == 0
    assert result.bounds is None
