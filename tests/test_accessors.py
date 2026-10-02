"""Sparse accessors, interleaved layouts, index types, degenerate triangles."""

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest
from glb_builder import BinBuilder, write_glb

from glbmeasure import measure

INCH = 1.0 / 0.0254


def _check_bounds(result, want_min, want_max):
    assert result.bounds is not None
    for got, want in zip(result.bounds[0], want_min):
        assert got == pytest.approx(want * INCH, abs=1e-4)
    for got, want in zip(result.bounds[1], want_max):
        assert got == pytest.approx(want * INCH, abs=1e-4)


def _base_gltf(builder, meshes, accessors, nodes=None):
    return {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes if nodes is not None else [{"mesh": 0}],
        "meshes": meshes,
        "bufferViews": builder.buffer_views,
        "accessors": accessors,
        "buffers": [{"byteLength": len(builder.data)}],
    }


def test_sparse_positions_with_base_indexed(tmp_path):
    """Sparse POSITION overriding base data, indexed TRIANGLES (uint32)."""
    builder = BinBuilder()
    positions = struct.pack(
        "<12f",
        0, 0, 0,  1, 0, 0,  0, 1, 0,  0, 0, 1,
    )
    pos_view = builder.add(positions)
    sparse_idx = builder.add(struct.pack("<I", 1))          # override vertex 1
    sparse_val = builder.add(struct.pack("<3f", 5, 0, 0))   # -> (5,0,0)
    indices = builder.add(struct.pack("<6I", 0, 1, 2, 1, 2, 3))

    gltf = _base_gltf(
        builder,
        meshes=[{"primitives": [
            {"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
        accessors=[
            {"bufferView": pos_view, "componentType": 5126, "count": 4,
             "type": "VEC3",
             "sparse": {"count": 1,
                        "indices": {"bufferView": sparse_idx,
                                    "componentType": 5125},
                        "values": {"bufferView": sparse_val}}},
            {"bufferView": indices, "componentType": 5125, "count": 6,
             "type": "SCALAR"},
        ],
    )
    write_glb(tmp_path / "a.glb", gltf, bytes(builder.data))

    result = measure(str(tmp_path / "a.glb"))
    assert result.triangle_count == 2
    _check_bounds(result, (0, 0, 0), (5, 1, 1))


def test_sparse_positions_no_base_strip(tmp_path):
    """Sparse POSITION without any base bufferView, non-indexed STRIP."""
    builder = BinBuilder()
    # 4 vertices, no base data: uncovered vertex 1 reads as (0,0,0).
    sparse_idx = builder.add(struct.pack("<3H", 0, 2, 3))
    sparse_val = builder.add(struct.pack("<9f",
                                         1, 0, 0,   # v0
                                         0, 1, 0,   # v2
                                         2, 0, 0))  # v3

    gltf = _base_gltf(
        builder,
        meshes=[{"primitives": [
            {"attributes": {"POSITION": 0}, "mode": 5}]}],
        accessors=[
            {"componentType": 5126, "count": 4, "type": "VEC3",
             "sparse": {"count": 3,
                        "indices": {"bufferView": sparse_idx,
                                    "componentType": 5123},
                        "values": {"bufferView": sparse_val}}},
        ],
    )
    write_glb(tmp_path / "b.glb", gltf, bytes(builder.data))

    result = measure(str(tmp_path / "b.glb"))
    assert result.triangle_count == 2  # strip of 4 -> 2 triangles
    _check_bounds(result, (0, 0, 0), (2, 1, 0))


def test_interleaved_positions_fan(tmp_path):
    """POSITION inside an interleaved bufferView (byteStride), FAN topology."""
    builder = BinBuilder()
    verts = [(0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2)]
    blob = bytearray()
    for vx, vy, vz in verts:
        blob += struct.pack("<3f", vx, vy, vz)   # POSITION
        blob += struct.pack("<3f", 9, 9, 9)      # NORMAL (must be ignored)
    pos_view = builder.add(bytes(blob), byte_stride=24)

    gltf = _base_gltf(
        builder,
        meshes=[{"primitives": [
            {"attributes": {"POSITION": 0}, "mode": 6}]}],
        accessors=[
            {"bufferView": pos_view, "componentType": 5126, "count": 4,
             "type": "VEC3"},
        ],
    )
    write_glb(tmp_path / "c.glb", gltf, bytes(builder.data))

    result = measure(str(tmp_path / "c.glb"))
    assert result.triangle_count == 2  # fan of 4 -> 2 triangles
    _check_bounds(result, (0, 0, 0), (2, 2, 2))


def test_sparse_indices_uint8(tmp_path):
    """Sparse override on the index accessor itself (UNSIGNED_BYTE base)."""
    builder = BinBuilder()
    positions = struct.pack("<12f", 0, 0, 0, 1, 0, 0, 0, 1, 0, 7, 7, 7)
    pos_view = builder.add(positions)
    idx_view = builder.add(struct.pack("<3B", 0, 1, 2))
    sparse_idx = builder.add(struct.pack("<B", 2))   # override 3rd index entry
    sparse_val = builder.add(struct.pack("<B", 3))   # -> vertex 3

    gltf = _base_gltf(
        builder,
        meshes=[{"primitives": [
            {"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
        accessors=[
            {"bufferView": pos_view, "componentType": 5126, "count": 4,
             "type": "VEC3"},
            {"bufferView": idx_view, "componentType": 5121, "count": 3,
             "type": "SCALAR",
             "sparse": {"count": 1,
                        "indices": {"bufferView": sparse_idx,
                                    "componentType": 5121},
                        "values": {"bufferView": sparse_val}}},
        ],
    )
    write_glb(tmp_path / "d.glb", gltf, bytes(builder.data))

    result = measure(str(tmp_path / "d.glb"))
    assert result.triangle_count == 1
    _check_bounds(result, (0, 0, 0), (7, 7, 7))


def test_degenerate_triangles_counted(tmp_path):
    """Degenerate triangles (repeated indices) still count and bound."""
    builder = BinBuilder()
    positions = struct.pack("<6f", 3, 4, 5, 1, 1, 1)
    pos_view = builder.add(positions)
    idx_view = builder.add(struct.pack("<3H", 0, 0, 0))

    gltf = _base_gltf(
        builder,
        meshes=[{"primitives": [
            {"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
        accessors=[
            {"bufferView": pos_view, "componentType": 5126, "count": 2,
             "type": "VEC3"},
            {"bufferView": idx_view, "componentType": 5123, "count": 3,
             "type": "SCALAR"},
        ],
    )
    write_glb(tmp_path / "e.glb", gltf, bytes(builder.data))

    result = measure(str(tmp_path / "e.glb"))
    assert result.triangle_count == 1
    _check_bounds(result, (3, 4, 5), (3, 4, 5))
