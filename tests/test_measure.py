"""Acceptance tests for glb_measure."""

import math
import struct

import pytest

from glb_measure import measure_glb
from tests.glb_builder import BinBuilder, write_glb

IN = 1.0 / 0.0254  # meters -> inches
TOL = 1e-4


def assert_bbox(bbox, expected_min, expected_max):
    assert bbox is not None
    for got, want in zip(bbox.min, expected_min):
        assert got == pytest.approx(want * IN, abs=TOL)
    for got, want in zip(bbox.max, expected_max):
        assert got == pytest.approx(want * IN, abs=TOL)


def build_gltf(bin_b, nodes, scenes, meshes, accessors, scene=None):
    gltf = {
        "asset": {"version": "2.0"},
        "scenes": scenes,
        "nodes": nodes,
        "meshes": meshes,
        "bufferViews": bin_b.buffer_views,
        "accessors": accessors,
    }
    if scene is not None:
        gltf["scene"] = scene
    return gltf


def test_rotated_mesh_ignores_unreferenced_extreme_vertices(tmp_path):
    """A rotated indexed triangle with stray extreme vertices in the same
    POSITION accessor: count stays 1 and bounds hug the real geometry."""
    b = BinBuilder()
    positions = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
        (100.0, 100.0, 100.0), (-100.0, -100.0, -100.0),
    ]
    pos_view = b.add_floats([c for p in positions for c in p])
    idx_view = b.add_uints([0, 1, 2], "I")

    half = math.pi / 4.0
    gltf = build_gltf(
        b,
        nodes=[{"mesh": 0,
                "rotation": [0.0, 0.0, math.sin(half), math.cos(half)]}],
        scenes=[{"nodes": [0]}],
        meshes=[{"primitives": [
            {"attributes": {"POSITION": 0}, "indices": 1}]}],
        accessors=[
            {"bufferView": pos_view, "componentType": 5126,
             "count": 5, "type": "VEC3"},
            {"bufferView": idx_view, "componentType": 5125,
             "count": 3, "type": "SCALAR"},
        ],
    )
    path = tmp_path / "rotated.glb"
    write_glb(path, gltf, bytes(b.data))

    result = measure_glb(str(path))
    assert result.triangle_count == 1
    # 90 deg about Z maps (x, y) -> (-y, x); the +/-100 vertices are
    # not referenced by any index and must not widen the box.
    assert_bbox(result.bbox, (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def test_same_mesh_instanced_under_different_ancestors(tmp_path):
    """One mesh referenced by three nodes with different ancestor
    transforms: each instance counts and contributes its own extent."""
    b = BinBuilder()
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    pos_view = b.add_floats([c for p in positions for c in p])

    half = math.pi / 4.0
    nodes = [
        {"children": [1, 2, 3]},
        # instance 1: plain translation
        {"mesh": 0, "translation": [10.0, 0.0, 0.0]},
        # instance 2: rotated + non-uniformly scaled parent, translated child
        {"rotation": [0.0, 0.0, math.sin(half), math.cos(half)],
         "scale": [2.0, 0.5, 1.0],
         "children": [4]},
        # instance 3: negative scale (mirror across YZ plane)
        {"mesh": 0, "scale": [-1.0, 1.0, 1.0]},
        {"mesh": 0, "translation": [0.0, 5.0, 0.0]},
    ]
    gltf = build_gltf(
        b,
        nodes=nodes,
        scenes=[{"nodes": [0]}],
        meshes=[{"primitives": [{"attributes": {"POSITION": 0}}]}],
        accessors=[
            {"bufferView": pos_view, "componentType": 5126,
             "count": 3, "type": "VEC3"},
        ],
    )
    path = tmp_path / "instanced.glb"
    write_glb(path, gltf, bytes(b.data))

    result = measure_glb(str(path))
    assert result.triangle_count == 3
    # instance 1 verts: (10,0,0) (11,0,0) (10,1,0)
    # instance 2 verts: (-2.5,0,0) (-2.5,2,0) (-3,0,0)
    # instance 3 verts: (0,0,0) (-1,0,0) (0,1,0)
    assert_bbox(result.bbox, (-3.0, 0.0, 0.0), (11.0, 2.0, 0.0))


def test_sparse_positions_with_base_indexed(tmp_path):
    """Sparse override on top of a base bufferView, indexed TRIANGLES."""
    b = BinBuilder()
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                 (5.0, 5.0, 5.0)]
    pos_view = b.add_floats([c for p in positions for c in p])
    idx_view = b.add_uints([0, 1, 2, 1, 2, 3], "H")
    sparse_idx_view = b.add_uints([3], "I")
    sparse_val_view = b.add_floats([0.0, 0.0, 1.0])

    gltf = build_gltf(
        b,
        nodes=[{"mesh": 0}],
        scenes=[{"nodes": [0]}],
        meshes=[{"primitives": [
            {"attributes": {"POSITION": 0}, "indices": 1}]}],
        accessors=[
            {"bufferView": pos_view, "componentType": 5126,
             "count": 4, "type": "VEC3",
             "sparse": {"count": 1,
                        "indices": {"bufferView": sparse_idx_view,
                                    "componentType": 5125},
                        "values": {"bufferView": sparse_val_view}}},
            {"bufferView": idx_view, "componentType": 5123,
             "count": 6, "type": "SCALAR"},
        ],
    )
    path = tmp_path / "sparse_base.glb"
    write_glb(path, gltf, bytes(b.data))

    result = measure_glb(str(path))
    assert result.triangle_count == 2
    # vertex 3 becomes (0,0,1); (5,5,5) never participates
    assert_bbox(result.bbox, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))


def test_sparse_positions_without_base_strip_unindexed(tmp_path):
    """Sparse accessor with no base bufferView (zeros elsewhere),
    non-indexed TRIANGLE_STRIP."""
    b = BinBuilder()
    sparse_idx_view = b.add_uints([0, 1, 2, 3], "H")
    sparse_val_view = b.add_floats(
        [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0])

    gltf = build_gltf(
        b,
        nodes=[{"mesh": 0}],
        scenes=[{"nodes": [0]}],
        meshes=[{"primitives": [
            {"attributes": {"POSITION": 0}, "mode": 5}]}],
        accessors=[
            {"componentType": 5126, "count": 4, "type": "VEC3",
             "sparse": {"count": 4,
                        "indices": {"bufferView": sparse_idx_view,
                                    "componentType": 5123},
                        "values": {"bufferView": sparse_val_view}}},
        ],
    )
    path = tmp_path / "sparse_nobase.glb"
    write_glb(path, gltf, bytes(b.data))

    result = measure_glb(str(path))
    assert result.triangle_count == 2  # strip of 4 vertices
    assert_bbox(result.bbox, (0.0, 0.0, 0.0), (2.0, 2.0, 0.0))


def test_interleaved_stride_fan_unindexed(tmp_path):
    """POSITION interleaved with normals via byteStride,
    non-indexed TRIANGLE_FAN."""
    b = BinBuilder()
    verts = [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 3.0, 0.0),
             (-3.0, 0.0, 0.0)]
    payload = bytearray()
    for v in verts:
        payload += struct.pack("<3f", *v)
        payload += struct.pack("<3f", 0.0, 0.0, 1.0)
    pos_view = b.add(bytes(payload), byte_stride=24)

    gltf = build_gltf(
        b,
        nodes=[{"mesh": 0}],
        scenes=[{"nodes": [0]}],
        meshes=[{"primitives": [
            {"attributes": {"POSITION": 0}, "mode": 6}]}],
        accessors=[
            {"bufferView": pos_view, "componentType": 5126,
             "count": 4, "type": "VEC3"},
        ],
    )
    path = tmp_path / "interleaved.glb"
    write_glb(path, gltf, bytes(b.data))

    result = measure_glb(str(path))
    assert result.triangle_count == 2  # fan of 4 vertices
    assert_bbox(result.bbox, (-3.0, 0.0, 0.0), (3.0, 3.0, 0.0))


def test_sparse_indices_accessor(tmp_path):
    """The index accessor itself may be sparse."""
    b = BinBuilder()
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                 (7.0, 8.0, 9.0)]
    pos_view = b.add_floats([c for p in positions for c in p])
    idx_view = b.add_uints([0, 1, 2], "H")
    sparse_idx_view = b.add_uints([2], "B")
    sparse_val_view = b.add_uints([3], "H")

    gltf = build_gltf(
        b,
        nodes=[{"mesh": 0}],
        scenes=[{"nodes": [0]}],
        meshes=[{"primitives": [
            {"attributes": {"POSITION": 0}, "indices": 1}]}],
        accessors=[
            {"bufferView": pos_view, "componentType": 5126,
             "count": 4, "type": "VEC3"},
            {"bufferView": idx_view, "componentType": 5123,
             "count": 3, "type": "SCALAR",
             "sparse": {"count": 1,
                        "indices": {"bufferView": sparse_idx_view,
                                    "componentType": 5121},
                        "values": {"bufferView": sparse_val_view}}},
        ],
    )
    path = tmp_path / "sparse_indices.glb"
    write_glb(path, gltf, bytes(b.data))

    result = measure_glb(str(path))
    assert result.triangle_count == 1
    # index 2 overridden to 3 -> vertices 0, 1, 3 participate
    assert_bbox(result.bbox, (0.0, 0.0, 0.0), (7.0, 8.0, 9.0))


def test_degenerate_triangles_still_count(tmp_path):
    """Degenerate triangles (repeated indices / collinear vertices)
    each count as one face and their vertices join the bounds."""
    b = BinBuilder()
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                 (0.0, 1.0, 0.0), (2.0, 2.0, 0.0)]
    pos_view = b.add_floats([c for p in positions for c in p])
    idx_view = b.add_uints([0, 1, 2, 3, 4], "I")

    gltf = build_gltf(
        b,
        nodes=[{"mesh": 0}],
        scenes=[{"nodes": [0]}],
        meshes=[{"primitives": [
            {"attributes": {"POSITION": 0}, "indices": 1, "mode": 5}]}],
        accessors=[
            {"bufferView": pos_view, "componentType": 5126,
             "count": 5, "type": "VEC3"},
            {"bufferView": idx_view, "componentType": 5125,
             "count": 5, "type": "SCALAR"},
        ],
    )
    path = tmp_path / "degenerate.glb"
    write_glb(path, gltf, bytes(b.data))

    result = measure_glb(str(path))
    assert result.triangle_count == 3  # strip of 5, degenerate included
    assert_bbox(result.bbox, (0.0, 0.0, 0.0), (2.0, 2.0, 0.0))


def test_no_triangles_gives_empty_bbox(tmp_path):
    """Non-triangle modes contribute nothing; bbox is empty, not a
    zero-size box at the origin."""
    b = BinBuilder()
    pos_view = b.add_floats([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    gltf = build_gltf(
        b,
        nodes=[{"mesh": 0}],
        scenes=[{"nodes": [0]}],
        meshes=[{"primitives": [
            {"attributes": {"POSITION": 0}, "mode": 1}]}],  # LINES
        accessors=[
            {"bufferView": pos_view, "componentType": 5126,
             "count": 2, "type": "VEC3"},
        ],
    )
    path = tmp_path / "lines.glb"
    write_glb(path, gltf, bytes(b.data))

    result = measure_glb(str(path))
    assert result.triangle_count == 0
    assert result.bbox is None


def test_scene_selection_and_unreferenced_content(tmp_path):
    """Only the selected scene is measured; other scenes and
    unreferenced meshes do not affect the result."""
    b = BinBuilder()
    pos_a = b.add_floats([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    pos_b = b.add_floats([50.0, 0.0, 0.0, 51.0, 0.0, 0.0, 50.0, 1.0, 0.0])
    pos_c = b.add_floats([900.0, 900.0, 900.0,
                          901.0, 900.0, 900.0, 900.0, 901.0, 900.0])

    accessors = [
        {"bufferView": pos_a, "componentType": 5126,
         "count": 3, "type": "VEC3"},
        {"bufferView": pos_b, "componentType": 5126,
         "count": 3, "type": "VEC3"},
        {"bufferView": pos_c, "componentType": 5126,
         "count": 3, "type": "VEC3"},
    ]
    meshes = [
        {"primitives": [{"attributes": {"POSITION": 0}}]},
        {"primitives": [{"attributes": {"POSITION": 1}}]},
        {"primitives": [{"attributes": {"POSITION": 2}}]},
    ]
    nodes = [{"mesh": 0}, {"mesh": 1}]

    # default scene explicitly set to scene 1
    gltf = build_gltf(b, nodes=nodes,
                      scenes=[{"nodes": [0]}, {"nodes": [1]}],
                      meshes=meshes, accessors=accessors, scene=1)
    path = tmp_path / "scenes.glb"
    write_glb(path, gltf, bytes(b.data))

    result = measure_glb(str(path))
    assert result.triangle_count == 1
    assert_bbox(result.bbox, (50.0, 0.0, 0.0), (51.0, 1.0, 0.0))

    # no default scene -> first scene wins
    gltf2 = build_gltf(b, nodes=nodes,
                       scenes=[{"nodes": [0]}, {"nodes": [1]}],
                       meshes=meshes, accessors=accessors)
    path2 = tmp_path / "scenes_first.glb"
    write_glb(path2, gltf2, bytes(b.data))

    result2 = measure_glb(str(path2))
    assert result2.triangle_count == 1
    assert_bbox(result2.bbox, (0.0, 0.0, 0.0), (1.0, 1.0, 0.0))


def test_explicit_node_matrix(tmp_path):
    """Nodes may carry a column-major matrix instead of TRS."""
    b = BinBuilder()
    pos_view = b.add_floats([0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                             0.0, 1.0, 0.0])

    # rotation 90 deg about Z followed by translation (5, 6, 7)
    matrix = [0.0, 1.0, 0.0, 0.0,
              -1.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 1.0, 0.0,
              5.0, 6.0, 7.0, 1.0]
    gltf = build_gltf(
        b,
        nodes=[{"mesh": 0, "matrix": matrix}],
        scenes=[{"nodes": [0]}],
        meshes=[{"primitives": [{"attributes": {"POSITION": 0}}]}],
        accessors=[
            {"bufferView": pos_view, "componentType": 5126,
             "count": 3, "type": "VEC3"},
        ],
    )
    path = tmp_path / "matrix.glb"
    write_glb(path, gltf, bytes(b.data))

    result = measure_glb(str(path))
    assert result.triangle_count == 1
    # (x,y) -> (-y,x) then +(5,6,7): verts (5,6,7) (5,7,7) (4,6,7)
    assert_bbox(result.bbox, (4.0, 6.0, 7.0), (5.0, 7.0, 7.0))
