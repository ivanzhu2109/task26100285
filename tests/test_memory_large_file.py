"""~500MB GLB: peak memory delta must stay under 128MiB and results must be
unaffected by a huge embedded image or an unselected scene."""

import json
import resource
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pytest
from glb_builder import BinBuilder, build_glb, write_glb

from glbmeasure import measure

INCH = 1.0 / 0.0254
IMAGE_SIZE = 450 * 1024 * 1024  # ~450MiB of irrelevant embedded image data
VERTEX_COUNT = 3_000_000  # 1M triangles, non-indexed


def _geometry_blob() -> bytes:
    rng = np.random.default_rng(42)
    verts = rng.uniform(-2.0, 2.0, size=(VERTEX_COUNT, 3)).astype("<f4")
    return verts.tobytes()


def _make_gltf(geom_view: int, geom_len: int, extra: dict) -> dict:
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}, {"nodes": [1]}],
        "nodes": [
            {"mesh": 0},
            {"mesh": 1, "translation": [500, 0, 0]},  # unselected scene
        ],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0}}]},
            {"primitives": [{"attributes": {"POSITION": 1}}]},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": geom_len},
            {"buffer": 0, "byteOffset": 0, "byteLength": 12},
        ],
        "accessors": [
            {"bufferView": geom_view, "componentType": 5126,
             "count": VERTEX_COUNT, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5126, "count": 3,
             "type": "VEC3", "byteOffset": 0},
        ],
        "buffers": [{"byteLength": geom_len}],
    }
    gltf.update(extra)
    return gltf


def _expected(geom: bytes):
    verts = np.frombuffer(geom, dtype="<f4").reshape(-1, 3).astype(np.float64)
    return VERTEX_COUNT // 3, verts.min(axis=0) * INCH, verts.max(axis=0) * INCH


def _rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports KiB.
    return rss if sys.platform == "darwin" else rss * 1024


@pytest.fixture(scope="module")
def geom_blob() -> bytes:
    return _geometry_blob()


def test_large_file_memory_and_isolation(tmp_path, geom_blob):
    geom_len = len(geom_blob)
    tri_count, want_min, want_max = _expected(geom_blob)

    # Baseline: geometry only.
    small = tmp_path / "small.glb"
    write_glb(small, _make_gltf(0, geom_len, {}), geom_blob)
    base = measure(str(small))
    assert base.triangle_count == tri_count

    # Large file: geometry + unselected scene + ~450MiB image bufferView.
    image_offset = geom_len
    image_offset += -image_offset % 4
    bin_len = image_offset + IMAGE_SIZE
    gltf = _make_gltf(0, geom_len, {})
    gltf["bufferViews"].append(
        {"buffer": 0, "byteOffset": image_offset, "byteLength": IMAGE_SIZE}
    )
    gltf["images"] = [{"bufferView": 2, "mimeType": "image/png"}]
    gltf["buffers"] = [{"byteLength": bin_len}]

    payload = build_glb(gltf, geom_blob)  # header+JSON+BIN with geometry only
    # Rebuild manually so we can extend the BIN chunk sparsely.
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode()
    json_bytes += b" " * (-len(json_bytes) % 4)
    bin_padded_len = bin_len + (-bin_len % 4)
    total = 12 + 8 + len(json_bytes) + 8 + bin_padded_len

    large = tmp_path / "large.glb"
    with open(large, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        fh.write(json_bytes)
        fh.write(struct.pack("<II", bin_padded_len, 0x004E4942))
        fh.write(geom_blob)
        fh.write(b"\x00" * (image_offset - geom_len))
        fh.truncate(12 + 8 + len(json_bytes) + 8 + bin_padded_len)
    assert large.stat().st_size > 450 * 1024 * 1024

    rss_before = _rss_bytes()
    big = measure(str(large))
    rss_delta = _rss_bytes() - rss_before

    assert big.triangle_count == base.triangle_count
    for got, want in zip(big.bounds[0] + big.bounds[1],
                         base.bounds[0] + base.bounds[1]):
        assert got == pytest.approx(want, abs=1e-9)
    for got, want in zip(big.bounds[0], want_min):
        assert got == pytest.approx(want, abs=1e-4)
    for got, want in zip(big.bounds[1], want_max):
        assert got == pytest.approx(want, abs=1e-4)
    assert rss_delta < 128 * 1024 * 1024, (
        f"peak RSS grew by {rss_delta / 2**20:.1f} MiB"
    )
