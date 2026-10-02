"""Memory validation with a near-500MB GLB file.

The file embeds a huge dummy image bufferView plus an unselected
scene; neither may influence the measurement, and the peak memory
delta of a single measure_glb call must stay below 128 MiB.
"""

import gc
import json
import resource
import struct
import sys

import pytest

from glb_measure import measure_glb

IN = 1.0 / 0.0254
TOL = 1e-4
MIB = 1024 * 1024

IMAGE_SIZE = 470 * MIB  # total file size ends up just under 500 MB


def _peak_rss_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss  # bytes on macOS
    return rss * 1024  # KiB on Linux


def _build_large_glb(path):
    positions = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    other_positions = [500.0, 500.0, 500.0, 501.0, 500.0, 500.0,
                       500.0, 501.0, 500.0]
    geometry = struct.pack("<9f", *positions)
    other_geometry = struct.pack("<9f", *other_positions)

    pos_offset = 0
    other_offset = len(geometry)
    image_offset = other_offset + len(other_geometry)
    bin_length = image_offset + IMAGE_SIZE

    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}, {"nodes": [1]}],
        "nodes": [{"mesh": 0}, {"mesh": 1}],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0}}]},
            {"primitives": [{"attributes": {"POSITION": 1}}]},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": pos_offset, "byteLength": 36},
            {"buffer": 0, "byteOffset": other_offset, "byteLength": 36},
            {"buffer": 0, "byteOffset": image_offset,
             "byteLength": IMAGE_SIZE},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126,
             "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5126,
             "count": 3, "type": "VEC3"},
        ],
        "images": [{"bufferView": 2, "mimeType": "image/png"}],
    }

    js = json.dumps(gltf).encode("utf-8")
    js += b" " * (-len(js) % 4)
    bin_pad = b"\x00" * (-bin_length % 4)
    total = 12 + 8 + len(js) + 8 + bin_length + len(bin_pad)

    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A))
        f.write(js)
        f.write(struct.pack("<II", bin_length + len(bin_pad), 0x004E4942))
        f.write(geometry)
        f.write(other_geometry)
        f.truncate(12 + 8 + len(js) + 8 + bin_length + len(bin_pad))


def test_large_file_memory_and_irrelevant_content(tmp_path):
    path = tmp_path / "large.glb"
    _build_large_glb(str(path))
    assert path.stat().st_size > IMAGE_SIZE

    gc.collect()
    before = _peak_rss_bytes()
    result = measure_glb(str(path))
    after = _peak_rss_bytes()

    assert result.triangle_count == 1
    for got, want in zip(result.bbox.min, (0.0, 0.0, 0.0)):
        assert got == pytest.approx(want * IN, abs=TOL)
    for got, want in zip(result.bbox.max, (1.0, 1.0, 0.0)):
        assert got == pytest.approx(want * IN, abs=TOL)

    delta_mib = (after - before) / MIB
    assert delta_mib <= 128, "peak memory delta %.1f MiB" % delta_mib
