"""Helpers to build minimal GLB files for tests."""

from __future__ import annotations

import json
import struct


def build_glb(gltf: dict, bin_data: bytes) -> bytes:
    """Assemble a GLB byte string from a glTF JSON dict and BIN chunk data."""
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * (-len(json_bytes) % 4)
    bin_padded = bin_data + b"\x00" * (-len(bin_data) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_padded)
    out = [struct.pack("<III", 0x46546C67, 2, total)]
    out.append(struct.pack("<II", len(json_bytes), 0x4E4F534A))
    out.append(json_bytes)
    out.append(struct.pack("<II", len(bin_padded), 0x004E4942))
    out.append(bin_padded)
    return b"".join(out)


class BinBuilder:
    """Accumulates binary blobs and records bufferViews."""

    def __init__(self) -> None:
        self.data = bytearray()
        self.buffer_views: list[dict] = []

    def add(self, blob: bytes, byte_stride: int | None = None, align: int = 4) -> int:
        pad = -len(self.data) % align
        self.data += b"\x00" * pad
        offset = len(self.data)
        self.data += blob
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(blob)}
        if byte_stride is not None:
            view["byteStride"] = byte_stride
        self.buffer_views.append(view)
        return len(self.buffer_views) - 1


def write_glb(path, gltf: dict, bin_data: bytes) -> None:
    with open(path, "wb") as fh:
        fh.write(build_glb(gltf, bin_data))
