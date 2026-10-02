"""Helpers to build synthetic GLB files for tests."""

import json
import struct


class BinBuilder:
    """Accumulates BIN chunk data and emits bufferView dicts."""

    def __init__(self):
        self.data = bytearray()
        self.buffer_views = []

    def add(self, payload: bytes, byte_stride=None) -> int:
        """Append raw bytes (4-byte aligned) and return bufferView index."""
        pad = -len(self.data) % 4
        self.data += b"\x00" * pad
        offset = len(self.data)
        self.data += payload
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
        if byte_stride is not None:
            view["byteStride"] = byte_stride
        self.buffer_views.append(view)
        return len(self.buffer_views) - 1

    def add_floats(self, values, byte_stride=None) -> int:
        return self.add(struct.pack("<%df" % len(values), *values), byte_stride)

    def add_uints(self, values, fmt) -> int:
        return self.add(struct.pack("<%d%s" % (len(values), fmt), *values))


def write_glb(path, gltf: dict, bin_data: bytes) -> None:
    """Write a GLB file with one JSON chunk and one BIN chunk."""
    js = json.dumps(gltf).encode("utf-8")
    js += b" " * (-len(js) % 4)
    bin_chunk = bin_data + b"\x00" * (-len(bin_data) % 4)
    total = 12 + 8 + len(js) + 8 + len(bin_chunk)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A))
        f.write(js)
        f.write(struct.pack("<II", len(bin_chunk), 0x004E4942))
        f.write(bin_chunk)
