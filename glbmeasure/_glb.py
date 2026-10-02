"""GLB container parsing: header, JSON chunk, BIN chunk access via mmap.

The whole file is memory-mapped so that large embedded resources (images)
never contribute to resident memory unless actually touched. Only the JSON
chunk (<= 2MiB per spec assumptions) and the bufferViews referenced by
measured accessors are ever read.
"""

from __future__ import annotations

import json
import mmap
import struct
from dataclasses import dataclass
from typing import BinaryIO

_GLB_MAGIC = 0x46546C67  # b"glTF"
_JSON_CHUNK_TYPE = 0x4E4F534A  # b"JSON"
_BIN_CHUNK_TYPE = 0x004E4942  # b"BIN\0"


@dataclass
class GlbFile:
    """An opened GLB file with parsed JSON and memory-mapped BIN access."""

    json: dict
    bin_base: int  # absolute file offset where the BIN chunk data starts
    bin_length: int
    _mmap: mmap.mmap
    _file: BinaryIO

    def bin_view(self, offset: int, length: int) -> memoryview:
        """Return a zero-copy view into the BIN chunk (offset relative to BIN start)."""
        if offset < 0 or length < 0 or offset + length > self.bin_length:
            raise ValueError("bufferView range outside BIN chunk")
        start = self.bin_base + offset
        return memoryview(self._mmap)[start : start + length]

    def close(self) -> None:
        self._mmap.close()
        self._file.close()

    def __enter__(self) -> "GlbFile":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def open_glb(path: str) -> GlbFile:
    """Open a GLB file, parse its JSON chunk, and mmap the BIN chunk."""
    file = open(path, "rb")
    try:
        header = file.read(12)
        if len(header) != 12:
            raise ValueError("file too small for GLB header")
        magic, _version, _total_length = struct.unpack("<III", header)
        if magic != _GLB_MAGIC:
            raise ValueError("not a GLB file (bad magic)")

        chunk_header = file.read(8)
        if len(chunk_header) != 8:
            raise ValueError("missing first chunk header")
        json_length, json_type = struct.unpack("<II", chunk_header)
        if json_type != _JSON_CHUNK_TYPE:
            raise ValueError("first chunk is not JSON")
        json_bytes = file.read(json_length)
        if len(json_bytes) != json_length:
            raise ValueError("truncated JSON chunk")
        gltf = json.loads(json_bytes)

        bin_base = 0
        bin_length = 0
        chunk_header = file.read(8)
        if len(chunk_header) == 8:
            bin_length, bin_type = struct.unpack("<II", chunk_header)
            if bin_type != _BIN_CHUNK_TYPE:
                raise ValueError("second chunk is not BIN")
            bin_base = file.tell()

        mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
    except Exception:
        file.close()
        raise
    return GlbFile(json=gltf, bin_base=bin_base, bin_length=bin_length, _mmap=mm, _file=file)
