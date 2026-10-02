"""GLB container parsing: JSON chunk plus memory-mapped BIN chunk."""

import json as _json
import mmap
import os
import struct
from dataclasses import dataclass
from typing import Optional

_GLB_MAGIC = 0x46546C67  # "glTF"
_CHUNK_JSON = 0x4E4F534A  # "JSON"
_CHUNK_BIN = 0x004E4942  # "BIN\0"


@dataclass
class GlbFile:
    """Parsed GLB container.

    ``json`` holds the decoded JSON chunk; ``bin_map`` is a read-only
    memory map of the whole file together with the offset of the BIN
    chunk, so large embedded images never need to be loaded.
    """

    json: dict
    bin_map: Optional[object]
    bin_offset: int = 0
    _file: Optional[object] = None

    def close(self) -> None:
        if self.bin_map is not None:
            self.bin_map.close()
            self.bin_map = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def slice_at(self, offset: int, length: int) -> memoryview:
        """Return a view at an absolute mmap offset without copying."""
        return memoryview(self.bin_map)[offset:offset + length]


def open_glb(path: str) -> GlbFile:
    """Open a GLB file and return its parsed container."""
    fileobj = open(path, "rb")
    try:
        header = fileobj.read(12)
        if len(header) != 12:
            raise ValueError("truncated GLB header")
        magic, _version, _total = struct.unpack("<III", header)
        if magic != _GLB_MAGIC:
            raise ValueError("not a GLB file")

        glb_json = None
        bin_offset = 0
        while True:
            chunk_header = fileobj.read(8)
            if len(chunk_header) != 8:
                break
            chunk_length, chunk_type = struct.unpack("<II", chunk_header)
            if chunk_type == _CHUNK_JSON:
                data = fileobj.read(chunk_length)
                glb_json = _json.loads(data.decode("utf-8"))
            elif chunk_type == _CHUNK_BIN:
                bin_offset = fileobj.tell()
                fileobj.seek(chunk_length, os.SEEK_CUR)
            else:
                fileobj.seek(chunk_length, os.SEEK_CUR)

        if glb_json is None:
            raise ValueError("GLB file has no JSON chunk")

        bin_map = None
        if bin_offset:
            bin_map = mmap.mmap(fileobj.fileno(), 0, access=mmap.ACCESS_READ)

        return GlbFile(json=glb_json, bin_map=bin_map,
                       bin_offset=bin_offset, _file=fileobj)
    except Exception:
        fileobj.close()
        raise
