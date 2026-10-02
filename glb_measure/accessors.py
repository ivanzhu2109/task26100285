"""glTF accessor decoding: bufferView offsets, byteStride, sparse data."""

import struct
from array import array

_COMPONENT_FORMAT = {
    5120: "b",  # BYTE
    5121: "B",  # UNSIGNED_BYTE
    5122: "h",  # SHORT
    5123: "H",  # UNSIGNED_SHORT
    5125: "I",  # UNSIGNED_INT
    5126: "f",  # FLOAT
}

_COMPONENT_SIZE = {code: struct.calcsize(fmt)
                   for code, fmt in _COMPONENT_FORMAT.items()}

_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT4": 16,
}


def _buffer_view_base(glb, buffer_view: dict) -> int:
    """Absolute mmap offset of a bufferView (BIN chunk is buffer 0)."""
    return glb.bin_offset + buffer_view.get("byteOffset", 0)


def _read_tightly_packed(glb, offset: int, component_type: int,
                         count: int) -> array:
    """Read ``count`` tightly packed components from the BIN chunk."""
    fmt = _COMPONENT_FORMAT[component_type]
    size = _COMPONENT_SIZE[component_type] * count
    view = glb.slice_at(offset, size)
    out = array(fmt)
    out.frombytes(view)
    return out


def _read_base(glb, accessor: dict, num_components: int) -> array:
    """Read the base (non-sparse) data of an accessor.

    Handles accessor/bufferView byte offsets and interleaved
    ``byteStride``.  Without a base bufferView, returns zeros.
    """
    component_type = accessor["componentType"]
    fmt = _COMPONENT_FORMAT[component_type]
    count = accessor["count"]
    total = count * num_components

    if "bufferView" not in accessor:
        return array(fmt, bytes(_COMPONENT_SIZE[component_type] * total))

    buffer_view = glb.json["bufferViews"][accessor["bufferView"]]
    base = (_buffer_view_base(glb, buffer_view)
            + accessor.get("byteOffset", 0))
    packed = _COMPONENT_SIZE[component_type] * num_components
    stride = buffer_view.get("byteStride", packed)

    if stride == packed:
        return _read_tightly_packed(glb, base, component_type, total)

    element_fmt = "<" + fmt * num_components
    out = array(fmt)
    for i in range(count):
        values = struct.unpack_from(element_fmt, glb.bin_map, base + i * stride)
        out.extend(values)
    return out


def _apply_sparse(glb, accessor: dict, base: array,
                  num_components: int) -> array:
    """Overlay sparse values onto base data, in place."""
    sparse = accessor.get("sparse")
    if sparse is None:
        return base

    component_type = accessor["componentType"]
    count = sparse["count"]

    index_view = glb.json["bufferViews"][sparse["indices"]["bufferView"]]
    index_type = sparse["indices"]["componentType"]
    index_offset = (_buffer_view_base(glb, index_view)
                    + sparse["indices"].get("byteOffset", 0))
    sparse_indices = _read_tightly_packed(glb, index_offset, index_type, count)

    value_view = glb.json["bufferViews"][sparse["values"]["bufferView"]]
    value_offset = (_buffer_view_base(glb, value_view)
                    + sparse["values"].get("byteOffset", 0))
    sparse_values = _read_tightly_packed(
        glb, value_offset, component_type, count * num_components)

    for k in range(count):
        dst = sparse_indices[k] * num_components
        src = k * num_components
        base[dst:dst + num_components] = sparse_values[src:src + num_components]
    return base


def read_accessor(glb, accessor: dict) -> array:
    """Decode an accessor to a flat array of its component type."""
    num_components = _TYPE_COMPONENTS[accessor["type"]]
    base = _read_base(glb, accessor, num_components)
    return _apply_sparse(glb, accessor, base, num_components)


def read_positions(glb, accessor: dict) -> array:
    """Decode a FLOAT/VEC3 POSITION accessor to a flat xyz array."""
    return read_accessor(glb, accessor)


def read_indices(glb, accessor: dict) -> array:
    """Decode a scalar unsigned-integer index accessor."""
    return read_accessor(glb, accessor)
