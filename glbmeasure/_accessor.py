"""glTF accessor decoding: bufferView offsets, byteStride, sparse overrides.

Dense accessors are exposed as zero-copy views into the memory-mapped BIN
chunk whenever possible, so resident memory only grows with data actually
consumed by the measurement, never with unrelated embedded resources.
"""

from __future__ import annotations

import numpy as np

from ._glb import GlbFile

_COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def _buffer_view_slice(glb: GlbFile, buffer_view_index: int) -> tuple[memoryview, int]:
    """Return (view, stride) for a bufferView; stride is 0 when tightly packed."""
    buffer_views = glb.json.get("bufferViews", [])
    bv = buffer_views[buffer_view_index]
    if bv.get("buffer", 0) != 0:
        raise ValueError("only the embedded BIN buffer is supported")
    offset = bv.get("byteOffset", 0)
    length = bv["byteLength"]
    return glb.bin_view(offset, length), bv.get("byteStride", 0)


def _read_dense(glb: GlbFile, accessor: dict) -> np.ndarray:
    count = accessor["count"]
    dtype = np.dtype(_COMPONENT_DTYPES[accessor["componentType"]])
    ncomp = _TYPE_COMPONENTS[accessor["type"]]
    shape = (count,) if ncomp == 1 else (count, ncomp)

    if "bufferView" not in accessor:
        # No base data: all elements are zero (only legal with sparse).
        return np.zeros(shape, dtype=dtype)

    view, stride = _buffer_view_slice(glb, accessor["bufferView"])
    byte_offset = accessor.get("byteOffset", 0)
    tight_stride = dtype.itemsize * ncomp
    if stride and stride != tight_stride:
        # Interleaved layout: build a strided view, no copy.
        return np.ndarray(
            shape=shape,
            dtype=dtype,
            buffer=view,
            offset=byte_offset,
            strides=(stride, dtype.itemsize),
        )
    nbytes = count * tight_stride
    arr = np.frombuffer(view, dtype=dtype, count=count * ncomp, offset=byte_offset)
    return arr.reshape(shape) if ncomp > 1 else arr


def _apply_sparse(glb: GlbFile, accessor: dict, base: np.ndarray) -> np.ndarray:
    sparse = accessor["sparse"]
    sparse_count = sparse["count"]

    idx_info = sparse["indices"]
    idx_dtype = np.dtype(_COMPONENT_DTYPES[idx_info["componentType"]])
    idx_view, _ = _buffer_view_slice(glb, idx_info["bufferView"])
    indices = np.frombuffer(
        idx_view, dtype=idx_dtype, count=sparse_count, offset=idx_info.get("byteOffset", 0)
    )

    val_info = sparse["values"]
    val_view, _ = _buffer_view_slice(glb, val_info["bufferView"])
    dtype = base.dtype
    ncomp = 1 if base.ndim == 1 else base.shape[1]
    values = np.frombuffer(
        val_view, dtype=dtype, count=sparse_count * ncomp, offset=val_info.get("byteOffset", 0)
    )
    if ncomp > 1:
        values = values.reshape(sparse_count, ncomp)

    result = base.copy()  # sparse overrides need a writable array
    result[indices] = values
    return result


def read_accessor(glb: GlbFile, accessor_index: int) -> np.ndarray:
    """Decode an accessor into an ndarray, honoring stride and sparse data."""
    accessor = glb.json["accessors"][accessor_index]
    base = _read_dense(glb, accessor)
    if "sparse" in accessor:
        return _apply_sparse(glb, accessor, base)
    return base
