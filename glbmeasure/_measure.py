"""Triangle counting and world-space AABB measurement.

Only vertices that actually participate in triangles contribute to the
bounding box, and the box is computed after applying full world transforms,
so rotations never inflate the result. Vertex transformation runs in chunks
so peak memory stays bounded regardless of mesh size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ._accessor import read_accessor
from ._glb import GlbFile
from ._scene import iter_mesh_instances

_TRIANGLES = 4
_TRIANGLE_STRIP = 5
_TRIANGLE_FAN = 6

_M_TO_INCH = 1.0 / 0.0254
_CHUNK_VERTICES = 1 << 18  # transform at most 256K vertices per chunk


@dataclass
class MeasureResult:
    """triangle_count: exact number of triangles drawn.
    bounds: ((min_x, min_y, min_z), (max_x, max_y, max_z)) in inches, or None
    when the scene contains no triangles."""

    triangle_count: int
    bounds: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]


def _participating_indices(mode: int, index_count: int) -> Tuple[int, int]:
    """Return (triangle_count, participating_vertex_reference_count).

    The second value is how many leading index references take part in at
    least one triangle; for strips and fans every referenced vertex
    participates once there are >= 3 references.
    """
    if mode == _TRIANGLES:
        full = index_count // 3
        return full, full * 3
    if mode in (_TRIANGLE_STRIP, _TRIANGLE_FAN):
        if index_count < 3:
            return 0, 0
        return index_count - 2, index_count
    return 0, 0


def _accumulate_bounds(
    positions: np.ndarray, world: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform positions by world and fold them into the running min/max."""
    linear = world[:3, :3]
    offset = world[:3, 3]
    for start in range(0, len(positions), _CHUNK_VERTICES):
        chunk = np.asarray(positions[start : start + _CHUNK_VERTICES], dtype=np.float64)
        transformed = chunk @ linear.T + offset
        lo = np.minimum(lo, transformed.min(axis=0))
        hi = np.maximum(hi, transformed.max(axis=0))
    return lo, hi


def measure_glb(glb: GlbFile) -> MeasureResult:
    """Measure the default (or first) scene of an opened GLB file."""
    gltf = glb.json
    scene_index = gltf.get("scene", 0)
    if not gltf.get("scenes"):
        return MeasureResult(triangle_count=0, bounds=None)

    triangle_count = 0
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)

    for mesh_index, world in iter_mesh_instances(gltf, scene_index):
        mesh = gltf["meshes"][mesh_index]
        for primitive in mesh.get("primitives", []):
            mode = primitive.get("mode", _TRIANGLES)
            if mode not in (_TRIANGLES, _TRIANGLE_STRIP, _TRIANGLE_FAN):
                continue
            if "POSITION" not in primitive.get("attributes", {}):
                continue

            if "indices" in primitive:
                indices = read_accessor(glb, primitive["indices"])
                tris, used = _participating_indices(mode, len(indices))
                triangle_count += tris
                if used == 0:
                    continue
                positions = read_accessor(glb, primitive["attributes"]["POSITION"])
                gathered = positions.take(indices[:used], axis=0)
                lo, hi = _accumulate_bounds(gathered, world, lo, hi)
            else:
                positions = read_accessor(glb, primitive["attributes"]["POSITION"])
                tris, used = _participating_indices(mode, len(positions))
                triangle_count += tris
                if used == 0:
                    continue
                lo, hi = _accumulate_bounds(positions[:used], world, lo, hi)

    if triangle_count == 0:
        return MeasureResult(triangle_count=0, bounds=None)

    bounds = (
        (float(lo[0] * _M_TO_INCH), float(lo[1] * _M_TO_INCH), float(lo[2] * _M_TO_INCH)),
        (float(hi[0] * _M_TO_INCH), float(hi[1] * _M_TO_INCH), float(hi[2] * _M_TO_INCH)),
    )
    return MeasureResult(triangle_count=triangle_count, bounds=bounds)
