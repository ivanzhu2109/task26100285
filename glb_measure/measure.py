"""Scene traversal and geometry accumulation."""

from dataclasses import dataclass
from typing import Optional, Tuple

from .accessors import read_indices, read_positions
from .glb import open_glb
from .math3d import IDENTITY, compose_trs, multiply, transform_point

_MODE_TRIANGLES = 4
_MODE_TRIANGLE_STRIP = 5
_MODE_TRIANGLE_FAN = 6

_METERS_TO_INCHES = 1.0 / 0.0254


@dataclass
class BBox:
    """Axis-aligned bounding box in inches (world space)."""

    min: Tuple[float, float, float]
    max: Tuple[float, float, float]


@dataclass
class MeasureResult:
    """Measurement outcome for a GLB asset."""

    triangle_count: int
    bbox: Optional[BBox]  # None when no triangles were measured


def _node_local_matrix(node: dict):
    """Local transform of a node: explicit matrix or TRS."""
    if "matrix" in node:
        return tuple(node["matrix"])
    return compose_trs(
        node.get("translation", (0.0, 0.0, 0.0)),
        node.get("rotation", (0.0, 0.0, 0.0, 1.0)),
        node.get("scale", (1.0, 1.0, 1.0)),
    )


class _Accumulator:
    """Running triangle count and world-space AABB (meters)."""

    def __init__(self) -> None:
        self.triangles = 0
        self.min = [float("inf")] * 3
        self.max = [float("-inf")] * 3

    def add_primitive(self, glb, primitive: dict, world) -> None:
        mode = primitive.get("mode", _MODE_TRIANGLES)
        if mode not in (_MODE_TRIANGLES, _MODE_TRIANGLE_STRIP,
                        _MODE_TRIANGLE_FAN):
            return

        accessors = glb.json["accessors"]
        if "indices" in primitive:
            sequence = read_indices(glb, accessors[primitive["indices"]])
        else:
            position_accessor = accessors[primitive["attributes"]["POSITION"]]
            sequence = range(position_accessor["count"])

        length = len(sequence)
        if mode == _MODE_TRIANGLES:
            triangle_count = length // 3
        else:
            triangle_count = max(length - 2, 0)
        if triangle_count == 0:
            return

        positions = read_positions(
            glb, accessors[primitive["attributes"]["POSITION"]])

        self.triangles += triangle_count
        for vertex_index in sequence:
            base = vertex_index * 3
            point = transform_point(
                world, (positions[base], positions[base + 1],
                        positions[base + 2]))
            for axis in range(3):
                if point[axis] < self.min[axis]:
                    self.min[axis] = point[axis]
                if point[axis] > self.max[axis]:
                    self.max[axis] = point[axis]


def _walk_node(glb, node_index: int, parent_world, acc: _Accumulator) -> None:
    node = glb.json["nodes"][node_index]
    world = multiply(parent_world, _node_local_matrix(node))

    if "mesh" in node:
        mesh = glb.json["meshes"][node["mesh"]]
        for primitive in mesh.get("primitives", []):
            acc.add_primitive(glb, primitive, world)

    for child_index in node.get("children", []):
        _walk_node(glb, child_index, world, acc)


def measure_glb(path: str) -> MeasureResult:
    """Measure the GLB file at ``path``.

    Returns the total triangle count of the default (or first) scene
    and its world-space AABB in inches.  Derived data is not stored.
    """
    glb = open_glb(path)
    try:
        gltf = glb.json
        scene_index = gltf.get("scene", 0)
        scene = gltf["scenes"][scene_index]

        acc = _Accumulator()
        for root_index in scene.get("nodes", []):
            _walk_node(glb, root_index, IDENTITY, acc)

        if acc.triangles == 0:
            return MeasureResult(triangle_count=0, bbox=None)

        bbox = BBox(
            min=tuple(v * _METERS_TO_INCHES for v in acc.min),
            max=tuple(v * _METERS_TO_INCHES for v in acc.max),
        )
        return MeasureResult(triangle_count=acc.triangles, bbox=bbox)
    finally:
        glb.close()
