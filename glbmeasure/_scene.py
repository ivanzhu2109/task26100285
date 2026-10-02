"""Scene graph traversal and node transform composition."""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np


def _quat_to_matrix(quat: list[float]) -> np.ndarray:
    x, y, z, w = quat
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n == 0.0:
        x, y, z, w = 0.0, 0.0, 0.0, 1.0
    else:
        x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def node_matrix(node: dict) -> np.ndarray:
    """Return the 4x4 local transform of a node (matrix or TRS)."""
    if "matrix" in node:
        # glTF matrices are column-major; reshape then transpose to row-major.
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T

    matrix = np.eye(4, dtype=np.float64)
    rotation = _quat_to_matrix(node.get("rotation", [0.0, 0.0, 0.0, 1.0]))
    scale = node.get("scale", [1.0, 1.0, 1.0])
    matrix[:3, :3] = rotation * np.asarray(scale, dtype=np.float64)
    matrix[:3, 3] = node.get("translation", [0.0, 0.0, 0.0])
    return matrix


def iter_mesh_instances(
    gltf: dict, scene_index: int
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (mesh_index, world_matrix) for every node in the scene hierarchy."""
    nodes = gltf.get("nodes", [])
    scene = gltf["scenes"][scene_index]

    def visit(node_index: int, parent: np.ndarray) -> Iterator[Tuple[int, np.ndarray]]:
        node = nodes[node_index]
        world = parent @ node_matrix(node)
        if "mesh" in node:
            yield node["mesh"], world
        for child in node.get("children", []):
            yield from visit(child, world)

    identity = np.eye(4, dtype=np.float64)
    for root in scene.get("nodes", []):
        yield from visit(root, identity)
