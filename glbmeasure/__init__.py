"""glbmeasure: static geometry measurement for glTF 2.0 binary (GLB) assets.

Public API:
    measure(path) -> MeasureResult
"""

from ._glb import open_glb
from ._measure import MeasureResult, measure_glb


def measure(path: str) -> MeasureResult:
    """Measure a GLB file: total triangle count and world-space AABB in inches."""
    with open_glb(path) as glb:
        return measure_glb(glb)


__all__ = ["measure", "MeasureResult", "measure_glb", "open_glb"]
