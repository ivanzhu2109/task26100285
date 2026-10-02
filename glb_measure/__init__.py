"""GLB static geometry measurement library.

Measures the total triangle count and the world-space axis-aligned
bounding box (in inches) of the geometry actually drawn by a glTF 2.0
GLB asset's default (or first) scene.
"""

from .measure import BBox, MeasureResult, measure_glb

__all__ = ["BBox", "MeasureResult", "measure_glb"]
