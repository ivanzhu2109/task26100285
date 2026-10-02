"""Minimal 4x4 matrix helpers for node transform composition."""

from typing import Sequence, Tuple

Mat4 = Tuple[float, ...]  # 16 values, column-major like glTF

IDENTITY: Mat4 = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


def compose_trs(translation, rotation, scale) -> Mat4:
    """Build a column-major matrix from glTF TRS values."""
    tx, ty, tz = translation
    qx, qy, qz, qw = rotation
    sx, sy, sz = scale

    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, xy, xz = qx * x2, qx * y2, qx * z2
    yy, yz, zz = qy * y2, qy * z2, qz * z2
    wx, wy, wz = qw * x2, qw * y2, qw * z2

    return (
        (1.0 - (yy + zz)) * sx, (xy + wz) * sx, (xz - wy) * sx, 0.0,
        (xy - wz) * sy, (1.0 - (xx + zz)) * sy, (yz + wx) * sy, 0.0,
        (xz + wy) * sz, (yz - wx) * sz, (1.0 - (xx + yy)) * sz, 0.0,
        tx, ty, tz, 1.0,
    )


def multiply(a: Mat4, b: Mat4) -> Mat4:
    """Return a * b for column-major 4x4 matrices."""
    out = [0.0] * 16
    for col in range(4):
        for row in range(4):
            out[col * 4 + row] = (
                a[row] * b[col * 4]
                + a[4 + row] * b[col * 4 + 1]
                + a[8 + row] * b[col * 4 + 2]
                + a[12 + row] * b[col * 4 + 3]
            )
    return tuple(out)


def transform_point(m: Mat4, p: Sequence[float]) -> Tuple[float, float, float]:
    """Transform a point by a column-major 4x4 matrix."""
    x, y, z = p
    return (
        m[0] * x + m[4] * y + m[8] * z + m[12],
        m[1] * x + m[5] * y + m[9] * z + m[13],
        m[2] * x + m[6] * y + m[10] * z + m[14],
    )
