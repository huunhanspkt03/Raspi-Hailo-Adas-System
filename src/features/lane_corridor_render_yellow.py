"""Ego-fixed yellow corridor and centerline shape for the render corridor."""
from __future__ import annotations

import numpy as np


def smooth_widths(widths: np.ndarray, taps: int) -> np.ndarray:
    """Edge-padded rolling mean across ROI rows.

    Keeps yellow gap stable row-to-row even when purple width has small
    polynomial wobble. ``taps <= 1`` is a no-op.
    """
    if taps <= 1 or widths.shape[0] < 3:
        return widths.astype(np.float32)
    pad = int(taps) // 2
    kernel = np.ones(int(taps), dtype=np.float32) / float(taps)
    padded = np.pad(widths.astype(np.float32), pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def yellow_corridor_from_center(
    left_xs: np.ndarray, right_xs: np.ndarray, center_xs: np.ndarray,
    frame_w: int,
    fixed_half_width_far_frac: float,
    fixed_half_width_near_frac: float,
    min_inner_gap_px: float,
    smooth_taps: int,
    red_dxdy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Yellow/cam guidance lines symmetric around ``center_xs``.

    Purple left/right are intentionally ignored for the yellow width.
    They remain in the signature only because the caller already
    resolves them for purple rendering. The half-width is a fixed
    camera-anchored perspective profile (narrow far, wide near). Yellow
    is offset along the normal of red when ``red_dxdy`` is supplied so
    a curved red produces a yellow corridor that bends with it; if
    ``red_dxdy`` is ``None`` (straight fallback) yellow is a symmetric
    horizontal offset:

        yellow_left(y)  = center_x(y) - half_width(y) * normal_x
        yellow_right(y) = center_x(y) + half_width(y) * normal_x

    where ``normal_x = 1 / sqrt(1 + (dx/dy)^2)``. For a straight red
    this reduces to ``normal_x = 1`` and the corridor is symmetric in
    the horizontal direction.
    """
    _ = left_xs, right_xs
    n = int(center_xs.shape[0])
    far = max(float(fixed_half_width_far_frac) * float(frame_w),
              float(min_inner_gap_px) * 0.5)
    near = max(float(fixed_half_width_near_frac) * float(frame_w),
               float(min_inner_gap_px) * 0.5)
    if n <= 1:
        half_w = np.full_like(center_xs, near, dtype=np.float32)
    else:
        t = np.linspace(0.0, 1.0, num=n, dtype=np.float32)
        half_w = (far + (near - far) * t).astype(np.float32)
    half_w = smooth_widths(half_w, int(smooth_taps))

    if (red_dxdy is not None
            and red_dxdy.size == center_xs.size
            and red_dxdy.size > 0):
        inv_norm = (1.0
                    / np.sqrt(1.0
                              + red_dxdy.astype(np.float32) ** 2)
                    ).astype(np.float32)
        offset = (half_w * inv_norm).astype(np.float32)
    else:
        offset = half_w
    yl = (center_xs - offset).astype(np.float32)
    yr = (center_xs + offset).astype(np.float32)
    yl = np.clip(yl, 0.0, float(frame_w - 1)).astype(np.float32)
    yr = np.clip(yr, 0.0, float(frame_w - 1)).astype(np.float32)

    realized_near = float(yr[-1] - yl[-1]) if yl.size > 0 else 0.0
    realized_far = float(yr[0] - yl[0]) if yl.size > 0 else 0.0
    return yl, yr, realized_near, realized_far


def ego_path_xs(ys: np.ndarray, frame_w: int) -> tuple[np.ndarray, str, str]:
    """Straight ego-anchored centerline fallback.

    Returns ``center_xs == frame_w / 2`` for every sampled ``y``. Used
    when no reliable purple curve is available - red stays vertically
    through the camera column.
    """
    ego_x = 0.5 * float(frame_w)
    center_xs = np.full_like(ys, ego_x, dtype=np.float32)
    return center_xs, "fixed_center", "no_reliable_purple"
