"""Geometry helpers used by ``LaneBoundaryTracker`` - pure functions.

Basic poly/EMA helpers, shape-validation helpers, and small interpolation
helpers live here. Inference helpers (``infer_missing_side_coeffs``,
``infer_side_from_width_profile``, ``sample_width_profile``,
``lane_area_width_profile``, ``lane_width_diagnostic``,
``shifted_parallel_coeffs``) are re-exported from
``lane_boundary_tracker_geom_inference``.
"""
from __future__ import annotations

import numpy as np

from features.lane_boundary_tracker_geom_inference import (
    infer_missing_side_coeffs,
    infer_side_from_width_profile,
    infer_side_xs_from_width_profile,
    lane_area_width_profile,
    lane_width_diagnostic,
    sample_width_profile,
    shifted_parallel_coeffs,
)

__all__ = (
    "ema", "sample_coeffs", "max_abs_delta", "profile_error",
    "sample_polyline", "interp_edge", "sample_slope",
    "parallel_shape_errors", "expected_from_opposite_xs",
    "infer_missing_side_coeffs", "infer_side_from_width_profile",
    "infer_side_xs_from_width_profile",
    "sample_width_profile", "lane_area_width_profile",
    "lane_width_diagnostic", "shifted_parallel_coeffs",
)


def ema(prev: tuple[float, float, float],
        new: tuple[float, float, float],
        alpha: float) -> tuple[float, float, float]:
    return ((1.0 - alpha) * prev[0] + alpha * new[0],
            (1.0 - alpha) * prev[1] + alpha * new[1],
            (1.0 - alpha) * prev[2] + alpha * new[2])


def sample_coeffs(coeffs: tuple[float, float, float] | None,
                  ys: np.ndarray) -> np.ndarray | None:
    """Evaluate ``a*y^2 + b*y + c`` on ``ys``. ``None`` when no coeffs."""
    if coeffs is None:
        return None
    a, b, c = coeffs
    return (a * ys * ys + b * ys + c).astype(np.float32)


def max_abs_delta(candidate_xs: np.ndarray | None,
                  previous_xs: np.ndarray | None) -> float | None:
    """Per-y max absolute x delta between two same-length samples.
    Returns ``None`` when either side is missing or sizes differ."""
    if (candidate_xs is None or previous_xs is None
            or candidate_xs.size == 0
            or candidate_xs.size != previous_xs.size):
        return None
    return float(np.max(np.abs(candidate_xs - previous_xs)))


def profile_error(candidate_width: np.ndarray | None,
                  memory_width: np.ndarray | None) -> float | None:
    """Per-y max absolute width delta between candidate and remembered
    purple width profile. ``None`` when either is missing/empty."""
    if (candidate_width is None or memory_width is None
            or candidate_width.size == 0
            or candidate_width.size != memory_width.size):
        return None
    return float(np.max(np.abs(candidate_width - memory_width)))


def sample_polyline(
    coeffs: tuple[float, float, float] | None, frame_h: int,
    *, top_frac: float, num_points: int,
) -> np.ndarray:
    if coeffs is None:
        return np.empty((0, 2), dtype=np.float32)
    a, b, c = coeffs
    ys = np.linspace(max(0.0, frame_h * top_frac), float(frame_h - 1),
                     num=num_points, dtype=np.float32)
    xs = (a * ys * ys + b * ys + c).astype(np.float32)
    return np.stack([xs, ys], axis=1).astype(np.float32)


def interp_edge(edge_pts: np.ndarray | None, ys: np.ndarray) -> np.ndarray | None:
    if edge_pts is None or len(edge_pts) < 2:
        return None
    sorted_edge = edge_pts[np.argsort(edge_pts[:, 1])]
    return np.interp(ys, sorted_edge[:, 1], sorted_edge[:, 0])


def sample_slope(coeffs: tuple[float, float, float] | None,
                 ys: np.ndarray) -> np.ndarray | None:
    """Per-y first derivative ``2*a*y + b`` of a quadratic fit. ``None``
    when ``coeffs`` is missing. Used by the parallel-shape validator to
    compare local slope between left and right candidates."""
    if coeffs is None:
        return None
    a, b, _ = coeffs
    return (2.0 * float(a) * ys + float(b)).astype(np.float32)


def parallel_shape_errors(
    left_coeffs: tuple[float, float, float] | None,
    right_coeffs: tuple[float, float, float] | None,
    ys: np.ndarray,
) -> tuple[float | None, float | None]:
    """Return ``(curvature_delta, slope_delta_change)``.

    * ``curvature_delta = |a_left - a_right|`` - the second-derivative gap
      between the two quadratic fits. Zero for parallel lines.
    * ``slope_delta_change = max(slope_l - slope_r) - min(...)`` over the
      ROI grid - captures curvature mismatch even when the constant slope
      offset (perspective convergence) is large. Zero for parallel.

    ``(None, None)`` when either side or ``ys`` is empty.
    """
    if left_coeffs is None or right_coeffs is None or ys.size == 0:
        return None, None
    a_l, b_l, _ = left_coeffs
    a_r, b_r, _ = right_coeffs
    curvature_delta = float(abs(float(a_l) - float(a_r)))
    slope_l = 2.0 * float(a_l) * ys + float(b_l)
    slope_r = 2.0 * float(a_r) * ys + float(b_r)
    delta = slope_l - slope_r
    slope_delta_change = float(np.max(delta) - np.min(delta))
    return curvature_delta, slope_delta_change


def expected_from_opposite_xs(
    other_xs: np.ndarray | None,
    width_profile: np.ndarray | None,
    *, infer_for: str,
) -> np.ndarray | None:
    """Per-y expected x of the missing/suspicious side, derived by
    shifting the opposite side by the per-row purple width.

    ``infer_for == "left"``  -> ``other_xs - width_profile``.
    ``infer_for == "right"`` -> ``other_xs + width_profile``.
    ``None`` when sizes mismatch."""
    if (other_xs is None or width_profile is None
            or other_xs.size == 0
            or other_xs.size != width_profile.size):
        return None
    if infer_for == "left":
        return (other_xs - width_profile).astype(np.float32)
    return (other_xs + width_profile).astype(np.float32)
