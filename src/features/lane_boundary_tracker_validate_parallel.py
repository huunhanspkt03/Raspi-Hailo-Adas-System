"""Parallel-shape and opposite-anchored override gates.

Pure helpers that ``validate_pair`` calls after the per-side and
cross-memory checks. Split out so ``lane_boundary_tracker_validate``
stays focused on the per-side rules and the public entry points.

Both gates require a strong/weak split between the two candidates —
without that there is no reliable anchor to drop the suspicious side.
"""
from __future__ import annotations

import numpy as np

from features.lane_boundary_tracker_geom import (
    expected_from_opposite_xs, max_abs_delta, parallel_shape_errors,
)

__all__ = (
    "apply_strong_side_copy_gate",
    "apply_parallel_shape_gate",
    "apply_opposite_anchored_gate",
)


def apply_strong_side_copy_gate(
    left, right, *,
    l_n: int, r_n: int,
    config,
    replace_reject,
    reject_reason: str,
):
    """Reject the weaker side when exactly one side has strong support.

    This is the user's current primary policy for one-sided confidence:
    when one purple side is clearly strong and the opposite side is weaker,
    do not let the weaker side fit its own independent curve. Reject it so
    ``resolve_side`` copies/shifts from the strong opposite side using
    purple memory or the lane-area bootstrap width.
    """
    if not (left.accepted and right.accepted):
        return left, right

    strong_n = int(config.min_strong_support_pixels)
    l_strong = l_n >= strong_n
    r_strong = r_n >= strong_n
    if r_strong and not l_strong and l_n < r_n:
        left = replace_reject(
            left, reject_reason, mark_parallel_reject=True,
        )
    elif l_strong and not r_strong and r_n < l_n:
        right = replace_reject(
            right, reject_reason, mark_parallel_reject=True,
        )
    return left, right


def apply_parallel_shape_gate(
    left, right, *,
    left_candidate, right_candidate,
    l_n: int, r_n: int,
    ys: np.ndarray,
    config,
    replace_reject,
    reject_reason: str,
):
    """Compute parallel-shape diagnostics on both results, then reject
    the weaker side when one is strong, the other is weaker, and the
    pair fails the curvature/slope thresholds.

    ``replace_reject(result, reason, *, mark_parallel_reject=True)`` is
    the caller-supplied helper that flips a result to rejected while
    preserving diagnostics. ``reject_reason`` is the constant the caller
    wants stamped onto a parallel-shape rejection.
    """
    cur_delta, slope_delta_change = parallel_shape_errors(
        left_candidate, right_candidate, ys,
    )
    left.parallel_curvature_error = cur_delta
    right.parallel_curvature_error = cur_delta
    left.parallel_slope_error = slope_delta_change
    right.parallel_slope_error = slope_delta_change

    strong_n = int(config.min_strong_support_pixels)
    l_strong = l_n >= strong_n
    r_strong = r_n >= strong_n
    not_parallel = (
        cur_delta is not None and slope_delta_change is not None
        and (cur_delta > float(config.max_parallel_curvature_delta)
             or slope_delta_change > float(config.max_parallel_slope_delta))
    )
    if not_parallel and left.accepted and right.accepted:
        if l_strong and not r_strong and r_n < l_n:
            right = replace_reject(right, reject_reason,
                                   mark_parallel_reject=True)
        elif r_strong and not l_strong and l_n < r_n:
            left = replace_reject(left, reject_reason,
                                  mark_parallel_reject=True)
    return left, right


def apply_opposite_anchored_gate(
    left, right, *,
    cand_lx: np.ndarray | None, cand_rx: np.ndarray | None,
    l_n: int, r_n: int,
    width_profile: np.ndarray | None,
    config,
    replace_reject,
    reject_reason: str,
):
    """Reject the weaker side when its candidate deviates from the
    position predicted by ``opposite_xs ± width_profile``.

    ``width_profile`` is the best available width source — purple
    memory if the caller has it, otherwise a lane_area-derived
    bootstrap profile. When ``None`` the gate is skipped, but the
    diagnostic ``expected_from_opposite_error_px`` is left at ``None``.
    """
    if (width_profile is None or cand_lx is None or cand_rx is None
            or width_profile.size != cand_lx.size):
        return left, right

    expected_lx = expected_from_opposite_xs(
        cand_rx, width_profile, infer_for="left")
    expected_rx = expected_from_opposite_xs(
        cand_lx, width_profile, infer_for="right")
    left.expected_from_opposite_error_px = max_abs_delta(cand_lx, expected_lx)
    right.expected_from_opposite_error_px = max_abs_delta(cand_rx, expected_rx)

    if not (left.accepted and right.accepted):
        return left, right

    strong_n = int(config.min_strong_support_pixels)
    l_strong = l_n >= strong_n
    r_strong = r_n >= strong_n
    max_err = float(config.max_expected_from_opposite_px)
    l_err = left.expected_from_opposite_error_px
    r_err = right.expected_from_opposite_error_px
    if (l_strong and not r_strong
            and r_err is not None and r_err > max_err):
        right = replace_reject(right, reject_reason, mark_parallel_reject=True)
    elif (r_strong and not l_strong
          and l_err is not None and l_err > max_err):
        left = replace_reject(left, reject_reason, mark_parallel_reject=True)
    return left, right
