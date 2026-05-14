"""Validation diagnostic merge into the tracker output dict."""
from __future__ import annotations

from features.lane_boundary_tracker_validate_config import CandidateValidationResult


def merge_validation_diagnostics(
    out: dict,
    left: CandidateValidationResult,
    right: CandidateValidationResult,
) -> None:
    """Add validation diagnostic fields to the tracker output dict in place."""
    out["left_detect_valid"] = bool(left.accepted)
    out["right_detect_valid"] = bool(right.accepted)
    out["left_detect_reject_reason"] = left.reason
    out["right_detect_reject_reason"] = right.reason
    out["left_memory_error_px"] = left.memory_error_px
    out["right_memory_error_px"] = right.memory_error_px
    out["purple_width_error_px"] = (
        left.purple_width_error_px if left.purple_width_error_px is not None
        else right.purple_width_error_px
    )
    out["corridor_pinch_reject"] = bool(
        left.pinch_detected or right.pinch_detected,
    )
    out["left_opposite_memory_error_px"] = left.opposite_memory_error_px
    out["right_opposite_memory_error_px"] = right.opposite_memory_error_px
    out["left_support_weak"] = bool(left.support_weak)
    out["right_support_weak"] = bool(right.support_weak)
    out["left_parallel_slope_error"] = left.parallel_slope_error
    out["right_parallel_slope_error"] = right.parallel_slope_error
    out["left_parallel_curvature_error"] = left.parallel_curvature_error
    out["right_parallel_curvature_error"] = right.parallel_curvature_error
    out["left_expected_from_opposite_error_px"] = (
        left.expected_from_opposite_error_px)
    out["right_expected_from_opposite_error_px"] = (
        right.expected_from_opposite_error_px)
    out["left_parallel_reject"] = bool(left.parallel_reject)
    out["right_parallel_reject"] = bool(right.parallel_reject)
