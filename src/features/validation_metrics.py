"""Per-frame lane validation metrics - pure extraction.

Off by default. Live wiring is via ``validation_metrics_logger.MetricsLogger``;
this module contains only the pure ``extract_frame_metrics`` function so it
can be unit-tested without touching the filesystem.

Reads (without mutating) the existing ``lane_bounds`` tracker dict,
``red_center`` fusion dict, optional ``render_corridor`` dict, and a
``LaneState`` / ``PerceptionState``. Robust to ``None`` / partial inputs
so the caller can wire it once and not gate per-mode.
"""
from __future__ import annotations

from typing import Any

from features.validation_metrics_types import ComponentCounts, FrameMetrics

__all__ = [
    "ComponentCounts", "FrameMetrics", "extract_frame_metrics",
]


def _f(v: Any) -> float | None:
    """Safe float coercion. ``None`` passes through; non-numeric -> ``None``."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _component_counts(lane_bounds: dict | None) -> ComponentCounts:
    counts = ComponentCounts()
    if not isinstance(lane_bounds, dict):
        return counts
    components = lane_bounds.get("components")
    if not components:
        return counts
    rejected: dict[str, int] = {}
    for comp in components:
        reason = getattr(comp, "rejected_reason", None)
        if reason:
            counts.rejected_total += 1
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        side = getattr(comp, "side", "unknown")
        if side == "left":
            counts.accepted_left += 1
        elif side == "right":
            counts.accepted_right += 1
        else:
            counts.accepted_unknown += 1
    counts.rejected_by_reason = rejected
    return counts


def _fill_lane_bounds(m: FrameMetrics, lane_bounds: dict) -> None:
    m.lane_state = lane_bounds.get("lane_state")
    m.boundary_confidence = _f(lane_bounds.get("boundary_confidence"))
    m.left_source = lane_bounds.get("left_source")
    m.right_source = lane_bounds.get("right_source")
    m.left_inference_kind = lane_bounds.get("left_inference_kind")
    m.right_inference_kind = lane_bounds.get("right_inference_kind")
    m.left_confidence_final = _f(lane_bounds.get("left_confidence_final"))
    m.right_confidence_final = _f(lane_bounds.get("right_confidence_final"))
    m.purple_lane_width_px = _f(lane_bounds.get("purple_lane_width_px"))
    m.purple_lane_width_valid = bool(
        lane_bounds.get("purple_lane_width_valid", False))
    m.purple_lane_width_source = lane_bounds.get("purple_lane_width_source")
    if "purple_memory_update_allowed" in lane_bounds:
        m.purple_memory_update_allowed = bool(
            lane_bounds.get("purple_memory_update_allowed"))
    m.purple_memory_update_reject_reason = lane_bounds.get(
        "purple_memory_update_reject_reason")
    m.lane_width_px = _f(lane_bounds.get("lane_width_px"))
    m.lane_width_valid = bool(lane_bounds.get("lane_width_valid", False))
    m.lane_width_source = lane_bounds.get("lane_width_source")
    m.lane_area_width_px = _f(lane_bounds.get("lane_area_width_px"))
    m.lane_area_width_valid = bool(
        lane_bounds.get("lane_area_width_valid", False))
    if "left_detect_valid" in lane_bounds:
        m.left_detect_valid = bool(lane_bounds.get("left_detect_valid"))
    if "right_detect_valid" in lane_bounds:
        m.right_detect_valid = bool(lane_bounds.get("right_detect_valid"))
    m.left_detect_reject_reason = lane_bounds.get("left_detect_reject_reason")
    m.right_detect_reject_reason = lane_bounds.get("right_detect_reject_reason")
    m.left_memory_error_px = _f(lane_bounds.get("left_memory_error_px"))
    m.right_memory_error_px = _f(lane_bounds.get("right_memory_error_px"))
    m.purple_width_error_px = _f(lane_bounds.get("purple_width_error_px"))
    if "corridor_pinch_reject" in lane_bounds:
        m.corridor_pinch_reject = bool(lane_bounds.get("corridor_pinch_reject"))
    m.left_opposite_memory_error_px = _f(
        lane_bounds.get("left_opposite_memory_error_px"))
    m.right_opposite_memory_error_px = _f(
        lane_bounds.get("right_opposite_memory_error_px"))
    if "left_support_weak" in lane_bounds:
        m.left_support_weak = bool(lane_bounds.get("left_support_weak"))
    if "right_support_weak" in lane_bounds:
        m.right_support_weak = bool(lane_bounds.get("right_support_weak"))
    m.left_parallel_slope_error = _f(lane_bounds.get("left_parallel_slope_error"))
    m.right_parallel_slope_error = _f(lane_bounds.get("right_parallel_slope_error"))
    m.left_parallel_curvature_error = _f(
        lane_bounds.get("left_parallel_curvature_error"))
    m.right_parallel_curvature_error = _f(
        lane_bounds.get("right_parallel_curvature_error"))
    m.left_expected_from_opposite_error_px = _f(
        lane_bounds.get("left_expected_from_opposite_error_px"))
    m.right_expected_from_opposite_error_px = _f(
        lane_bounds.get("right_expected_from_opposite_error_px"))
    if "left_parallel_reject" in lane_bounds:
        m.left_parallel_reject = bool(lane_bounds.get("left_parallel_reject"))
    if "right_parallel_reject" in lane_bounds:
        m.right_parallel_reject = bool(lane_bounds.get("right_parallel_reject"))


def extract_frame_metrics(
    *,
    lane_bounds: dict | None = None,
    red_center: dict | None = None,
    render_corridor: dict | None = None,
    perception: Any = None,
    lane_state: Any = None,
    frame_id: int | None = None,
    timestamp: float | None = None,
) -> FrameMetrics:
    """Pure extraction. Explicit kwargs win over ``perception.*`` fallbacks."""
    m = FrameMetrics()
    p_fid = getattr(perception, "frame_id", None) if perception is not None else None
    p_ts = getattr(perception, "timestamp", None) if perception is not None else None
    m.frame_id = frame_id if frame_id is not None else p_fid
    m.timestamp = timestamp if timestamp is not None else p_ts

    if isinstance(lane_bounds, dict):
        _fill_lane_bounds(m, lane_bounds)

    if isinstance(red_center, dict):
        m.center_source = red_center.get("center_source")
        m.center_confidence = _f(red_center.get("center_confidence"))
        m.center_reason = red_center.get("reason")
        m.lane_area_confidence = _f(red_center.get("lane_area_confidence"))
        m.purple_confidence = _f(red_center.get("purple_confidence"))
        m.road_type = red_center.get("road_type")
        m.red_center_valid = bool(red_center.get("valid", False))
        if not m.red_center_valid:
            m.red_center_reason = red_center.get("reason")

    if isinstance(render_corridor, dict):
        m.render_corridor_valid = bool(render_corridor.get("valid", False))
        if not m.render_corridor_valid:
            m.render_corridor_reason = render_corridor.get("reason")

    m.component_counts = _component_counts(lane_bounds)

    ls = lane_state
    if ls is None and perception is not None:
        ls = getattr(perception, "lane_state", None)
    if ls is not None:
        err = getattr(ls, "error_px", None)
        m.lane_error_px = int(err) if err is not None else None
        m.steering = _f(getattr(ls, "steering", None))

    return m
