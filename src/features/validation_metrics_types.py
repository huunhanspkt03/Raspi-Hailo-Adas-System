"""Per-frame validation metric dataclasses."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ComponentCounts:
    """Per-side accepted + rejected-by-reason breakdown."""
    accepted_left: int = 0
    accepted_right: int = 0
    accepted_unknown: int = 0
    rejected_total: int = 0
    rejected_by_reason: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class FrameMetrics:
    """Lightweight per-frame metrics. All fields nullable so a record is
    still emitted on degenerate frames (lane LOST, etc.)."""
    frame_id: int | None = None
    timestamp: float | None = None
    lane_state: str | None = None
    boundary_confidence: float | None = None
    left_source: str | None = None
    right_source: str | None = None
    left_inference_kind: str | None = None
    right_inference_kind: str | None = None
    left_confidence_final: float | None = None
    right_confidence_final: float | None = None
    purple_lane_width_px: float | None = None
    purple_lane_width_valid: bool = False
    purple_lane_width_source: str | None = None
    purple_memory_update_allowed: bool | None = None
    purple_memory_update_reject_reason: str | None = None
    lane_width_px: float | None = None
    lane_width_valid: bool = False
    lane_width_source: str | None = None
    lane_area_width_px: float | None = None
    lane_area_width_valid: bool = False
    center_source: str | None = None
    center_confidence: float | None = None
    center_reason: str | None = None
    lane_area_confidence: float | None = None
    purple_confidence: float | None = None
    road_type: str | None = None
    red_center_valid: bool | None = None
    red_center_reason: str | None = None
    render_corridor_valid: bool | None = None
    render_corridor_reason: str | None = None
    component_counts: ComponentCounts = field(default_factory=ComponentCounts)
    left_detect_valid: bool | None = None
    right_detect_valid: bool | None = None
    left_detect_reject_reason: str | None = None
    right_detect_reject_reason: str | None = None
    left_memory_error_px: float | None = None
    right_memory_error_px: float | None = None
    purple_width_error_px: float | None = None
    corridor_pinch_reject: bool | None = None
    left_opposite_memory_error_px: float | None = None
    right_opposite_memory_error_px: float | None = None
    left_support_weak: bool | None = None
    right_support_weak: bool | None = None
    left_parallel_slope_error: float | None = None
    right_parallel_slope_error: float | None = None
    left_parallel_curvature_error: float | None = None
    right_parallel_curvature_error: float | None = None
    left_expected_from_opposite_error_px: float | None = None
    right_expected_from_opposite_error_px: float | None = None
    left_parallel_reject: bool | None = None
    right_parallel_reject: bool | None = None
    lane_error_px: int | None = None
    steering: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable plain dict."""
        return asdict(self)
