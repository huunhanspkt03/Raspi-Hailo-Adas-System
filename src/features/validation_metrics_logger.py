"""JSONL / CSV writer for ``FrameMetrics``. No-op when ``path`` is ``None``.

JSONL is the default — nested ``component_counts`` map serializes
cleanly. CSV flattens the counts into top-level columns and stores
``rejected_by_reason`` as a JSON string in a single cell.
"""
from __future__ import annotations

import json
from typing import Any

from features.validation_metrics import FrameMetrics

__all__ = ["MetricsLogger"]

_CSV_COLUMNS = (
    "frame_id", "timestamp", "lane_state", "boundary_confidence",
    "left_source", "right_source",
    "left_inference_kind", "right_inference_kind",
    "left_confidence_final", "right_confidence_final",
    "purple_lane_width_px", "purple_lane_width_valid", "purple_lane_width_source",
    "purple_memory_update_allowed", "purple_memory_update_reject_reason",
    "lane_width_px", "lane_width_valid", "lane_width_source",
    "lane_area_width_px", "lane_area_width_valid",
    "center_source", "center_confidence", "center_reason",
    "lane_area_confidence", "purple_confidence",
    "road_type", "red_center_valid", "red_center_reason",
    "render_corridor_valid", "render_corridor_reason",
    "accepted_left", "accepted_right", "accepted_unknown",
    "rejected_total", "rejected_by_reason",
    "left_detect_valid", "right_detect_valid",
    "left_detect_reject_reason", "right_detect_reject_reason",
    "left_memory_error_px", "right_memory_error_px",
    "purple_width_error_px", "corridor_pinch_reject",
    "left_opposite_memory_error_px", "right_opposite_memory_error_px",
    "left_support_weak", "right_support_weak",
    "left_parallel_slope_error", "right_parallel_slope_error",
    "left_parallel_curvature_error", "right_parallel_curvature_error",
    "left_expected_from_opposite_error_px",
    "right_expected_from_opposite_error_px",
    "left_parallel_reject", "right_parallel_reject",
    "lane_error_px", "steering",
)


def _json_default(o: Any) -> Any:
    """Coerce numpy scalars / arrays so ``json.dumps`` does not choke on
    ``np.float32`` confidence values that may slip through."""
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    raise TypeError(
        f"Object of type {type(o).__name__} is not JSON serializable")


def _csv_row(m: FrameMetrics) -> list[str]:
    d = m.to_dict()
    cc = d.pop("component_counts", {}) or {}
    flat = {**d, **cc}
    flat["rejected_by_reason"] = json.dumps(
        flat.get("rejected_by_reason") or {}, sort_keys=True)
    out: list[str] = []
    for col in _CSV_COLUMNS:
        v = flat.get(col)
        out.append("" if v is None else str(v))
    return out


class MetricsLogger:
    """Optional per-frame metrics writer."""

    SUPPORTED_FORMATS = ("jsonl", "csv")

    def __init__(self, path: str | None = None, *, fmt: str = "jsonl") -> None:
        self.format = str(fmt).lower()
        if self.format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported metrics format: {fmt!r} "
                f"(supported: {self.SUPPORTED_FORMATS})")
        self.path = path
        self.enabled = bool(path)
        self._fp = None
        if self.enabled:
            self._fp = open(path, "w", encoding="utf-8", newline="")
            if self.format == "csv":
                self._fp.write(",".join(_CSV_COLUMNS) + "\n")

    @classmethod
    def disabled(cls) -> "MetricsLogger":
        return cls(path=None)

    def log(self, metrics: FrameMetrics) -> None:
        if not self.enabled or self._fp is None:
            return
        if self.format == "jsonl":
            self._fp.write(json.dumps(metrics.to_dict(), default=_json_default))
            self._fp.write("\n")
        else:
            self._fp.write(",".join(_csv_row(metrics)) + "\n")
        self._fp.flush()

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None
        self.enabled = False

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
