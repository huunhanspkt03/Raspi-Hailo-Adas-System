"""
Typed state structures for ADAS pipeline results.

Small dataclasses that replace loose dicts flowing between pipeline stages.
Keeps the contract explicit without adding weight.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BehaviorState:
    """Snapshot of the behavior-logic decision for one frame."""

    action: str = "go"  # "stop" | "follow" | "go"
    reason: str = "lane_follow"
    throttle: float = 0.35
    leading_vehicle: dict | None = None
    speed_sign: str | None = None

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "reason": self.reason,
            "throttle": self.throttle,
            "leading_vehicle": self.leading_vehicle,
            "speed_sign": self.speed_sign,
        }


@dataclass(slots=True)
class LaneState:
    """Bundled lane-perception result for one frame.

    Groups lane center path output with the derived steering values so
    downstream consumers (HUD, RC control) read a single object instead
    of juggling 4-5 loose variables.
    """

    detected: bool = False
    error_px: int | None = None
    steering: float = 0.0
    confidence: float = 0.0
    control_x: int | None = None
    control_y: int | None = None
    mean_width_bev: float = 0.0
    poly_coeffs: tuple[float, ...] = ()
    center_source: str | None = None
    road_type: str | None = None

    @staticmethod
    def from_path_result(
        path_result: dict | None,
        error_px: int | None,
        steering: float,
    ) -> LaneState:
        if path_result is None:
            return LaneState(detected=False, steering=steering)
        return LaneState(
            detected=True,
            error_px=error_px,
            steering=steering,
            confidence=float(path_result.get("confidence", 0.0)),
            control_x=path_result.get("control_x"),
            control_y=path_result.get("control_y"),
            mean_width_bev=float(path_result.get("mean_width_bev", 0.0)),
            poly_coeffs=tuple(path_result.get("poly_coeffs", ())),
            center_source=path_result.get("center_source"),
            road_type=path_result.get("road_type"),
        )


@dataclass(slots=True)
class PerceptionState:
    """Per-frame perception bundle produced by the perception pipeline.

    This is the contract returned by the pipeline per frame and is
    intentionally lightweight so downstream modules (behavior/control)
    can consume a single structured object.
    """

    frame_id: int | None = None
    timestamp: float | None = None
    lane_state: LaneState | None = None
    object_state: dict | None = None
    detections: list | None = None
    perception_confidence: float = 0.0
    raw_frame_shape: tuple[int, int] | None = None
    # Prototype ego-speed estimate from front-camera optical flow.
    # ``None`` when the estimator did not run; otherwise the dict returned
    # by ``EgoSpeedEstimator.update`` (includes ``valid`` + ``scale_source``).
    ego_speed: dict | None = None

    def as_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "lane_state": self.lane_state,
            "object_state": self.object_state,
            "detections": self.detections,
            "perception_confidence": self.perception_confidence,
            "raw_frame_shape": self.raw_frame_shape,
            "ego_speed": self.ego_speed,
        }
