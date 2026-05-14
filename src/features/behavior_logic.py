"""
Behavior Logic — translates YOLO26 detections into RC throttle/action state.

Priority (highest first):
  1. red_light detected       → stop
  3. vehicle ahead in path    → follow (reduced throttle)
  4. active speed-limit sign  → throttle cap or floor
  5. default                  → go (lane follow)

Temporal smoothing (BehaviorSmoother):
  - Majority-vote over a sliding window prevents single-frame jitter.
  - Hysteresis: releasing "stop" requires several consecutive non-stop
    frames, avoiding dangerous stop→go→stop oscillations.
  - Throttle uses the median of the window, rejecting outlier spikes.
"""
from __future__ import annotations

from collections import deque
from collections import Counter
from typing import Any

import numpy as np

from features.detection_filters import find_leading_vehicle, resolve_speed_sign
from features.state_types import BehaviorState

# Throttle caps for speed-limit signs (upper bound on throttle)
_SPEED_CAPS: dict[str, float] = {
    "speed_limit_40": 0.20,
    "speed_limit_50": 0.25,
    "speed_limit_60": 0.30,
    "speed_limit_70": 0.35,
}


def _has_active_red_light(detections: list[dict[str, Any]]) -> bool:
    red_lights = [d for d in detections if d.get("label") == "red_light"]
    if not red_lights:
        return False
    return any(d.get("pose_valid", True) for d in red_lights)


def compute_behavior_state(
    detections: list[dict[str, Any]],
    control_cfg: dict[str, Any],
    frame_w: int,
) -> BehaviorState:
    """
    Evaluate detection results and return a BehaviorState (single-frame, no smoothing).

    This is the raw per-frame decision.  Wrap with ``BehaviorSmoother`` for
    temporally stable output.
    """
    default_throttle = float(control_cfg.get("default_throttle", 0.35))
    safe_throttle    = float(control_cfg.get("safe_throttle", 0.20))
    stop_throttle    = float(control_cfg.get("stop_throttle", 0.0))

    leading_vehicle = find_leading_vehicle(detections, img_w=frame_w)
    speed_sign      = resolve_speed_sign(detections)

    if _has_active_red_light(detections):
        return BehaviorState(
            action="stop", reason="red_light", throttle=stop_throttle,
            leading_vehicle=leading_vehicle, speed_sign=speed_sign,
        )

    throttle = safe_throttle if leading_vehicle else default_throttle
    action   = "follow" if leading_vehicle else "go"
    reason   = "vehicle_ahead" if leading_vehicle else "lane_follow"

    if speed_sign in _SPEED_CAPS:
        throttle = min(throttle, _SPEED_CAPS[speed_sign])
    elif speed_sign == "speed_limit_80":
        throttle = max(min(throttle, 0.45), 0.40)

    return BehaviorState(
        action=action, reason=reason, throttle=throttle,
        leading_vehicle=leading_vehicle, speed_sign=speed_sign,
    )


class BehaviorSmoother:
    """Temporal smoothing over a sliding window of BehaviorState snapshots.

    Design choices (inspired by TaskConditions-style patterns):
      - **Action**: majority vote over the window.
      - **Stop hysteresis**: once ``stop`` is the smoothed action, it holds
        until ``release_frames`` consecutive non-stop raw frames appear.
        This prevents the car from lurching forward on a single missed
        red-light detection.
      - **Throttle**: median of the window — robust to one-off spikes.
      - **Reason / metadata**: taken from the latest raw frame (most current).

    Parameters
    ----------
    window : int
        Number of recent frames to consider (default 5).
    release_frames : int
        Consecutive non-stop frames required to exit ``stop`` (default 3).
    """

    def __init__(self, window: int = 5, release_frames: int = 3) -> None:
        self._window = max(1, window)
        self._release_frames = max(1, release_frames)
        self._history: deque[BehaviorState] = deque(maxlen=self._window)
        self._non_stop_streak: int = 0
        self._locked_stop: bool = False

    def _majority_action(self) -> str:
        counts = Counter(state.action for state in self._history)
        if not counts:
            return "go"

        top_count = max(counts.values())
        for state in reversed(self._history):
            if counts[state.action] == top_count:
                return state.action
        return self._history[-1].action

    def update(self, raw: BehaviorState) -> BehaviorState:
        """Push a raw per-frame state and return the smoothed result."""
        self._history.append(raw)

        # --- action: majority vote ---
        voted_action = self._majority_action()

        # --- stop hysteresis ---
        if raw.action != "stop":
            self._non_stop_streak += 1
        else:
            self._non_stop_streak = 0

        if voted_action == "stop" or raw.action == "stop":
            self._locked_stop = True
        if self._locked_stop and self._non_stop_streak >= self._release_frames:
            self._locked_stop = False

        smoothed_action = "stop" if self._locked_stop else voted_action

        # --- throttle: median ---
        throttles = [s.throttle for s in self._history]
        smoothed_throttle = float(np.median(throttles))
        if smoothed_action == "stop":
            for state in reversed(self._history):
                if state.action == "stop":
                    smoothed_throttle = state.throttle
                    break

        # --- reason: from latest frame matching the smoothed action,
        #     falling back to the raw frame's reason ---
        smoothed_reason = raw.reason
        for s in reversed(self._history):
            if s.action == smoothed_action:
                smoothed_reason = s.reason
                break

        return BehaviorState(
            action=smoothed_action,
            reason=smoothed_reason,
            throttle=smoothed_throttle,
            leading_vehicle=raw.leading_vehicle,
            speed_sign=raw.speed_sign,
        )

    def reset(self) -> None:
        """Clear history (e.g. on mode switch)."""
        self._history.clear()
        self._non_stop_streak = 0
        self._locked_stop = False
