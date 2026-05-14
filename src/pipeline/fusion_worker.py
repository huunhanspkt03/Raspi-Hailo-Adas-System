"""Fusion / behavior + safety supervisor workers for the threaded runtime.

Synchronizes LanePacket + ObjectPacket by frame_id (with a stale
tolerance) and produces a BehaviorPacket carrying the final
ControlCommand. A separate ``safety_supervisor`` clamps actuator output
and forwards the command to the RC send function.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from pipeline.latest_queue import LatestQueue
from pipeline.packets import (
    BehaviorPacket, ControlCommand, FramePacket, LanePacket, ObjectPacket,
)
from features.behavior_logic import compute_behavior_state
from features.controller import (
    compute_steering, gate_by_confidence, rate_limit, smooth_value,
)
from features.state_types import LaneState


class FusionWorker:
    """Builds WorldModel + BehaviorState + ControlCommand from matched packets."""

    def __init__(
        self,
        *,
        ctrl_cfg: dict[str, Any],
        rc_cfg: dict[str, Any],
        deadband_px: int,
        max_steer: float,
        smooth_alpha: float,
        smoothing: bool,
        default_throttle: float,
        behavior_enabled: bool,
        rc_enabled: bool,
        behavior_smoother: Any | None = None,
    ) -> None:
        self._ctrl_cfg = ctrl_cfg
        self._rc_cfg = rc_cfg
        self._deadband_px = deadband_px
        self._max_steer = max_steer
        self._smooth_alpha = smooth_alpha
        self._smoothing = smoothing
        self._default_throttle = default_throttle
        self._behavior_enabled = behavior_enabled
        self._rc_enabled = rc_enabled
        self._behavior_smoother = behavior_smoother
        self._steering_smooth: float = 0.0

    def fuse(
        self,
        frame_pkt: FramePacket,
        lane_pkt: LanePacket | None,
        object_pkt: ObjectPacket | None,
    ) -> BehaviorPacket:
        steering = 0.0
        throttle = self._default_throttle
        action, reason = _default_action(lane_pkt is not None, object_pkt is not None)
        lane_state = self._fuse_lane(lane_pkt, frame_pkt)
        if lane_pkt is not None:
            lane_pkt.lane_state = lane_state
            steering = lane_state.steering

        if self._behavior_enabled and object_pkt is not None:
            raw_state = compute_behavior_state(
                object_pkt.detections, self._ctrl_cfg, frame_pkt.frame_w,
            )
            state = self._behavior_smoother.update(raw_state) \
                if self._behavior_smoother else raw_state
            action, reason = state.action, state.reason
            throttle = float(state.throttle)

        control = ControlCommand(
            steering=float(steering), throttle=float(throttle),
            rc_enabled=bool(self._rc_enabled and self._rc_cfg.get("enabled", False)),
            safe=True,
        )
        return BehaviorPacket(
            frame_id=frame_pkt.frame_id, timestamp=frame_pkt.timestamp,
            frame=frame_pkt.frame, lane=lane_pkt, objects=object_pkt,
            behavior_action=action, behavior_reason=reason, control=control,
        )

    def _fuse_lane(
        self,
        lane_pkt: LanePacket | None,
        frame_pkt: FramePacket,
    ) -> LaneState:
        if lane_pkt is None:
            return LaneState(detected=False, steering=0.0)
        path = lane_pkt.lane_path
        if path is None or path.get("control_x") is None:
            return LaneState.from_path_result(path, None, self._steering_smooth)
        ctrl_cfg = self._ctrl_cfg
        err_px, raw_steer = compute_steering(
            path["control_x"], frame_pkt.frame_w,
            self._deadband_px, self._max_steer,
        )
        conf = float(path.get("confidence", 0.0))
        conf_hold = float(ctrl_cfg.get("confidence_hold_thresh", 0.25))
        conf_soft = float(ctrl_cfg.get("confidence_soft_thresh", 0.45))
        max_rate = float(ctrl_cfg.get("max_steering_rate_per_frame", 0.08))
        gated = gate_by_confidence(raw_steer, self._steering_smooth,
                                   conf, conf_hold, conf_soft)
        limited = rate_limit(gated, self._steering_smooth,
                             max_rate * max(conf, conf_hold))
        self._steering_smooth = smooth_value(
            self._steering_smooth, limited, self._smooth_alpha,
        )
        steering = float(self._steering_smooth) if self._smoothing else limited
        return LaneState.from_path_result(path, err_px, steering)


def _default_action(lane_on: bool, obj_on: bool) -> tuple[str, str]:
    if lane_on and not obj_on:
        return "go", "lane_only"
    if obj_on and not lane_on:
        return "observe", "detection_only"
    return "go", "full_stack"


def safety_supervisor(
    pkt: BehaviorPacket,
    *,
    max_steer: float,
    rc_send: Callable[[float, float, bool], None],
) -> BehaviorPacket:
    """Clamp actuator outputs and forward to RC."""
    cmd = pkt.control
    steering = float(np.clip(cmd.steering, -max_steer, max_steer))
    throttle = float(np.clip(cmd.throttle, 0.0, 1.0))
    pkt.control = ControlCommand(
        steering=steering, throttle=throttle,
        rc_enabled=cmd.rc_enabled, safe=cmd.safe,
    )
    if cmd.rc_enabled:
        rc_send(steering, throttle, True)
    return pkt


def drain_state_queues(
    lane_state_q: LatestQueue[LanePacket],
    object_state_q: LatestQueue[ObjectPacket],
    pending_lane: dict[int, LanePacket],
    pending_obj: dict[int, ObjectPacket],
    want_lane: bool, want_obj: bool,
) -> bool:
    """Pull available state packets into pending buffers. Returns True if any."""
    got_any = False
    if want_lane:
        lp = lane_state_q.get(timeout=0.05)
        if lp is not None:
            pending_lane[lp.frame_id] = lp
            got_any = True
    if want_obj:
        op = object_state_q.try_get()
        while op is not None:
            pending_obj[op.frame_id] = op
            got_any = True
            op = object_state_q.try_get()
        if not got_any and not want_lane:
            op = object_state_q.get(timeout=0.05)
            if op is not None:
                pending_obj[op.frame_id] = op
                got_any = True
    return got_any


def emit_fused(
    fusion: "FusionWorker",
    behavior_q: LatestQueue[BehaviorPacket],
    pending_lane: dict[int, LanePacket],
    pending_obj: dict[int, ObjectPacket],
    want_lane: bool, want_obj: bool,
    stale_tolerance: int = 8,
) -> None:
    """Match lane/object by frame_id and push fused BehaviorPackets."""
    if want_lane and want_obj:
        for fid in sorted(set(pending_lane) & set(pending_obj)):
            lane_pkt = pending_lane.pop(fid)
            obj_pkt = pending_obj.pop(fid)
            behavior_q.put(fusion.fuse(_frame_pkt_from_lane(lane_pkt),
                                       lane_pkt, obj_pkt))
        newest = max(pending_lane.keys() | pending_obj.keys(), default=0)
        _evict_stale(pending_lane, newest, stale_tolerance)
        _evict_stale(pending_obj, newest, stale_tolerance)
    elif want_lane:
        for fid in sorted(pending_lane):
            lp = pending_lane.pop(fid)
            behavior_q.put(fusion.fuse(_frame_pkt_from_lane(lp), lp, None))
    elif want_obj:
        for fid in sorted(pending_obj):
            op = pending_obj.pop(fid)
            behavior_q.put(fusion.fuse(_frame_pkt_from_object(op), None, op))


def _frame_pkt_from_lane(lp: LanePacket) -> FramePacket:
    h, w = lp.source_frame.shape[:2]
    return FramePacket(frame_id=lp.frame_id, timestamp=lp.timestamp,
                       frame=lp.source_frame, frame_h=h, frame_w=w)


def _frame_pkt_from_object(op: ObjectPacket) -> FramePacket:
    h, w = op.source_frame.shape[:2]
    return FramePacket(frame_id=op.frame_id, timestamp=op.timestamp,
                       frame=op.source_frame, frame_h=h, frame_w=w)


def _evict_stale(d: dict, newest: int, tolerance: int) -> None:
    stale = [fid for fid in d if newest - fid > tolerance]
    for fid in stale:
        d.pop(fid, None)
