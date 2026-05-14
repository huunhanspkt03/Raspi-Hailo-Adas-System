"""Rendering + save + display helpers for the threaded runtime output thread.

Isolated so ``threaded_runtime.py`` stays orchestration-only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
from loguru import logger

from pipeline.packets import BehaviorPacket
from pipeline.postprocessor import (
    draw_detections, draw_fixed_guidance, draw_hud, draw_lane_debug,
)
from features.state_types import LaneState


def render_behavior_packet(
    pkt: BehaviorPacket,
    *,
    fast_overlay: bool,
    hailo_active: bool,
    display_fps: float,
) -> Any:
    out = pkt.frame.copy()
    corridor = pkt.lane.corridor if pkt.lane is not None else None
    if pkt.lane is not None:
        lane = pkt.lane
        out = draw_lane_debug(out, lane.lane_area, lane.semantics, lane.class_probs,
                              lane.lane_path, lane.lane_bounds, fast=fast_overlay,
                              guidance=corridor)
    else:
        draw_fixed_guidance(out)
    if pkt.objects is not None and pkt.objects.detections:
        out = draw_detections(out, pkt.objects.detections, copy_frame=False)
    lane_state = pkt.lane.lane_state if pkt.lane is not None \
        else LaneState(detected=False, steering=pkt.control.steering)
    front = pkt.objects.front_object if pkt.objects is not None else None
    out = draw_hud(out, display_fps, lane_state,
                   pkt.control.throttle, pkt.behavior_action, pkt.behavior_reason,
                   hailo_active, front, copy_frame=False, corridor=corridor)
    return out


def open_video_writer(
    save_path: str,
    size_wh: tuple[int, int],
    output_fps: float,
) -> cv2.VideoWriter:
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             output_fps, size_wh)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {out_path}")
    logger.info("Saving output video to {} @ {:.3f} fps", out_path, output_fps)
    return writer


def resize_for_display(out: Any, proc_w: int, orig_w: int, orig_h: int) -> Any:
    proc_h = proc_w * orig_h // max(orig_w, 1)
    if out.shape[1] != proc_w or out.shape[0] != proc_h:
        out = cv2.resize(out, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
    return out
