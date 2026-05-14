"""Core pipeline loop — runs lane segmentation and/or object detection each frame."""
from __future__ import annotations

import time

import cv2
import numpy as np
from loguru import logger

from pipeline.preprocessor import preprocess_segmentation, preprocess_yolo26
from pipeline.postprocessor import (
    draw_detections, draw_fixed_guidance, draw_lane_debug,
)
from pipeline.raw_overlay import draw_raw_overlay
from pipeline.lane_debug_panel import build_lane_debug_panel
from features.lane_corridor import fixed_guidance_template
from features.lane_corridor_render import (
    build_render_corridor, reset_render_corridor_state,
)
from features.dynamic_corridor_tracker import DynamicCorridorTracker
from features.lane_postprocess import (
    decode_lane_components,
    decode_segmentation_probabilities,
)
from features.lane_boundary_tracker import LaneBoundaryTracker
from features.lane_path import mask_centerline
from features.object_localizer import ObjectLocalizer
from features.object_detection import decode_yolo26
from features.object_tracker import ByteTracker
from features.controller import compute_steering, gate_by_confidence, rate_limit, smooth_value
from features.ego_speed_estimator import EgoSpeedEstimator
from features.red_center_from_boundaries import compute_red_centerline
from features.ego_anchored_path import build_ego_anchored_path
from features.state_types import LaneState, PerceptionState

_OBJECT_LOCALIZER = ObjectLocalizer()
_OBJECT_TRACKER = ByteTracker()
_DYNAMIC_CORRIDOR = DynamicCorridorTracker()
_LANE_BOUNDARY_TRACKER = LaneBoundaryTracker()
_EGO_SPEED_ESTIMATOR = EgoSpeedEstimator()


def _fmt_coeffs(coeffs) -> str:
    if coeffs is None:
        return "None"
    a, b, c = coeffs
    return f"({float(a):+.3e},{float(b):+.3f},{float(c):+.1f})"


def _fmt_pts(pts) -> str:
    if pts is None or len(pts) == 0:
        return "empty"
    arr = np.asarray(pts, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return "bad"
    arr = arr[np.argsort(arr[:, 1])]
    top = arr[0]
    bot = arr[-1]
    return (f"top=({top[0]:.0f},{top[1]:.0f}) "
            f"bot=({bot[0]:.0f},{bot[1]:.0f}) n={arr.shape[0]}")


def _log_lane_geometry_debug(frame_id: int, lane_bounds: dict | None,
                             render_corr: dict | None) -> None:
    """Terminal geometry debug for lane_area / purple render state."""
    if not isinstance(lane_bounds, dict):
        logger.info("LANEDBG f={} lane_bounds=None", frame_id)
        return
    rc = render_corr if isinstance(render_corr, dict) else {}
    red = lane_bounds.get("red_center") or {}
    logger.info(
        "LANEDBG f={} state={} center={} conf={} "
        "lane_w={}/{} area_w={} area_valid={} purple_w={}/{} "
        "render_valid={} reason={}",
        frame_id,
        lane_bounds.get("lane_state"),
        red.get("center_source"),
        f"{float(red.get('center_confidence', 0.0)):.2f}",
        lane_bounds.get("lane_width_px"),
        lane_bounds.get("lane_width_source"),
        lane_bounds.get("lane_area_width_px"),
        lane_bounds.get("lane_area_width_valid"),
        lane_bounds.get("purple_lane_width_px"),
        lane_bounds.get("purple_lane_width_valid"),
        rc.get("valid"),
        rc.get("reason"),
    )
    for side in ("left", "right"):
        logger.info(
            "LANEDBG f={} {} src={} kind={} coeffs={} det_pts={} "
            "render_src={} render_pts={}",
            frame_id,
            side,
            lane_bounds.get(f"{side}_source"),
            lane_bounds.get(f"{side}_inference_kind"),
            _fmt_coeffs(lane_bounds.get(f"{side}_coeffs")),
            _fmt_pts(lane_bounds.get(f"{side}_pts_img")),
            lane_bounds.get(f"{side}_render_src"),
            _fmt_pts(rc.get(f"purple_{side}_pts_img")),
        )
    # Width-x learner diagnostics are surfaced for metrics, but missing-side
    # inference is disabled so the profile no longer draws or controls a side.
    logger.info(
        "LANEDBG f={} width_profile ready={} age={}/{} median={} "
        "min={} max={} skip={}",
        frame_id,
        lane_bounds.get("width_profile_ready"),
        lane_bounds.get("width_profile_age"),
        lane_bounds.get("width_profile_learning_frames"),
        lane_bounds.get("width_profile_median_px"),
        lane_bounds.get("width_profile_min_px"),
        lane_bounds.get("width_profile_max_px"),
        lane_bounds.get("width_profile_skip_reason"),
    )
    logger.info(
        "LANEDBG f={} lane_area_edges left={} right={}",
        frame_id,
        _fmt_pts(lane_bounds.get("lane_area_left_pts")),
        _fmt_pts(lane_bounds.get("lane_area_right_pts")),
    )


def reset_lane_tracker() -> None:
    """Reset stateful lane geometry trackers."""
    _LANE_BOUNDARY_TRACKER.reset()
    reset_render_corridor_state()


def reset_object_tracker() -> None:
    _OBJECT_TRACKER.reset()


def reset_dynamic_corridor() -> None:
    _DYNAMIC_CORRIDOR.reset()


def reset_ego_speed_estimator() -> None:
    _EGO_SPEED_ESTIMATOR.reset()


def _run_lane_branch(
    raw_output: dict,
    frame_h: int,
    frame_w: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray, dict | None, dict | None]:
    """Decode YOLO26s_seg → masks → tracked boundaries → red-center lane_path.

    Returns the legacy 6-tuple ``(mask, class_probs, semantics, lane_area,
    lane_path, lane_bounds)`` that downstream renderers still consume.
    """
    mask, class_probs, semantics, lane_area = decode_lane_components(raw_output, frame_h, frame_w)
    area_path = mask_centerline(lane_area)
    boundary_track = _LANE_BOUNDARY_TRACKER.update(semantics, lane_area, frame_h, frame_w)
    red_center = compute_red_centerline(lane_area, boundary_track, frame_h, frame_w)
    lane_bounds = dict(boundary_track)
    lane_bounds["red_center"] = red_center
    lane_bounds["lane_area_path"] = area_path
    lane_path = _lane_path_from_red_center(red_center, frame_h, frame_w)
    if lane_path is None:
        lane_path = area_path
    if lane_path is not None:
        lane_path["display_lane_area_img"] = lane_area
        if red_center.get("valid"):
            lane_path["red_center"] = red_center
            lane_path["center_source"] = red_center.get("center_source")
            lane_path["road_type"] = red_center.get("road_type")
    return mask, class_probs, semantics, lane_area, lane_path, lane_bounds


def _lane_path_from_red_center(
    red_center: dict, frame_h: int, frame_w: int,
) -> dict | None:
    """Build the ego-anchored red path + lookahead control point.

    The path's tail sits at ``ego_x = frame_w / 2`` (so the visualization
    starts from the camera column) and its head bends toward the per-row
    lane-center targets from ``red_center``. ``control_x`` is sampled at
    the lookahead row, not the tail — using the tail would always yield
    zero lateral error and freeze the steering at neutral.
    """
    if not red_center.get("valid"):
        return None
    pts = red_center.get("center_pts_img")
    if pts is None or len(pts) == 0:
        return None
    ys = np.asarray(pts[:, 1], dtype=np.float32)
    xs_ego, diag = build_ego_anchored_path(pts, ys, frame_h, frame_w)
    ego_pts = np.stack([xs_ego, ys], axis=1).astype(np.float32)
    control_x = int(round(float(diag["ego_path_control_x"])))
    control_y = int(round(float(diag["ego_path_control_y"])))
    return {
        "is_red_center": True,
        "control_x": control_x,
        "control_y": control_y,
        "fitted_pts_img": ego_pts,
        "lane_target_pts_img": pts,
        "ego_path_a": float(diag["ego_path_a"]),
        "ego_path_anchor_y": float(diag["ego_path_anchor_y"]),
        "ego_path_lookahead_y": float(diag["ego_path_lookahead_y"]),
        "ego_path_top_disp_clamped": bool(diag.get("ego_path_top_disp_clamped", False)),
        "confidence": float(red_center.get("center_confidence", 0.0)),
        "poly_coeffs": tuple(red_center.get("center_coeffs") or ()),
        "row_count": int(len(ego_pts)),
    }


def _run_detection_branch(
    raw_output: dict,
    conf_thresh: float,
    frame_w: int,
    frame_h: int,
    yolo_size: int,
) -> list[dict]:
    return decode_yolo26(raw_output, conf_thresh, frame_w, frame_h, yolo_size, yolo_size)


def run_raw_overlay_frame(
    frame, out, cam_t1, loop_t0, infer, fps,
    segmentation_size, fast_overlay, render_output,
):
    """Process one frame in raw-overlay mode (segmentation debug only)."""
    t0 = cam_t1
    prep = preprocess_segmentation(frame, segmentation_size)
    t1 = time.perf_counter()
    raw = infer.infer("lane_segmentation", prep)
    t2 = time.perf_counter()
    frame_h, frame_w = frame.shape[:2]
    class_probs = decode_segmentation_probabilities(raw, frame_h, frame_w)
    t3 = time.perf_counter()
    if render_output and out is not None:
        out = draw_raw_overlay(out, None, class_probs, fast=fast_overlay)
    t4 = time.perf_counter()
    fps.update()
    ms_cam = (cam_t1 - loop_t0) * 1e3
    ms_pre = (t1 - t0) * 1e3
    ms_inf = (t2 - t1) * 1e3
    ms_dec = (t3 - t2) * 1e3
    ms_ov = (t4 - t3) * 1e3
    if render_output and out is not None:
        cv2.putText(out, f"FPS {fps.get():.1f}  [raw overlay]", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, f"cam={ms_cam:.0f}ms  pre={ms_pre:.0f}ms  inf={ms_inf:.0f}ms  dec={ms_dec:.0f}ms  ov={ms_ov:.0f}ms",
                    (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 180), 2, cv2.LINE_AA)
    return out, ms_pre, ms_inf, ms_dec, ms_ov


def run_pipeline_frame(
    frame, out, infer, fps,
    *,
    lane_enabled, object_enabled, behavior_enabled, rc_enabled,
    segmentation_size, yolo_size, conf_thresh,
    frame_w, frame_h,
    min_lane_w, deadband_px, max_steer, smooth_alpha, smoothing,
    default_throttle, fast_overlay, render_output,
    branch_executor, behavior_smoother, ctrl_cfg,
    hailo_active, steering_smooth, send_rc_fn, rc_cfg,
    lane_debug=False,
    metrics_logger=None,
):
    """Process one frame in full pipeline mode.

    Returns (out, steering_smooth, perception_state). Behavior and RC
    are intentionally performed downstream so the perception pipeline
    stays a pure producer of PerceptionState.
    """
    detections: list[dict] = []
    front_object: dict | None = None
    steering = 0.0
    lane_state = LaneState(detected=False, steering=steering)
    # Always-defined locals so the optional metrics_logger call below
    # works whether or not the lane branch ran.
    lane_bounds: dict | None = None
    red_center: dict | None = None
    render_corr: dict | None = None
    ego_speed: dict | None = None
    # Camera-anchored fixed guidance is the default and the only fallback.
    # If the lane branch produces a valid render-corridor (purple-anchored
    # but ego-fixed in red/yellow geometry), it replaces this in place;
    # otherwise the displayed red/yellow corridor stays exactly here, with
    # no dependency on lane_area, dynamic corridor, or red_center.
    guidance: dict = fixed_guidance_template(frame_h, frame_w)
    if lane_enabled and not object_enabled:
        behavior_action, behavior_reason = "go", "lane_only"
    elif object_enabled and not lane_enabled:
        behavior_action, behavior_reason = "observe", "detection_only"
    else:
        behavior_action, behavior_reason = "go", "full_stack"
    _ = (behavior_action, behavior_reason, behavior_enabled, rc_enabled,
         min_lane_w, behavior_smoother, send_rc_fn, rc_cfg, hailo_active)

    if branch_executor is not None:
        lane_prep_future = branch_executor.submit(preprocess_segmentation, frame, segmentation_size) if lane_enabled else None
        yolo_prep_future = branch_executor.submit(preprocess_yolo26, frame, yolo_size) if object_enabled else None
        lane_prep = lane_prep_future.result() if lane_prep_future is not None else None
        yolo_prep = yolo_prep_future.result() if yolo_prep_future is not None else None
    else:
        lane_prep = preprocess_segmentation(frame, segmentation_size) if lane_enabled else None
        yolo_prep = preprocess_yolo26(frame, yolo_size) if object_enabled else None

    raw_lane = infer.infer("lane_segmentation", lane_prep) if lane_enabled else None
    raw_yolo = infer.infer("yolo26", yolo_prep) if object_enabled else None

    if branch_executor is not None:
        lane_future = branch_executor.submit(_run_lane_branch, raw_lane, frame_h, frame_w) if lane_enabled else None
        det_future = branch_executor.submit(
            _run_detection_branch, raw_yolo, conf_thresh, frame_w, frame_h, yolo_size,
        ) if object_enabled else None
    else:
        lane_future = det_future = None

    if lane_enabled:
        if lane_future is not None:
            mask, class_probs, semantics, lane_area, lane_path, lane_bounds = lane_future.result()
        else:
            mask, class_probs, semantics, lane_area, lane_path, lane_bounds = _run_lane_branch(
                raw_lane, frame_h, frame_w,
            )
        # Prototype ego-speed (image-space scale, NOT calibrated). Side-effect
        # free w.r.t. steering/corridor: result is attached to PerceptionState.
        ego_speed = _EGO_SPEED_ESTIMATOR.update(
            frame, lane_area, fps=fps.get_instant() if fps is not None else None,
        )
        red_center = (lane_bounds or {}).get("red_center")
        # ``build_render_corridor`` produces the display corridor and the
        # render-resolved purple polylines. It may infer one missing purple
        # side for visualization without changing the tracker's semantic
        # source labels used by control/metrics.
        render_corr = build_render_corridor(
            lane_bounds, frame_h, frame_w, red_center=red_center,
        ) if lane_bounds is not None else {"valid": False}
        frame_id = int(getattr(fps, "frame_count", 0)) + 1
        if lane_debug and frame_id % 10 == 0:
            _log_lane_geometry_debug(frame_id, lane_bounds, render_corr)
        if render_corr.get("valid"):
            guidance = render_corr
            # Override the tracker's per-side polylines for rendering so
            # render_boundary_polylines draws BOTH purple sides (including
            # the inferred-from-other-side case).
            if lane_bounds is not None:
                lane_bounds["left_pts_img"] = render_corr["purple_left_pts_img"]
                lane_bounds["right_pts_img"] = render_corr["purple_right_pts_img"]
                lane_bounds["purple_render_valid"] = True
                lane_bounds["purple_render_left_source"] = (
                    render_corr.get("purple_left_source"))
                lane_bounds["purple_render_right_source"] = (
                    render_corr.get("purple_right_source"))
        if render_output and out is not None:
            display_lane_area = lane_area
            if lane_path is not None:
                display_lane_area = lane_path.get("display_lane_area_img", display_lane_area)
            if lane_debug:
                out = build_lane_debug_panel(frame, semantics, lane_path)
            else:
                out = draw_lane_debug(
                    out, display_lane_area, semantics, class_probs, lane_path, lane_bounds,
                    fast=fast_overlay, guidance=guidance,
                )
        lane_error_px = None
        if lane_path is not None and lane_path.get("control_x") is not None:
            lane_error_px, raw_steer = compute_steering(
                lane_path["control_x"], frame_w, deadband_px, max_steer,
            )
            conf = float(lane_path.get("confidence", 0.0))
            conf_hold = float(ctrl_cfg.get("confidence_hold_thresh", 0.25))
            conf_soft = float(ctrl_cfg.get("confidence_soft_thresh", 0.45))
            max_rate = float(ctrl_cfg.get("max_steering_rate_per_frame", 0.08))
            gated = gate_by_confidence(raw_steer, steering_smooth or 0.0, conf, conf_hold, conf_soft)
            limited = rate_limit(gated, steering_smooth or 0.0, max_rate * max(conf, conf_hold))
            steering_smooth = smooth_value(steering_smooth or 0.0, limited, smooth_alpha)
            steering = float(steering_smooth) if smoothing else limited
        lane_state = LaneState.from_path_result(lane_path, lane_error_px, steering)

    if object_enabled:
        if det_future is not None:
            detections = det_future.result()
        else:
            detections = _run_detection_branch(raw_yolo, conf_thresh, frame_w, frame_h, yolo_size)
        detections = _OBJECT_TRACKER.update(detections)
        front = _OBJECT_LOCALIZER.localize(
            detections, frame_w, frame_h,
            corridor_polygon=guidance["fixed_corridor_polygon_img"],
        )
        front_object = front.det if front.valid else None
        if render_output and out is not None:
            if not lane_enabled:
                draw_fixed_guidance(out, guidance=guidance)
            out = draw_detections(out, detections, copy_frame=False)

    fps.update()

    object_state = {"detections": detections, "leading_vehicle": front_object}
    perception_confidence = float(lane_state.confidence if lane_state is not None else 0.0)
    perception = PerceptionState(
        frame_id=None,
        timestamp=time.time(),
        lane_state=lane_state,
        object_state=object_state,
        detections=detections,
        perception_confidence=perception_confidence,
        raw_frame_shape=(frame_h, frame_w),
        ego_speed=ego_speed,
    )
    if metrics_logger is not None and getattr(metrics_logger, "enabled", False):
        from features.validation_metrics import extract_frame_metrics
        metrics_logger.log(extract_frame_metrics(
            lane_bounds=lane_bounds,
            red_center=red_center,
            render_corridor=render_corr,
            perception=perception,
            frame_id=getattr(fps, "frame_count", None),
        ))
    return out, steering_smooth, perception
