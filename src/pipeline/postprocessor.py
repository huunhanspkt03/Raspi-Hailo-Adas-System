"""Postprocessor — ADAS visualization overlays.

Render layers, in draw order:

  1. Soft green fill of the model ``lane_area``.
  2. Semantic line markings: dotted_line and solid_line.
  3. Yellow corridor borders + faint corridor fill.
  4. Red centerline.

Layers 1-2 are produced by ``pipeline.lane_overlay``. Layers 3-4 come
from a precomputed ``guidance`` dict (lane-area-aware when available,
camera-anchored otherwise).
"""
from __future__ import annotations

import cv2
import numpy as np

from features.lane_corridor import fixed_guidance_template
from features.lane_boundaries_render import render_boundary_polylines
from pipeline.lane_overlay import render_lane_overlay, render_lane_overlay_fast

_CENTER_GUIDANCE_COLOR = (0, 0, 255)
_BOUNDARY_LINE_SHADOW = (255, 255, 255)
_CORRIDOR_BOUNDARY_COLOR = (0, 255, 255)
_CORRIDOR_FILL_COLOR = np.array((0, 220, 220), dtype=np.float32).reshape(1, 1, 3)
_CORRIDOR_FILL_ALPHA = 0.10


def _polyline(out: np.ndarray, pts: np.ndarray, color: tuple[int, int, int],
              thickness: int, *, shadow: bool) -> None:
    curve = np.round(np.asarray(pts)).astype(np.int32).reshape(-1, 1, 2)
    if shadow:
        cv2.polylines(out, [curve], False, _BOUNDARY_LINE_SHADOW, thickness + 2, cv2.LINE_AA)
    cv2.polylines(out, [curve], False, color, thickness, cv2.LINE_AA)


def draw_fixed_guidance(
    out: np.ndarray,
    guidance: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Red centerline + yellow corridor from a precomputed guidance dict.

    When ``guidance`` is None, falls back to the camera-anchored
    ``fixed_guidance_template`` for ``out.shape``. Callers that have a
    lane-area-aware corridor must pass it in here so the rendered yellow
    lines match the polygon used for ``in_path`` testing. Mutates ``out``
    in place; returns it for chaining.
    """
    h, w = out.shape[:2]
    template = guidance if guidance is not None else fixed_guidance_template(h, w)
    polygon = template["fixed_corridor_polygon_img"]
    if len(polygon) >= 3:
        poly_int = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
        fill_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(fill_mask, [poly_int], 1)
        nz = fill_mask > 0
        if np.any(nz):
            patch = out[nz].astype(np.float32)
            out[nz] = (
                patch * (1.0 - _CORRIDOR_FILL_ALPHA)
                + _CORRIDOR_FILL_COLOR.ravel() * _CORRIDOR_FILL_ALPHA
            ).astype(np.uint8)
    for key in ("fixed_corridor_left_pts_img", "fixed_corridor_right_pts_img"):
        _polyline(out, template[key], _CORRIDOR_BOUNDARY_COLOR, 2, shadow=True)
    _polyline(out, template["fixed_centerline_pts_img"],
              _CENTER_GUIDANCE_COLOR, 2, shadow=True)
    return out


def draw_lane_debug(
    frame: np.ndarray,
    lane_area_mask: np.ndarray,
    semantics: dict[str, np.ndarray],
    class_probs: np.ndarray | None = None,
    lane_path: dict | None = None,
    lane_bounds: dict | None = None,
    *,
    fast: bool = False,
    guidance: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Render lane fill + semantic markings, then the guidance overlay.

    The guidance dict (red centerline + yellow corridor + polygon) is
    expected to come from ``lane_area_guidance_template`` so the rendered
    yellow lines exactly match the polygon used for ``in_path``. Falls
    back to the fixed camera-anchored template when ``guidance`` is None.
    ``lane_bounds`` may carry the tracked purple boundaries and red center; the
    red center is already injected into ``guidance`` by the caller.
    """
    _ = lane_path
    if fast:
        out = render_lane_overlay_fast(frame, lane_area_mask, semantics)
    else:
        out = render_lane_overlay(frame, lane_area_mask, semantics, class_probs)
    out = draw_fixed_guidance(out, guidance=guidance)
    if lane_bounds is not None:
        render_boundary_polylines(out, lane_bounds, thickness=3)
    return out


def draw_detections(
    frame: np.ndarray,
    detections: list[dict],
    *,
    copy_frame: bool = True,
) -> np.ndarray:
    out = frame.copy() if copy_frame else frame
    _draw_traffic_light_horizon(out, detections)
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = (0, 165, 255) if det.get("in_path") else (0, 255, 255)
        label = f"{det['label']} {det['score']:.2f}"
        if det.get("pose_valid") and det.get("distance_display_valid", True) and float(det.get("distance_m", -1.0)) > 0.0:
            label += f" {det['distance_m']:.1f}m"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        foot_x = det.get("foot_x", det.get("bottom_center_x"))
        foot_y = det.get("foot_y", det.get("bottom_center_y"))
        if foot_x is not None and foot_y is not None:
            cv2.circle(out, (int(foot_x), int(foot_y)), 3, color, thickness=-1)
        cv2.putText(out, label, (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def _draw_traffic_light_horizon(out: np.ndarray, detections: list[dict]) -> None:
    ys = [
        float(det["traffic_light_horizon_y"])
        for det in detections
        if "traffic_light_horizon_y" in det
    ]
    if not ys:
        return
    h, w = out.shape[:2]
    y = int(round(float(np.median(ys))))
    if y < 0 or y >= h:
        return
    color = (255, 180, 0)
    cv2.line(out, (0, y), (w - 1, y), color, 1, cv2.LINE_AA)
    cv2.putText(
        out, "traffic light horizon",
        (12, max(18, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
    )


def draw_hud(
    frame: np.ndarray,
    fps: float,
    lane_state,
    throttle: float,
    behavior_action: str,
    behavior_reason: str,
    hailo_active: bool,
    front_object: dict | None = None,
    *,
    copy_frame: bool = True,
    corridor: dict | None = None,
    ego_speed: dict | None = None,
) -> np.ndarray:
    out = frame.copy() if copy_frame else frame
    lines = [
        f"FPS: {fps:.1f}",
        f"Backend: {'Hailo' if hailo_active else 'ONNX'}",
        f"Steering: {lane_state.steering:+.3f}",
        f"Throttle: {throttle:.2f}",
        f"Behavior: {behavior_action} ({behavior_reason})",
    ]
    if lane_state.error_px is not None:
        lines.insert(2, f"Lane error(px): {lane_state.error_px:+d}")
    if lane_state.detected:
        source = getattr(lane_state, "center_source", None)
        source_txt = f" src={source}" if source else ""
        lines.append(
            f"Lane: conf={lane_state.confidence:.2f} "
            f"ctrl=({lane_state.control_x},{lane_state.control_y}){source_txt}"
        )
    if ego_speed is not None:
        if ego_speed.get("valid"):
            lines.append(
                f"Ego speed: {float(ego_speed.get('speed_kmh', 0.0)):.1f} km/h "
                f"conf={float(ego_speed.get('confidence', 0.0)):.2f}"
            )
        else:
            reason = ego_speed.get("reason", "invalid")
            lines.append(f"Ego speed: -- ({reason})")
    if corridor is not None and corridor.get("is_dynamic"):
        cstate = corridor.get("lane_state", "?")
        cconf = float(corridor.get("confidence", 0.0))
        npts = int(corridor.get("n_points", 0))
        miss = int(corridor.get("missing_streak", 0))
        offs = float(corridor.get("half_width_near_px", 0.0))
        cached = bool(corridor.get("from_cache", False))
        src = "cache" if cached else "live"
        lines.append(
            f"Corridor: {cstate} pts={npts} d_near={offs:.0f}px "
            f"conf={cconf:.2f} miss={miss} src={src}"
        )
    if front_object is not None and front_object.get("pose_valid"):
        lines.append(
            "Front: "
            f"{front_object['label']} "
            f"z={front_object['z_m']:.1f}m "
            f"x={front_object['x_m']:+.1f}m "
            f"bearing={front_object['bearing_deg']:+.1f}deg"
        )
    y = 28
    for line in lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)
        y += 28
    return out
