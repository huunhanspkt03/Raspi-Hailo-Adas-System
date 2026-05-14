"""Traffic-light pose estimation and ground-anchor group fusion."""
from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

from loguru import logger

from features.object_localizer_priors import (
    ObjectPrior,
    _HORIZON_GUARD_PX,
    _TRAFFIC_LIGHT_GROUP_GAP_RATIO,
    _TRAFFIC_LIGHT_GROUP_MIN_GAP_PX,
    _TRAFFIC_LIGHT_LABELS,
    _TRAFFIC_LIGHT_STABLE_DY_PX,
)

if TYPE_CHECKING:
    from features.object_localizer import ObjectLocalizer

_DEBUG_TRAFFIC_LIGHT = os.getenv("ADAS_TRAFFIC_LIGHT_DEBUG", "0") == "1"


def estimate_traffic_light_pose(
    loc: "ObjectLocalizer",
    det: dict,
    prior: ObjectPrior,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    bbox_w: float,
    bbox_h: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> bool:
    center_x = x1 + bbox_w * 0.5
    center_y = y1 + bbox_h * 0.5
    det["traffic_light_center_x"] = int(round(center_x))
    det["traffic_light_center_y"] = int(round(center_y))
    det["foot_x"] = int(round(center_x))
    det["foot_y"] = int(round(center_y))
    det["foot_source"] = "traffic_light_center"
    det["traffic_light_ground_anchor_valid"] = False

    # Positive pitch means the camera looks upward, moving the horizon down
    # in image coordinates (OpenCV y increases downward).
    horizon_y = cy + fy * math.tan(loc.pitch_rad)
    dy = horizon_y - center_y
    ray_slope_y = (cy - center_y) / max(fy, 1e-3)
    elevation_rad = math.atan(ray_slope_y) + loc.pitch_rad
    det["traffic_light_horizon_y"] = float(horizon_y)
    det["traffic_light_elevation_deg"] = float(math.degrees(elevation_rad))
    z_size = prior.height_m * fy / bbox_h
    height_candidates = _height_distance_candidates(loc, elevation_rad)
    track_key = _traffic_light_track_key(loc, det, center_x, center_y)
    previous_mount = _previous_mount_source(loc, track_key)
    if dy <= _HORIZON_GUARD_PX or elevation_rad <= 0.0 or not height_candidates:
        det["distance_source"] = "traffic_light_below_horizon"
        det["traffic_light_track_key"] = track_key
        det["traffic_light_dy_px"] = float(dy)
        _log_traffic_light_debug(
            det, center_x, center_y, bbox_w, bbox_h, horizon_y, dy,
            elevation_rad,
            z_height=None, z_size=None, ground_z=None,
        )
        return False
    raw: dict[str, float] = {}
    confidence = 0.62
    mount_source, mount_height, z_height = _select_height_candidate(
        loc, height_candidates, z_size, bbox_h, previous_mount,
    )
    for source_name, candidate_height, candidate_z in height_candidates:
        raw[f"traffic_light_height_{source_name}"] = candidate_z
        raw[f"traffic_light_mount_{source_name}_m"] = candidate_height
    raw["traffic_light_height"] = z_height
    raw["traffic_light_size"] = z_size
    raw["traffic_light_mount_selected_m"] = mount_height
    ground_z, ground_confidence = _ground_anchor_from_elevation(
        loc, z_height, z_size, bbox_h, dy,
    )
    if ground_z is not None:
        raw["traffic_light_ground_anchor"] = ground_z

    # The pinhole size method depends only on fy and the housing prior
    # (~0.6 m). The elevation method divides by tan(elev) of a 3-6 deg
    # angle, so a 1 deg pitch error or a 30% mount-height error inflates
    # it 2-6x. Prefer size when the bbox is well-resolved; reserve the
    # elevation method for tiny bboxes where pixel quantization makes
    # size unreliable.
    source = "traffic_light_size"
    z = z_size
    if bbox_h < 12.0 and z_height <= loc.traffic_light_max_distance_m:
        z = z_height
        source = "traffic_light_height"
    elif dy < _TRAFFIC_LIGHT_STABLE_DY_PX:
        z = max(z_size, loc.traffic_light_size_fallback_min_m)
        source = "traffic_light_size_floor"
        confidence = 0.32

    z = min(max(z, loc.traffic_light_min_distance_m), loc.traffic_light_max_distance_m)
    u = float(center_x)
    pose_z = ground_z if ground_z is not None else z
    pose_confidence = ground_confidence if ground_z is not None else confidence
    x_m = ((u - cx) / max(fx, 1e-3)) * pose_z
    pose_z, x_m, pose_confidence = _smooth_traffic_light_pose(
        loc, det, track_key, center_x, center_y, pose_z, x_m,
        pose_confidence, mount_source,
    )
    if ground_z is not None:
        ground_z = pose_z

    det["pose_valid"] = True
    det["x_m"] = float(x_m)
    det["z_m"] = float(pose_z)
    det["distance_m"] = -1.0
    det["ground_distance_m"] = -1.0
    det["bearing_deg"] = float(math.degrees(math.atan2(x_m, pose_z)))
    det["distance_confidence"] = float(pose_confidence)
    det["distance_source"] = f"{source}_no_ground_anchor"
    det["traffic_light_height_estimate_m"] = float(z_height)
    det["traffic_light_track_key"] = track_key
    det["traffic_light_mount_height_selected_m"] = float(mount_height)
    det["traffic_light_mount_source"] = mount_source
    det["traffic_light_ground_anchor_z_m"] = ground_z if ground_z is not None else -1.0
    det["distance_display_valid"] = False
    if ground_z is not None:
        det["z_m"] = float(ground_z)
        det["distance_m"] = float(ground_z)
        det["ground_distance_m"] = float(ground_z)
        det["bearing_deg"] = float(math.degrees(math.atan2(x_m, ground_z)))
        det["traffic_light_ground_anchor_valid"] = True
        det["distance_display_valid"] = True
        det["distance_source"] = f"traffic_light_{mount_source}_height_ground_anchor"
    det["traffic_light_dy_px"] = float(dy)
    det["distance_raw_m"] = raw
    det["in_path"] = False
    _log_traffic_light_debug(
        det, center_x, center_y, bbox_w, bbox_h, horizon_y, dy,
        elevation_rad,
        z_height=z_height, z_size=z_size, ground_z=ground_z,
    )
    return True


def _height_distance_candidates(
    loc: "ObjectLocalizer",
    elevation_rad: float,
) -> list[tuple[str, float, float]]:
    tan_elev = math.tan(elevation_rad)
    if tan_elev <= 1e-6:
        return []
    heights = [("high", float(loc.traffic_light_mount_height_m))]
    low_height = getattr(loc, "traffic_light_low_mount_height_m", None)
    if low_height is not None:
        heights.append(("low", float(low_height)))
    candidates: list[tuple[str, float, float]] = []
    seen: set[float] = set()
    for source_name, mount_height in heights:
        if mount_height in seen:
            continue
        seen.add(mount_height)
        height_diff = mount_height - loc.camera_height_m
        if height_diff <= 0.1:
            continue
        candidates.append((source_name, mount_height, height_diff / tan_elev))
    return candidates


def _select_height_candidate(
    loc: "ObjectLocalizer",
    candidates: list[tuple[str, float, float]],
    z_size: float,
    bbox_h: float,
    previous_mount: str | None,
) -> tuple[str, float, float]:
    if len(candidates) == 1:
        return candidates[0]
    by_source = {item[0]: item for item in candidates}
    if not (math.isfinite(z_size) and z_size > 0.0 and bbox_h >= 10.0):
        if previous_mount in by_source:
            return by_source[previous_mount]
        return candidates[0]

    def score(item: tuple[str, float, float]) -> float:
        return abs(math.log(max(item[2], 1e-6) / z_size))

    best = min(candidates, key=score)
    if previous_mount in by_source:
        prev = by_source[previous_mount]
        switch_margin = math.log(max(float(loc.traffic_light_mount_switch_ratio), 1.01))
        if score(prev) <= score(best) + switch_margin:
            return prev
    return best


def _traffic_light_track_key(
    loc: "ObjectLocalizer",
    det: dict,
    center_x: float,
    center_y: float,
) -> str:
    tid = det.get("track_id")
    tracks = getattr(loc, "_traffic_light_tracks", {})
    if tid is not None:
        try:
            key = f"id:{int(tid)}"
            if key in tracks:
                return key
        except (TypeError, ValueError):
            key = None
        else:
            key = key
    else:
        key = None

    label = str(det.get("label", "traffic_light"))
    best_key = None
    best_d2 = float("inf")
    for state_key, state in tracks.items():
        if state.get("label") != label:
            continue
        dx = center_x - float(state.get("cx", center_x))
        dy = center_y - float(state.get("cy", center_y))
        d2 = dx * dx + dy * dy
        if d2 < best_d2 and abs(dx) <= 80.0 and abs(dy) <= 80.0:
            best_key = state_key
            best_d2 = d2
    if best_key is not None:
        return best_key
    if key is not None:
        return key
    return f"pos:{label}:{int(round(center_x / 80.0))}:{int(round(center_y / 80.0))}"


def _previous_mount_source(loc: "ObjectLocalizer", track_key: str) -> str | None:
    state = getattr(loc, "_traffic_light_tracks", {}).get(track_key)
    if not state:
        return None
    mount_source = state.get("mount_source")
    return str(mount_source) if mount_source else None


def _smooth_traffic_light_pose(
    loc: "ObjectLocalizer",
    det: dict,
    track_key: str,
    center_x: float,
    center_y: float,
    z: float,
    x_m: float,
    confidence: float,
    mount_source: str,
) -> tuple[float, float, float]:
    tracks = getattr(loc, "_traffic_light_tracks", {})
    if not loc.smoothing_enabled:
        tracks[track_key] = {
            "z": z, "x_m": x_m, "confidence": confidence,
            "cx": center_x, "cy": center_y, "label": det.get("label"),
            "mount_source": mount_source,
            "frame": getattr(loc, "_traffic_light_frame_seq", 0),
        }
        return z, x_m, confidence

    prev = tracks.get(track_key)
    alpha = min(max(float(loc.traffic_light_smoothing_alpha), 0.0), 1.0)
    if prev is None:
        tracks[track_key] = {
            "z": z, "x_m": x_m, "confidence": confidence,
            "cx": center_x, "cy": center_y, "label": det.get("label"),
            "mount_source": mount_source,
            "frame": getattr(loc, "_traffic_light_frame_seq", 0),
        }
        return z, x_m, confidence

    prev_z = float(prev.get("z", z))
    prev_x = float(prev.get("x_m", x_m))
    max_step = max(float(loc.traffic_light_max_depth_step_m), 0.5)
    z_limited = min(max(z, prev_z - max_step), prev_z + max_step)
    z_smooth = prev_z + alpha * (z_limited - prev_z)
    x_smooth = prev_x + alpha * (x_m - prev_x)
    conf_smooth = max(float(confidence), 0.85 * float(prev.get("confidence", 0.0)))
    tracks[track_key] = {
        "z": z_smooth, "x_m": x_smooth, "confidence": conf_smooth,
        "cx": center_x, "cy": center_y, "label": det.get("label"),
        "mount_source": mount_source,
        "frame": getattr(loc, "_traffic_light_frame_seq", 0),
    }
    return z_smooth, x_smooth, conf_smooth


def _log_traffic_light_debug(
    det: dict,
    center_x: float,
    center_y: float,
    bbox_w: float,
    bbox_h: float,
    horizon_y: float,
    dy: float,
    elevation_rad: float,
    *,
    z_height: float | None,
    z_size: float | None,
    ground_z: float | None,
) -> None:
    if not _DEBUG_TRAFFIC_LIGHT:
        return
    logger.info(
        "TLDBG label={} score={:.2f} box=({:.0f},{:.0f},{:.0f},{:.0f}) "
        "center=({:.1f},{:.1f}) bbox=({:.1f}x{:.1f}) horizon_y={:.1f} "
        "dy={:.1f} elev={:.2f}deg mount={} mount_src={} "
        "z_height={} z_size={} ground_z={} valid={} display={} source={}",
        det.get("label", "?"),
        float(det.get("score", 0.0)),
        float(det.get("box", [0, 0, 0, 0])[0]),
        float(det.get("box", [0, 0, 0, 0])[1]),
        float(det.get("box", [0, 0, 0, 0])[2]),
        float(det.get("box", [0, 0, 0, 0])[3]),
        center_x,
        center_y,
        bbox_w,
        bbox_h,
        horizon_y,
        dy,
        math.degrees(elevation_rad),
        _fmt_m(det.get("traffic_light_mount_height_selected_m")),
        det.get("traffic_light_mount_source", "none"),
        _fmt_m(z_height),
        _fmt_m(z_size),
        _fmt_m(ground_z),
        bool(det.get("pose_valid", False)),
        bool(det.get("distance_display_valid", False)),
        det.get("distance_source", "none"),
    )


def _fmt_m(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "none"
    return f"{float(value):.1f}m"


def _ground_anchor_from_elevation(
    loc: "ObjectLocalizer",
    z_height: float,
    z_size: float,
    bbox_h: float,
    dy: float,
) -> tuple[float | None, float]:
    """Estimate ground distance from elevation: z = delta_h / tan(theta)."""
    if not math.isfinite(z_height):
        return None, 0.0
    if z_height < loc.traffic_light_min_distance_m:
        return None, 0.0
    if z_height > loc.traffic_light_max_distance_m:
        return None, 0.0

    confidence = 0.58
    if dy < _TRAFFIC_LIGHT_STABLE_DY_PX:
        confidence = 0.34
    if bbox_h < 8.0:
        confidence = min(confidence, 0.40)
    if math.isfinite(z_size) and z_size > 0.0:
        ratio = max(z_height / z_size, z_size / z_height)
        if ratio > 2.5:
            confidence = min(confidence, 0.36)
    return float(z_height), float(confidence)


def fuse_traffic_light_groups(
    loc: "ObjectLocalizer", detections: list[dict], frame_w: int, frame_h: int,
) -> None:
    lights = [
        det for det in detections
        if det.get("label") in _TRAFFIC_LIGHT_LABELS and det.get("pose_valid")
    ]
    if len(lights) < 2:
        return

    fx, _fy, cx, _cy = loc._make_intrinsics(frame_w, frame_h)
    group_gap_px = max(
        _TRAFFIC_LIGHT_GROUP_MIN_GAP_PX,
        _TRAFFIC_LIGHT_GROUP_GAP_RATIO * float(frame_w),
    )
    lights.sort(key=lambda det: float(det.get("traffic_light_center_x", det.get("foot_x", 0))))

    groups: list[list[dict]] = []
    current: list[dict] = []
    last_x = 0.0
    for det in lights:
        center_x = float(det.get("traffic_light_center_x", det.get("foot_x", 0)))
        if current and center_x - last_x > group_gap_px:
            groups.append(current)
            current = []
        current.append(det)
        last_x = center_x
    if current:
        groups.append(current)

    for idx, group in enumerate(groups):
        if len(group) < 2:
            continue
        ground_estimates = [
            float(det["traffic_light_ground_anchor_z_m"]) for det in group
            if math.isfinite(float(det.get("traffic_light_ground_anchor_z_m", -1.0)))
            and float(det.get("traffic_light_ground_anchor_z_m", -1.0)) > 0.0
        ]
        pool = ground_estimates
        if not pool:
            continue

        fused_z = min(
            max(float(sorted(pool)[len(pool) // 2]), loc.traffic_light_min_distance_m),
            loc.traffic_light_max_distance_m,
        )
        for det in group:
            old_z = float(det.get("z_m", fused_z))
            old_source = str(det.get("distance_source", "traffic_light"))
            center_x = float(det.get("traffic_light_center_x", det.get("foot_x", 0)))
            x_m = ((center_x - cx) / max(fx, 1e-3)) * fused_z
            raw = dict(det.get("distance_raw_m", {}))
            raw["traffic_light_ground_anchor_group"] = fused_z
            det["traffic_light_individual_z_m"] = old_z
            det["traffic_light_group_id"] = idx
            det["traffic_light_ground_anchor_valid"] = True
            det["x_m"] = float(x_m)
            det["z_m"] = float(fused_z)
            det["distance_m"] = float(fused_z)
            det["ground_distance_m"] = float(fused_z)
            det["bearing_deg"] = float(math.degrees(math.atan2(x_m, fused_z)))
            det["distance_confidence"] = max(float(det.get("distance_confidence", 0.0)), 0.55)
            det["distance_display_valid"] = True
            det["distance_source"] = (
                old_source if old_source.endswith("+traffic_light_group")
                else f"{old_source}+traffic_light_ground_anchor_group"
            )
            det["distance_raw_m"] = raw
