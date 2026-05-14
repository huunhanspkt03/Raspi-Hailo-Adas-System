"""Visual lane-geometry debug frames.

Runs the lane stack on selected video frames and saves PNG overlays that
expose the geometry sources behind the live red/yellow guidance.

Example:
    .venv/Scripts/python.exe tools/visual_debug_lane_geometry.py \
        --video test5_720p30.mp4 --model src/models/yolo26s_seg.onnx \
        --frames 98 961 --out-dir outputs/visual_lane_debug
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.controller import compute_steering  # noqa: E402
from features.ego_anchored_path import build_ego_anchored_path  # noqa: E402
from features.lane_boundary_tracker import LaneBoundaryTracker  # noqa: E402
from features.lane_boundary_tracker_contract import (  # noqa: E402
    SOURCE_DETECTED,
    SOURCE_HELD,
    SOURCE_LOST,
)
from features.lane_corridor_render import build_render_corridor  # noqa: E402
from features.lane_postprocess import decode_lane_components  # noqa: E402
from features.red_center_from_boundaries import compute_red_centerline  # noqa: E402
from pipeline.inference import create_inference  # noqa: E402
from pipeline.preprocessor import preprocess_segmentation  # noqa: E402


GREEN = (40, 190, 50)
PURPLE_DETECTED = (230, 30, 230)
PURPLE_HELD = (150, 90, 180)
PURPLE_LOST = (80, 80, 80)
LANE_AREA_CENTER = (255, 255, 0)
PURPLE_MIDPOINT = (255, 170, 80)
RAW_RED_TARGET = (0, 165, 255)
EGO_RED = (0, 0, 255)
YELLOW = (0, 255, 255)
EGO_LINE = (255, 255, 255)
LOOKAHEAD = (0, 255, 0)
TEXT = (255, 255, 255)
BLACK = (0, 0, 0)


def _as_pts(pts) -> np.ndarray | None:
    if pts is None:
        return None
    arr = np.asarray(pts, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return None
    return arr[:, :2]


def _draw_polyline(
    img: np.ndarray,
    pts,
    color: tuple[int, int, int],
    thickness: int,
    *,
    shadow: bool = True,
) -> None:
    arr = _as_pts(pts)
    if arr is None:
        return
    curve = np.round(arr).astype(np.int32).reshape(-1, 1, 2)
    if shadow:
        cv2.polylines(img, [curve], False, BLACK, thickness + 2, cv2.LINE_AA)
    cv2.polylines(img, [curve], False, color, thickness, cv2.LINE_AA)


def _draw_label(
    img: np.ndarray,
    text: str,
    xy: tuple[int, int],
    color: tuple[int, int, int] = TEXT,
    scale: float = 0.48,
) -> None:
    x, y = xy
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, BLACK, 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def _draw_endpoint_label(
    img: np.ndarray,
    pts,
    text: str,
    color: tuple[int, int, int],
    *,
    at_top: bool = True,
) -> None:
    arr = _as_pts(pts)
    if arr is None:
        return
    idx = int(np.argmin(arr[:, 1]) if at_top else np.argmax(arr[:, 1]))
    x = int(np.clip(round(float(arr[idx, 0])), 0, img.shape[1] - 1))
    y = int(np.clip(round(float(arr[idx, 1])), 14, img.shape[0] - 8))
    _draw_label(img, text, (x + 6, y), color)


def _tint_mask(img: np.ndarray, mask: np.ndarray | None) -> None:
    if mask is None or mask.size == 0 or not np.any(mask):
        return
    nz = mask > 0
    overlay = img.copy()
    overlay[nz] = GREEN
    cv2.addWeighted(overlay, 0.28, img, 0.72, 0, dst=img)


def _draw_tracker_sides(img: np.ndarray, track: dict) -> None:
    style = {
        SOURCE_DETECTED: (PURPLE_DETECTED, 4, "DETECTED"),
        SOURCE_HELD: (PURPLE_HELD, 2, "HELD hidden"),
        SOURCE_LOST: (PURPLE_LOST, 1, "LOST"),
    }
    for side in ("left", "right"):
        src = track.get(f"{side}_source", SOURCE_LOST)
        if src not in style:
            continue
        pts = track.get(f"{side}_pts_img")
        color, thick, label = style.get(src, style[SOURCE_LOST])
        _draw_polyline(img, pts, color, thick)
        _draw_endpoint_label(img, pts, f"{side} {label}", color)


def _draw_render_corridor(img: np.ndarray, render_corr: dict) -> None:
    if not isinstance(render_corr, dict) or not render_corr.get("valid"):
        return
    _draw_polyline(img, render_corr.get("yellow_left_pts_img"), YELLOW, 2)
    _draw_polyline(img, render_corr.get("yellow_right_pts_img"), YELLOW, 2)


def _build_ego_path(red_center: dict, frame_h: int, frame_w: int) -> tuple[np.ndarray | None, dict]:
    if not isinstance(red_center, dict) or not red_center.get("valid"):
        return None, {}
    target = _as_pts(red_center.get("center_pts_img"))
    if target is None:
        return None, {}
    ys = target[:, 1].astype(np.float32)
    xs, diag = build_ego_anchored_path(target, ys, frame_h, frame_w)
    ego_pts = np.stack([xs, ys], axis=1).astype(np.float32)
    return ego_pts, diag


def _draw_centers(
    img: np.ndarray,
    red_center: dict,
    ego_pts: np.ndarray | None,
    ego_diag: dict,
) -> None:
    _draw_polyline(img, red_center.get("lane_area_center_pts_img"),
                   LANE_AREA_CENTER, 1)
    _draw_endpoint_label(img, red_center.get("lane_area_center_pts_img"),
                         "lane_area center", LANE_AREA_CENTER)
    _draw_polyline(img, red_center.get("purple_center_pts_img"),
                   PURPLE_MIDPOINT, 1)
    _draw_endpoint_label(img, red_center.get("purple_center_pts_img"),
                         "purple midpoint target", PURPLE_MIDPOINT)
    _draw_polyline(img, red_center.get("center_pts_img"), RAW_RED_TARGET, 2)
    _draw_endpoint_label(img, red_center.get("center_pts_img"),
                         "raw red target", RAW_RED_TARGET, at_top=False)
    _draw_polyline(img, ego_pts, EGO_RED, 3)
    _draw_endpoint_label(img, ego_pts, "ego fitted red", EGO_RED)

    if ego_diag:
        cx = int(round(float(ego_diag["ego_path_control_x"])))
        cy = int(round(float(ego_diag["ego_path_control_y"])))
        cv2.circle(img, (cx, cy), 9, BLACK, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 7, LOOKAHEAD, -1, cv2.LINE_AA)
        _draw_label(img, f"lookahead ctrl=({cx},{cy})", (cx + 12, cy - 8),
                    LOOKAHEAD)


def _draw_references(img: np.ndarray, ego_diag: dict) -> None:
    h, w = img.shape[:2]
    ego_x = int(round(0.5 * w))
    cv2.line(img, (ego_x, int(0.52 * h)), (ego_x, h - 1),
             EGO_LINE, 1, cv2.LINE_AA)
    _draw_label(img, f"ego_x={ego_x}", (ego_x + 8, h - 18), EGO_LINE)
    if ego_diag:
        y = int(round(float(ego_diag["ego_path_lookahead_y"])))
        cv2.line(img, (0, y), (w - 1, y), (60, 180, 60), 1, cv2.LINE_AA)
        _draw_label(img, f"lookahead_y={y}", (8, max(20, y - 8)), LOOKAHEAD)


def _hud(
    img: np.ndarray,
    *,
    frame_idx: int,
    track: dict,
    red_center: dict,
    render_corr: dict,
    ego_diag: dict,
    lane_error: int | None,
    steering: float | None,
) -> None:
    left_src = track.get("left_source", "-")
    right_src = track.get("right_source", "-")
    lines = [
        f"frame={frame_idx}  lane_state={track.get('lane_state', '-')}  "
        f"render={render_corr.get('valid', False)} "
        f"shape={render_corr.get('ego_path_shape_source', '-')}",
        f"L={left_src} conf={float(track.get('left_confidence_final', 0.0)):.2f} "
        f"R={right_src} conf={float(track.get('right_confidence_final', 0.0)):.2f} "
        f"red_src={red_center.get('center_source', '-')} "
        f"red_conf={float(red_center.get('center_confidence', 0.0)):.2f} "
        f"area_conf={float(red_center.get('lane_area_confidence', 0.0)):.2f} "
        f"purple_conf={float(red_center.get('purple_confidence', 0.0)):.2f}",
        f"ctrl=({ego_diag.get('ego_path_control_x', '-')}," 
        f"{ego_diag.get('ego_path_control_y', '-')}) "
        f"lane_error={lane_error} steer={steering if steering is not None else '-'} "
        f"a={float(ego_diag.get('ego_path_a', 0.0)):.3e}",
        "color: magenta=DETECTED purple, violet=HELD hidden, orange=raw red target, red=ego path, cyan=lane-area center",
    ]
    box_h = 24 * len(lines) + 10
    cv2.rectangle(img, (0, 0), (img.shape[1], box_h), BLACK, -1)
    for i, line in enumerate(lines):
        _draw_label(img, line, (8, 22 + i * 24), TEXT, scale=0.55)


def _summary_dict(
    frame_idx: int,
    frame_h: int,
    frame_w: int,
    track: dict,
    red_center: dict,
    render_corr: dict,
    ego_diag: dict,
    lane_error: int | None,
    steering: float | None,
) -> dict:
    return {
        "frame_idx": int(frame_idx),
        "frame_w": int(frame_w),
        "frame_h": int(frame_h),
        "ego_x": 0.5 * float(frame_w),
        "lane_state": track.get("lane_state"),
        "left_source": track.get("left_source"),
        "right_source": track.get("right_source"),
        "left_confidence_final": float(track.get("left_confidence_final", 0.0)),
        "right_confidence_final": float(track.get("right_confidence_final", 0.0)),
        "center_source": red_center.get("center_source"),
        "center_confidence": float(red_center.get("center_confidence", 0.0)),
        "lane_area_confidence": float(red_center.get("lane_area_confidence", 0.0)),
        "purple_confidence": float(red_center.get("purple_confidence", 0.0)),
        "road_type": red_center.get("road_type"),
        "render_corridor_valid": bool(render_corr.get("valid", False)),
        "render_shape_source": render_corr.get("ego_path_shape_source"),
        "render_shape_reason": render_corr.get("ego_path_shape_reason"),
        "control_x": (
            None if "ego_path_control_x" not in ego_diag
            else float(ego_diag["ego_path_control_x"])
        ),
        "control_y": (
            None if "ego_path_control_y" not in ego_diag
            else int(ego_diag["ego_path_control_y"])
        ),
        "lane_error_px": lane_error,
        "steering": steering,
        "ego_path_a": float(ego_diag.get("ego_path_a", 0.0)),
        "ego_path_reason": ego_diag.get("ego_path_reason"),
        "lane_width_px": (
            None if track.get("lane_width_px") is None
            else float(track.get("lane_width_px"))
        ),
        "purple_lane_width_px": (
            None if track.get("purple_lane_width_px") is None
            else float(track.get("purple_lane_width_px"))
        ),
        "purple_lane_width_valid": bool(track.get("purple_lane_width_valid", False)),
    }


def _render_debug(
    frame: np.ndarray,
    lane_area: np.ndarray,
    track: dict,
    red_center: dict,
    render_corr: dict,
    frame_idx: int,
) -> tuple[np.ndarray, dict]:
    h, w = frame.shape[:2]
    out = frame.copy()
    _tint_mask(out, lane_area)
    ego_pts, ego_diag = _build_ego_path(red_center, h, w)
    lane_error = steering = None
    if ego_diag and "ego_path_control_x" in ego_diag:
        lane_error, steering = compute_steering(
            int(round(float(ego_diag["ego_path_control_x"]))),
            w,
            deadband_px=0,
            max_steering_value=1.0,
        )

    _draw_render_corridor(out, render_corr)
    _draw_tracker_sides(out, track)
    _draw_centers(out, red_center, ego_pts, ego_diag)
    _draw_references(out, ego_diag)
    _hud(
        out,
        frame_idx=frame_idx,
        track=track,
        red_center=red_center,
        render_corr=render_corr,
        ego_diag=ego_diag,
        lane_error=lane_error,
        steering=steering,
    )
    summary = _summary_dict(
        frame_idx, h, w, track, red_center, render_corr,
        ego_diag, lane_error, steering,
    )
    return out, summary


def _parse_frames(values: list[str]) -> list[int]:
    frames: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            frames.append(max(0, int(part)))
    return sorted(set(frames))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="src/models/yolo26s_seg.onnx")
    ap.add_argument("--frames", nargs="+", required=True,
                    help="Frame indices to save, e.g. --frames 98 961 or 98,961.")
    ap.add_argument("--out-dir", default="outputs/visual_lane_debug")
    ap.add_argument("--seg-size", type=int, default=640)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    video = Path(args.video)
    if not video.exists():
        print(f"video not found: {video}", file=sys.stderr)
        return 2
    model = Path(args.model)
    if not model.exists():
        print(f"model not found: {model}", file=sys.stderr)
        return 2

    frames = _parse_frames(args.frames)
    if not frames:
        print("no frames requested", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"cannot open video: {video}", file=sys.stderr)
        return 1

    tracker = LaneBoundaryTracker()
    wanted = set(frames)
    last_frame = max(frames)
    saved: list[dict] = []
    frame_idx = -1
    print(f"video={video} model={model} frames={frames}")

    with create_inference({"lane_segmentation": str(model)}, force_onnx=True) as infer:
        while frame_idx < last_frame:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            h, w = frame.shape[:2]
            prep = preprocess_segmentation(frame, int(args.seg_size))
            raw = infer.infer("lane_segmentation", prep)
            _, _, semantics, lane_area = decode_lane_components(raw, h, w)
            track = tracker.update(semantics, lane_area, h, w)
            red_center = compute_red_centerline(lane_area, track, h, w)
            render_corr = build_render_corridor(
                track, h, w, red_center=red_center,
            )
            if frame_idx not in wanted:
                if frame_idx % 100 == 0 and frame_idx > 0:
                    print(f"processed frame {frame_idx}/{last_frame}")
                continue

            debug_img, summary = _render_debug(
                frame, lane_area, track, red_center, render_corr, frame_idx,
            )
            stem = f"{video.stem}_frame_{frame_idx:05d}"
            png_path = out_dir / f"{stem}.png"
            json_path = out_dir / f"{stem}.json"
            cv2.imwrite(str(png_path), debug_img)
            json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            saved.append({"png": str(png_path), "json": str(json_path), **summary})
            print(
                f"saved {png_path}  ctrl=({summary['control_x']},"
                f"{summary['control_y']}) err={summary['lane_error_px']} "
                f"src={summary['center_source']} "
                f"L={summary['left_source']} R={summary['right_source']}"
            )

    cap.release()
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    print(f"wrote manifest: {manifest}")
    if len(saved) != len(wanted):
        missing = sorted(wanted - {int(item["frame_idx"]) for item in saved})
        print(f"missing frames: {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
