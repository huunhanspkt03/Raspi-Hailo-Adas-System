"""Preview tool for the Step-4 red target center line.

Runs YOLO26-seg on a video, drives ``LaneBoundaryTracker``, calls
``compute_red_centerline``, and renders a debug overlay frame-by-frame.
Live ``cv2.imshow`` window when ``--display`` is set; ``--save-video``
writes an MP4 alongside (works headless). Press ``q`` or ``ESC`` to quit.

This is a Step-4 validation tool only — it does NOT touch the live
pipeline, postprocessor, steering, or any Step-5 green-corridor work.

Examples::
    python tools/preview_red_center_from_boundaries.py --video test4_720p30.mp4 --display
    python tools/preview_red_center_from_boundaries.py --video test4_720p30.mp4 \
        --no-display --save-video --out outputs/red_center_test4.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.lane_boundary_tracker import LaneBoundaryTracker  # noqa: E402
from features.lane_boundary_tracker_contract import (  # noqa: E402
    SOURCE_DETECTED, SOURCE_HELD,
)
from features.lane_corridor_render import build_render_corridor  # noqa: E402
from features.lane_postprocess import decode_lane_components  # noqa: E402
from features.red_center_from_boundaries import compute_red_centerline  # noqa: E402
from pipeline.inference import create_inference  # noqa: E402
from pipeline.preprocessor import preprocess_segmentation  # noqa: E402

PURPLE_BGR = (180, 60, 180)
HELD_BGR = (140, 70, 140)
LANE_AREA_GUIDE_BGR = (0, 220, 220)   # cyan — lane_area-only evidence
PURPLE_MID_BGR = (255, 180, 90)       # light blue — purple midpoint evidence
YELLOW_BGR = (0, 255, 255)            # inner safe-corridor boundaries
RED_BGR = (40, 40, 235)               # final red center line
LANE_AREA_TINT_BGR = (60, 200, 60)


def _draw_curve(img, pts, color, thickness=2):
    if pts is None or len(pts) < 2:
        return
    curve = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [curve], False, color, thickness, cv2.LINE_AA)


def _draw_purple_sides(img, track):
    for side, color in (("left", PURPLE_BGR), ("right", PURPLE_BGR)):
        src = track.get(f"{side}_source")
        pts = track.get(f"{side}_pts_img")
        if src == SOURCE_DETECTED:
            _draw_curve(img, pts, color, 3)
        elif src == SOURCE_HELD:
            _draw_curve(img, pts, HELD_BGR, 1)


def _tint_lane_area(img, lane_area):
    if lane_area is None or not np.any(lane_area):
        return
    mask = (lane_area > 0)
    overlay = img.copy()
    overlay[mask] = LANE_AREA_TINT_BGR
    cv2.addWeighted(overlay, 0.20, img, 0.80, 0, dst=img)


def _hud(img, frame_idx, track, red):
    lines = [
        f"frame={frame_idx}  lane_state={track.get('lane_state', '-')}  "
        f"road={red.get('road_type', '-')}",
        f"L src={track.get('left_source','-'):<8s} conf={track.get('left_confidence_final',0.0):.2f} "
        f"R src={track.get('right_source','-'):<8s} conf={track.get('right_confidence_final',0.0):.2f}",
        f"red_src={red.get('center_source','-'):<22s} red_conf={red.get('center_confidence',0.0):.2f}  "
        f"area_conf={red.get('lane_area_confidence',0.0):.2f}  purple_conf={red.get('purple_confidence',0.0):.2f}",
        f"valid={red.get('valid', False)}  reason={red.get('reason','')}",
    ]
    cv2.rectangle(img, (0, 0), (img.shape[1], 22 * len(lines) + 8), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (8, 20 + 22 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)


def _render(frame, lane_area, track, red, frame_idx, render_corr):
    out = frame.copy()
    _tint_lane_area(out, lane_area)
    if render_corr.get("valid"):
        # Purple sides from the render corridor when available.
        _draw_curve(out, render_corr["purple_left_pts_img"], PURPLE_BGR, 3)
        _draw_curve(out, render_corr["purple_right_pts_img"], PURPLE_BGR, 3)
        # Yellow inner-corridor boundaries — constant inward x-offset from
        # purple, NOT a normal-offset from the centerline.
        _draw_curve(out, render_corr["yellow_left_pts_img"], YELLOW_BGR, 2)
        _draw_curve(out, render_corr["yellow_right_pts_img"], YELLOW_BGR, 2)
    else:
        # Fallback: tracker per-side polylines (may show only one side).
        _draw_purple_sides(out, track)
    _draw_curve(out, red.get("lane_area_center_pts_img"), LANE_AREA_GUIDE_BGR, 1)
    _draw_curve(out, red.get("purple_center_pts_img"), PURPLE_MID_BGR, 1)
    _draw_curve(out, red.get("center_pts_img"), RED_BGR, 3)
    _hud(out, frame_idx, track, red)
    return out


def _resolve_video(arg_path: str | None) -> Path | None:
    if arg_path:
        p = Path(arg_path)
        return p if p.exists() else None
    for cand in sorted(ROOT.glob("test4*.mp4")):
        if cand.is_file():
            return cand
    return None


def _open_writer(out_path: Path, w: int, h: int, fps: float) -> cv2.VideoWriter:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps if fps > 0 else 30.0, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open writer at {out_path}")
    return writer


def _try_imshow(title: str, img: np.ndarray) -> bool:
    try:
        cv2.imshow(title, img)
        return True
    except cv2.error:
        return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--video", default=None,
                   help="Path to a video file. Defaults to the first test4*.mp4 in repo root.")
    p.add_argument("--model", default="src/models/yolo26s_seg.onnx")
    p.add_argument("--display", dest="display", action="store_true",
                   help="Open a live cv2 window. Press q/ESC to quit.")
    p.add_argument("--no-display", dest="display", action="store_false",
                   help="Disable display (headless / save-video only).")
    p.add_argument("--save-video", action="store_true",
                   help="Write the annotated video to --out.")
    p.add_argument("--out", default="outputs/red_center_preview.mp4")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0,
                   help="0 = run until EOF.")
    p.add_argument("--stride", type=int, default=1,
                   help="Process every Nth frame (>=1).")
    p.add_argument("--resize-display", type=int, default=0,
                   help="Optional max display width (preserves aspect).")
    p.set_defaults(display=True)
    return p.parse_args(argv)


def _resize_for_display(img: np.ndarray, max_w: int) -> np.ndarray:
    if max_w <= 0 or img.shape[1] <= max_w:
        return img
    scale = max_w / float(img.shape[1])
    new_size = (max_w, max(1, int(round(img.shape[0] * scale))))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    video = _resolve_video(args.video)
    if video is None:
        print(f"could not resolve a video (passed {args.video!r}; no test4*.mp4 in {ROOT})",
              file=sys.stderr)
        return 2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"cannot open {video}", file=sys.stderr)
        return 1
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start_frame))

    writer = _open_writer(Path(args.out), src_w, src_h, fps) if args.save_video else None
    title = f"red_center :: {video.name}"
    display_ok = bool(args.display)
    display_warned = False
    tracker = LaneBoundaryTracker()
    stride = max(1, int(args.stride))
    processed = 0
    frame_idx = int(args.start_frame)

    print(f"video       = {video}")
    print(f"display     = {args.display}    save_video={args.save_video}    out={args.out}")
    print(f"start_frame = {args.start_frame}    max_frames={args.max_frames}    stride={stride}")
    print("press q or ESC in the window to quit (when --display is on)")

    with create_inference({"lane_segmentation": args.model}, force_onnx=True) as infer:
        while True:
            if args.max_frames and processed >= args.max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if (frame_idx - int(args.start_frame)) % stride != 0:
                frame_idx += 1
                continue
            h, w = frame.shape[:2]
            prep = preprocess_segmentation(frame, 640)
            raw = infer.infer("lane_segmentation", prep)
            _, _, semantics, lane_area = decode_lane_components(raw, h, w)
            track = tracker.update(semantics, lane_area, h, w)
            red = compute_red_centerline(lane_area, track, h, w)
            render_corr = build_render_corridor(track, h, w, red_center=red)
            rendered = _render(frame, lane_area, track, red, frame_idx, render_corr)
            if writer is not None:
                writer.write(rendered)
            if display_ok:
                disp = _resize_for_display(rendered, int(args.resize_display))
                if not _try_imshow(title, disp):
                    if not display_warned:
                        print("display unavailable (headless?); continuing without window. "
                              "Use --save-video --out PATH to capture results.",
                              file=sys.stderr)
                        display_warned = True
                    display_ok = False
                else:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        print("quit requested")
                        break
            processed += 1
            frame_idx += 1
            if processed % 60 == 0:
                print(f"  processed {processed} frames (frame_idx={frame_idx})")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"wrote {processed} frames to {args.out}")
    if args.display:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
