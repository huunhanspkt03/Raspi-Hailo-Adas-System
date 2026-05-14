"""Multi-frame preview of LaneBoundaryTracker — exercises the Step-3 contract.

Writes one PNG per frame: tracked frame on the left + debug overlay on the
right. HUD shows per-side ``source`` (DETECTED / HELD / LOST),
``confidence_final``, missing/held streaks, and the aggregate ``lane_state``
plus ``lane_width_px``.

Usage::
    python tools/preview_lane_boundary_tracker.py --video test1_720p30.mp4 \
        --start 85 --count 40
    python tools/preview_lane_boundary_tracker.py --video test1_720p30.mp4 \
        --start 90 --count 6 --suppress-left
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.lane_boundaries_render import (  # noqa: E402
    PURPLE_BGR, render_full_debug, render_lane_area_edges,
)
from features.lane_boundary_tracker import LaneBoundaryTracker  # noqa: E402
from features.lane_boundary_tracker_contract import (  # noqa: E402
    SOURCE_DETECTED,
)
from features.lane_postprocess import decode_lane_components  # noqa: E402
from pipeline.inference import create_inference  # noqa: E402
from pipeline.preprocessor import preprocess_segmentation  # noqa: E402

def _hud_lines(out: dict, frame_idx: int) -> list[str]:
    return [
        f"frame={frame_idx}  lane_state={out['lane_state']}  "
        f"boundary_conf={out['boundary_confidence']:.2f}  "
        f"lane_w={('%.0f' % out['lane_width_px']) if out['lane_width_valid'] else '-'}",
        f"L  src={out['left_source']:<8s} conf={out['left_confidence_final']:.2f} "
        f"raw_miss={out['left_raw_missing_streak']:>2d} "
        f"held={out['left_held_streak']:>2d} out_miss={out['left_output_missing_streak']:>2d}",
        f"R  src={out['right_source']:<8s} conf={out['right_confidence_final']:.2f} "
        f"raw_miss={out['right_raw_missing_streak']:>2d} "
        f"held={out['right_held_streak']:>2d} out_miss={out['right_output_missing_streak']:>2d}",
    ]


def _draw_curve(img: np.ndarray, pts: np.ndarray, color: tuple[int, int, int],
                thickness: int = 2) -> None:
    if pts is None or len(pts) < 2:
        return
    curve = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [curve], False, color, thickness, cv2.LINE_AA)


def _render_tracked_panel(frame: np.ndarray, out: dict, frame_idx: int) -> np.ndarray:
    base = frame.copy()
    # Color per source: DETECTED->purple, HELD->dim purple, LOST->nothing.
    for side in ("left", "right"):
        src = out[f"{side}_source"]
        pts = out[f"{side}_pts_img"]
        if src == SOURCE_DETECTED:
            _draw_curve(base, pts, PURPLE_BGR, 2)
        else:  # HELD
            if pts is not None and len(pts) >= 2:
                _draw_curve(base, pts, (140, 70, 140), 1)
    render_lane_area_edges(base, out)
    cv2.rectangle(base, (0, 0), (base.shape[1], 78), (0, 0, 0), -1)
    for i, line in enumerate(_hud_lines(out, frame_idx)):
        cv2.putText(base, line, (8, 22 + 22 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    return base


def _suppress_left_lines(semantics: dict[str, np.ndarray]) -> None:
    if not semantics:
        return
    first = next(iter(semantics.values()))
    w = first.shape[1]
    for k in ("dotted_line", "solid_line"):
        m = semantics.get(k)
        if m is not None:
            m[:, :w // 2] = 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=int, default=80)
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--model", default="src/models/yolo26s_seg.onnx")
    ap.add_argument("--out-dir", default="outputs/tracker_preview")
    ap.add_argument("--suppress-left", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"cannot open {args.video}", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, args.start))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracker = LaneBoundaryTracker()
    rows: list[tuple] = []
    with create_inference({"lane_segmentation": args.model}, force_onnx=True) as infer:
        for i in range(args.count):
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            prep = preprocess_segmentation(frame, 640)
            raw = infer.infer("lane_segmentation", prep)
            _, _, semantics, lane_area = decode_lane_components(raw, h, w)
            if args.suppress_left:
                _suppress_left_lines(semantics)
            out = tracker.update(semantics, lane_area, h, w)
            tracked = _render_tracked_panel(frame, out, args.start + i)
            comp = frame.copy()
            render_full_debug(comp, out)
            cv2.imwrite(str(out_dir / f"f{args.start + i:04d}.png"),
                        np.hstack([tracked, comp]))
            rows.append((args.start + i, out["lane_state"],
                         out["left_source"], out["right_source"],
                         out["left_confidence_final"], out["right_confidence_final"],
                         out["left_raw_missing_streak"],
                         out["left_held_streak"], out["left_output_missing_streak"]))
    cap.release()
    print("frame  lane_state                       L_src    R_src    L_conf R_conf "
          "L_raw L_held L_out")
    for r in rows:
        (fid, ls, lsrc, rsrc, lconf, rconf, lraw, lheld, lout) = r
        print(f"{fid:5d}  {ls:<32s} {lsrc:<8s} {rsrc:<8s} "
              f"{lconf:.2f}   {rconf:.2f}   {lraw:>3d}   "
              f"{lheld:>3d}    {lout:>3d}")
    print(f"wrote {len(rows)} montages to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
