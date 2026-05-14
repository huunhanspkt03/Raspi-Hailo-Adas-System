"""Print per-component diagnostics for one frame to tune rejection thresholds."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.lane_boundaries import fit_lane_boundaries  # noqa: E402
from features.lane_boundary_components import (  # noqa: E402
    extract_lane_area_edges,
)
from features.lane_postprocess import decode_lane_components  # noqa: E402
from pipeline.inference import create_inference  # noqa: E402
from pipeline.preprocessor import preprocess_segmentation  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=90)
    ap.add_argument("--model", default="src/models/yolo26s_seg.onnx")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, args.frame))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return 1
    h, w = frame.shape[:2]
    with create_inference({"lane_segmentation": args.model}, force_onnx=True) as infer:
        prep = preprocess_segmentation(frame, 640)
        raw = infer.infer("lane_segmentation", prep)
    _, _, semantics, lane_area = decode_lane_components(raw, h, w)

    print(f"frame {args.frame}: {w}x{h}")
    print(f"lane_area pixels: {int(np.sum(lane_area > 0))}")
    for k in ("dotted_line", "solid_line"):
        print(f"  {k}: {int(np.sum(semantics[k] > 0))} px")
    top_cut = int(round(h * 0.55))
    le, ri, _ = extract_lane_area_edges(lane_area, top_cut)
    if le.size:
        valid = ~np.isnan(le)
        if np.any(valid):
            print(f"lane_area edges (rows {top_cut}..{h-1}):")
            print(f"  bottom row: left={float(le[-1]) if not np.isnan(le[-1]) else 'NaN'} "
                  f"right={float(ri[-1]) if not np.isnan(ri[-1]) else 'NaN'}")
            valid_left = le[valid]
            print(f"  left edge: min={float(valid_left.min()):.0f} max={float(valid_left.max()):.0f} "
                  f"mean={float(valid_left.mean()):.0f}")
            valid_right = ri[valid]
            print(f"  right edge: min={float(valid_right.min()):.0f} max={float(valid_right.max()):.0f} "
                  f"mean={float(valid_right.mean()):.0f}")

    fit = fit_lane_boundaries(semantics, lane_area, h, w)
    print(f"\nresult: {fit['status']}  geom={fit['geometry_reject_reason']}")
    print(f"max_edge_dist threshold = 0.06 * {w} = {0.06*w:.0f}px")
    print(f"\ncomponents ({len(fit['components'])}):")
    for c in fit["components"]:
        x, y, ww, hh = c.bbox
        print(f"  label={c.label:2d} bbox=({x:4d},{y:4d},{ww:3d},{hh:3d}) "
              f"area={c.n_pixels:5d} aspect={c.aspect_ratio:.2f} "
              f"angle={c.angle_from_vertical_deg:.0f}° "
              f"side={c.side:7s} dist={c.edge_distance_px:.1f} "
              f"reject={c.rejected_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
