"""Run the purple-boundary fit on one real frame and write a PNG.

By default, includes the full debug overlay: accepted components (cyan/orange
boxes), rejected components (red boxes with reason tag), lane_area edges
(pink/sky polylines), and the fitted purple boundaries. ``--no-debug``
draws purple lines only.

Usage::
    python tools/preview_purple_boundaries.py --video test1_720p30.mp4 --frame 90
    python tools/preview_purple_boundaries.py --video test1_720p30.mp4 --frame 400 --no-debug
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.lane_boundaries import fit_lane_boundaries  # noqa: E402
from features.lane_boundaries_render import (  # noqa: E402
    render_boundary_polylines, render_full_debug,
)
from features.lane_postprocess import decode_lane_components  # noqa: E402
from pipeline.inference import create_inference  # noqa: E402
from pipeline.preprocessor import preprocess_segmentation  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=60)
    ap.add_argument("--model", default="src/models/yolo26s_seg.onnx")
    ap.add_argument("--out", default="outputs/purple_boundaries.png")
    ap.add_argument("--no-debug", dest="debug", action="store_false")
    ap.set_defaults(debug=True)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"cannot open {args.video}", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, args.frame))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"cannot read frame {args.frame}", file=sys.stderr)
        return 1
    h, w = frame.shape[:2]
    with create_inference({"lane_segmentation": args.model}, force_onnx=True) as infer:
        prep = preprocess_segmentation(frame, 640)
        raw = infer.infer("lane_segmentation", prep)
    _, _, semantics, lane_area = decode_lane_components(raw, h, w)
    fit = fit_lane_boundaries(semantics, lane_area, h, w)
    out = frame.copy()
    if args.debug:
        render_full_debug(out, fit)
    else:
        render_boundary_polylines(out, fit)
    n_acc = sum(1 for c in fit["components"] if c.rejected_reason is None)
    n_rej = sum(1 for c in fit["components"] if c.rejected_reason is not None)
    geom = fit["geometry_reject_reason"] or "-"
    txt1 = (f"L:{fit['status']['left']} c={fit['left_confidence']:.2f} n={fit['n_left']}  "
            f"R:{fit['status']['right']} c={fit['right_confidence']:.2f} n={fit['n_right']}  "
            f"split_x={fit['side_split_x_used']}  geom={geom}")
    txt2 = f"components: accepted={n_acc} rejected={n_rej}"
    cv2.rectangle(out, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(out, txt1, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, txt2, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (200, 200, 200), 1, cv2.LINE_AA)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, out)
    print(f"wrote {args.out}  fit={fit['status']}  geom={geom}  acc={n_acc} rej={n_rej}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
