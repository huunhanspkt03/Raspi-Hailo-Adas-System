"""Side-by-side: fixed vs. dynamic corridor on one synthetic + one real frame.

Writes ``outputs/corridor_compare_synthetic.png`` and (if a video is given via
``--video``) ``outputs/corridor_compare_<basename>.png``. Run from repo root::

    python tools/compare_corridor.py --video test1_720p30.mp4 --frame 90
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.dynamic_corridor_tracker import DynamicCorridorTracker  # noqa: E402
from features.lane_corridor import fixed_guidance_template  # noqa: E402
from features.lane_postprocess import decode_lane_components  # noqa: E402
from pipeline.inference import create_inference  # noqa: E402
from pipeline.postprocessor import draw_fixed_guidance, draw_lane_debug  # noqa: E402
from pipeline.preprocessor import preprocess_segmentation  # noqa: E402


def _stack(left: np.ndarray, right: np.ndarray, label_l: str, label_r: str) -> np.ndarray:
    out = np.hstack([left, right])
    h = out.shape[0]
    cv2.rectangle(out, (0, 0), (left.shape[1], 32), (0, 0, 0), -1)
    cv2.rectangle(out, (left.shape[1], 0), (out.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(out, label_l, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, label_r, (left.shape[1] + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(out, (left.shape[1], 0), (left.shape[1], h), (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _synthetic_curved_lane(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    frame = np.full((h, w, 3), 60, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    ys = np.arange(int(h * 0.5), h, dtype=np.float32)
    cxs = (1.5e-3 * (ys - h * 0.5) ** 2 + (w // 2 - 100)).astype(np.int32)
    for y, cx in zip(ys.astype(np.int32), cxs):
        x0 = max(0, int(cx) - 90)
        x1 = min(w, int(cx) + 90)
        mask[y, x0:x1] = 1
    return frame, mask


def _draw_pair(frame: np.ndarray, mask: np.ndarray, sem: dict) -> tuple[np.ndarray, np.ndarray]:
    h, w = frame.shape[:2]
    fixed = fixed_guidance_template(h, w)
    fixed_rendered = draw_lane_debug(frame.copy(), mask, sem, None, None, None,
                                     fast=True, guidance=fixed)
    tracker = DynamicCorridorTracker()
    dyn = tracker.update(mask, h, w)
    if dyn is None:
        dyn_rendered = draw_fixed_guidance(frame.copy())
    else:
        dyn_rendered = draw_lane_debug(frame.copy(), mask, sem, None, None, None,
                                       fast=True, guidance=dyn)
    return fixed_rendered, dyn_rendered


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="Optional MP4 to sample one frame from")
    ap.add_argument("--frame", type=int, default=60, help="Frame index inside --video")
    ap.add_argument("--model", default="src/models/yolo26s_seg.onnx",
                    help="ONNX seg model used to produce the lane_area for the real frame")
    ap.add_argument("--out", default="outputs", help="Directory for the comparison PNGs")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic frame: deterministic ground truth.
    syn_frame, syn_mask = _synthetic_curved_lane(480, 854)
    syn_sem = {k: np.zeros_like(syn_mask) for k in
               ("dotted_line", "lane_area", "solid_line")}
    syn_sem["lane_area"] = syn_mask
    fixed_syn, dyn_syn = _draw_pair(syn_frame, syn_mask, syn_sem)
    syn_out = out_dir / "corridor_compare_synthetic.png"
    cv2.imwrite(str(syn_out), _stack(fixed_syn, dyn_syn, "FIXED (camera-anchored)", "DYNAMIC (lane mask)"))
    print(f"Wrote {syn_out}")

    if not args.video:
        return 0

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Cannot open {args.video}", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, args.frame))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"Cannot read frame {args.frame} from {args.video}", file=sys.stderr)
        return 1
    h, w = frame.shape[:2]
    inputs = {"lane_segmentation": args.model}
    infer_ctx = create_inference(inputs, force_onnx=True)
    with infer_ctx as infer:
        prep = preprocess_segmentation(frame, 640)
        raw = infer.infer("lane_segmentation", prep)
    _, _, semantics, lane_area = decode_lane_components(raw, h, w)
    fixed_real, dyn_real = _draw_pair(frame, lane_area, semantics)
    name = Path(args.video).stem
    real_out = out_dir / f"corridor_compare_{name}.png"
    cv2.imwrite(str(real_out), _stack(fixed_real, dyn_real,
                                       f"FIXED  ({name} f{args.frame})",
                                       f"DYNAMIC ({name} f{args.frame})"))
    print(f"Wrote {real_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
