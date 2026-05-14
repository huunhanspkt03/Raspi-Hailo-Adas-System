"""Dump one frame from a video to PNG. ``python tools/sample_video_frame.py <mp4> <frame_idx> <out>``."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print("usage: sample_video_frame.py <mp4> <frame_idx> <out_png>", file=sys.stderr)
        return 2
    src, idx, dst = argv[1], int(argv[2]), argv[3]
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"cannot open {src}", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"cannot read frame {idx} from {src}", file=sys.stderr)
        return 1
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(dst, frame)
    print(f"wrote {dst}  shape={frame.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
