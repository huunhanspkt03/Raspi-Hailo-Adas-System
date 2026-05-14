"""Build YOLO26-seg calibration .npy files (640x640 RGB uint8) from a folder of images.

Default source: ./calib_images/deeplab
Default output: ./calib_data/yolo26n_seg

Each output file is one preprocessed frame matching the YOLO26 input layout
(letterboxed RGB uint8) — the format expected by `hailo optimize --calib-set-path`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.preprocessor import YOLO_INPUT_SIZE, preprocess_yolo26  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=ROOT / "calib_images" / "deeplab")
    p.add_argument("--dst", type=Path, default=ROOT / "calib_data" / "yolo26n_seg")
    p.add_argument("--limit", type=int, default=0, help="Max images (0 = all)")
    p.add_argument("--imgsz", type=int, default=YOLO_INPUT_SIZE)
    p.add_argument("--clean", action="store_true", help="Delete existing .npy in dst first")
    return p.parse_args()


def collect_images(src: Path) -> list[Path]:
    if not src.is_dir():
        raise SystemExit(f"Source directory not found: {src}")
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise SystemExit(f"No images found in {src}")
    return files


def main() -> None:
    args = parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)

    if args.clean:
        for old in args.dst.glob("*.npy"):
            old.unlink()

    images = collect_images(args.src)
    if args.limit > 0:
        images = images[: args.limit]

    print(f"Source : {args.src}")
    print(f"Dest   : {args.dst}")
    print(f"Count  : {len(images)} images -> {args.imgsz}x{args.imgsz} RGB uint8")

    written = 0
    for idx, img_path in enumerate(images):
        frame = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"  skip (unreadable): {img_path.name}")
            continue
        sample = preprocess_yolo26(frame, input_size=args.imgsz)
        if sample.shape != (args.imgsz, args.imgsz, 3) or sample.dtype != np.uint8:
            raise RuntimeError(f"Bad sample shape/dtype: {sample.shape} {sample.dtype}")
        out = args.dst / f"frame_{idx:05d}.npy"
        np.save(out, sample)
        written += 1
        if written % 200 == 0:
            print(f"  {written}/{len(images)}")

    print(f"Done. Wrote {written} .npy files to {args.dst}")


if __name__ == "__main__":
    main()
