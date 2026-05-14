#!/usr/bin/env python3
"""Export the trained YOLO26s_seg lane checkpoint to ONNX for the dev fallback.

Run on the TRAINING / dev MACHINE (not on Pi). The output ONNX feeds:
    - the Python ONNX fallback at src/pipeline/onnx_backend.py
    - the Hailo compile flow at scripts/compile_hailo_yolo26s_seg.sh

Default flow:
    python scripts/export_yolo26s_seg.py
    -> reads checkpoints/YOLO26s_seg/yolo26_seg0705.pt
    -> writes src/models/yolo26s_seg.onnx (640x640 static input)

The export is a standard Ultralytics YOLO-seg ONNX export. Two outputs are
produced:
    output0 : (1, 4 + nc + 32, A)  packed predictions
    output1 : (1, 32, Hp, Wp)      mask prototype tensor
which is exactly what src/features/lane_seg_postprocess.py expects.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DEFAULT_WEIGHTS = Path("checkpoints/YOLO26s_seg/yolo26_seg0705.pt")
DEFAULT_OUTPUT = Path("src/models/yolo26s_seg.onnx")


def _print_class_names(model) -> None:
    try:
        names = getattr(model.model, "names", None) or getattr(model, "names", None)
    except Exception:
        names = None
    if not names:
        return
    print("Detected class names:")
    if isinstance(names, dict):
        for k, v in names.items():
            print(f"  {k}: {v}")
    else:
        for i, name in enumerate(names):
            print(f"  {i}: {name}")


def _locate_exported_onnx(returned: object, weights_path: Path) -> Path | None:
    if isinstance(returned, (str, Path)):
        path = Path(returned)
        return path if path.exists() else None
    sibling = weights_path.with_suffix(".onnx")
    return sibling if sibling.exists() else None


def export_yolo26s_seg(
    weights: str | Path,
    output: str | Path | None = None,
    imgsz: int = 640,
    opset: int = 12,
    simplify: bool = True,
    dynamic: bool = False,
    half: bool = False,
) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Ultralytics not installed. Run: pip install ultralytics") from exc

    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    output_path = Path(output) if output else DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLO26s_seg checkpoint: {weights_path}")
    model = YOLO(str(weights_path))
    _print_class_names(model)

    print("\nExport settings:")
    print(f"  output   : {output_path}")
    print(f"  imgsz    : {imgsz}")
    print(f"  opset    : {opset}")
    print(f"  simplify : {simplify}")
    print(f"  dynamic  : {dynamic}")
    print(f"  half     : {half}")

    returned = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=dynamic,
        half=half,
    )

    exported_path = _locate_exported_onnx(returned, weights_path)
    if exported_path is None:
        raise FileNotFoundError(
            "Export command finished but the ONNX file could not be located. "
            "Check the Ultralytics version and the export logs."
        )

    if exported_path.resolve() != output_path.resolve():
        shutil.move(str(exported_path), str(output_path))

    print(f"\n[ok] Exported ONNX: {output_path}")
    print("\nNext steps:")
    print("  - Sanity-check the ONNX path on dev:")
    print("      python src/main.py --config config/lane_only.yaml --mode lane --onnx --video <clip>")
    print("  - Compile to HEF on a Hailo DFC host:")
    print("      bash scripts/compile_hailo_yolo26s_seg.sh")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO26s_seg lane model to ONNX")
    parser.add_argument(
        "--weights", default=str(DEFAULT_WEIGHTS),
        help=f"Path to trained YOLO26s_seg .pt checkpoint (default: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help=f"Output ONNX path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Static input size for export")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument("--no-simplify", action="store_true", help="Disable ONNX graph simplification")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic input shapes")
    parser.add_argument("--half", action="store_true", help="Export ONNX in FP16 when supported")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_yolo26s_seg(
        weights=args.weights,
        output=args.output,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=not args.no_simplify,
        dynamic=args.dynamic,
        half=args.half,
    )
