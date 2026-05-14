#!/usr/bin/env python3
"""
Export custom YOLO26 detection model to ONNX for Hailo compilation.
Run on the TRAINING MACHINE, not on Raspberry Pi.

Project-aligned intent:
- export custom-trained YOLO26 .pt -> .onnx
- keep static input shape for downstream Hailo workflow
- keep NMS-free / end-to-end behavior unless you explicitly change it
- use imgsz=640 by default for this project

Example:
    python scripts/export_yolo26.py --weights runs/train/best.pt
    python scripts/export_yolo26.py --weights checkpoints/Object_Detection/yolo2404.pt
    python scripts/export_yolo26.py --weights checkpoints/Object_Detection/yolo2404.pt --output src/models/yolo26_adas_2404.onnx
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

RAW_END_NODES_FILE = Path("config/hailo/yolo26_raw_end_nodes.txt")
DEFAULT_EXPORT_OUTPUTS = {
    "yolo2404.pt": Path("src/models/yolo26_adas_2404.onnx"),
    "yolo1404.pt": Path("src/models/yolo26_adas_1404.onnx"),
}


def _load_raw_end_nodes() -> list[str]:
    if not RAW_END_NODES_FILE.exists():
        return []
    return [line.strip() for line in RAW_END_NODES_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def _validate_raw_head_nodes(onnx_path: Path) -> None:
    raw_end_nodes = _load_raw_end_nodes()
    if not raw_end_nodes:
        print("Warning: raw-head node list not found; skipping Hailo raw-head validation.")
        return

    try:
        import onnx
    except ImportError:
        print("Warning: onnx package not installed; skipping Hailo raw-head validation.")
        return

    model = onnx.load(str(onnx_path))
    available_outputs = {name for node in model.graph.node for name in node.output}
    missing = [name for name in raw_end_nodes if f"{name}_output_0" not in available_outputs]

    if missing:
        print("\nWarning: exported ONNX is missing expected raw-head outputs for Hailo parsing:")
        for name in missing:
            print(f"  - {name}")
        print("The current compile_hailo.sh raw-head flow may not match this ONNX graph.")
        return

    print("\nHailo raw-head validation:")
    print(f"  raw end nodes: {len(raw_end_nodes)} found")
    print(f"  config file   : {RAW_END_NODES_FILE}")


def _default_output_path(weights_path: Path) -> Path:
    return DEFAULT_EXPORT_OUTPUTS.get(weights_path.name, weights_path.with_suffix(".onnx"))


def export_yolo26(
    weights: str,
    output: str | None = None,
    imgsz: int = 640,
    opset: int = 11,
    simplify: bool = True,
    dynamic: bool = False,
    half: bool = False,
    end2end: bool | None = None,
) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("Ultralytics is not installed. Run: pip install ultralytics") from e

    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    output_path = Path(output) if output else _default_output_path(weights_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLO26 checkpoint: {weights_path}")
    model = YOLO(str(weights_path))

    # Helpful visibility for custom dataset export
    try:
        names = getattr(model.model, "names", None) or getattr(model, "names", None)
        if names:
            print("Detected class names:")
            if isinstance(names, dict):
                for k, v in names.items():
                    print(f"  {k}: {v}")
            else:
                for i, name in enumerate(names):
                    print(f"  {i}: {name}")
    except Exception:
        pass

    print("\nExport settings:")
    print(f"  output   : {output_path}")
    print(f"  imgsz    : {imgsz}")
    print(f"  opset    : {opset}")
    print(f"  simplify : {simplify}")
    print(f"  dynamic  : {dynamic}")
    print(f"  half     : {half}")
    print(f"  end2end  : {end2end}")

    export_kwargs = dict(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=dynamic,
        half=half,
    )

    # Ultralytics supports end2end for YOLO26 models that use NMS-free inference.
    # Keep it only when user explicitly wants to override the model/export behavior.
    if end2end is not None:
        export_kwargs["end2end"] = end2end

    exported = model.export(**export_kwargs)

    # Ultralytics may save next to weights or return a generated path depending on version.
    exported_path = None

    if isinstance(exported, (str, Path)):
        exported_path = Path(exported)
    else:
        candidate = weights_path.with_suffix(".onnx")
        if candidate.exists():
            exported_path = candidate

    if exported_path is None or not exported_path.exists():
        raise FileNotFoundError(
            "Export command completed but ONNX file could not be located. "
            "Check Ultralytics version and export logs."
        )

    if exported_path.resolve() != output_path.resolve():
        shutil.move(str(exported_path), str(output_path))

    _validate_raw_head_nodes(output_path)

    print(f"\n✓ Exported ONNX: {output_path}")
    print("\nVerify ONNX with:")
    print(
        f'python -c "import onnx; '
        f'm=onnx.load(r\'{output_path}\'); '
        f'onnx.checker.check_model(m); '
        f'print(\'ONNX OK\')"'
    )
    print("\nNext steps:")
    print("python scripts/build_calib_data.py --model yolo26")
    print("bash scripts/compile_hailo_yolo.sh")
    print("\nFor HEF+ONNX split pipeline (Hailo reference architecture):")
    print(f"python scripts/extract_yolo26_postprocessing.py --input {output_path} --output {output_path.with_stem(output_path.stem + '_postprocess')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export custom YOLO26 model to ONNX")
    parser.add_argument("--weights", required=True, help="Path to trained YOLO26 .pt checkpoint")
    parser.add_argument(
        "--output",
        default=None,
        help="Output ONNX path (default: known project checkpoints map into src/models/)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Static input size for export",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset version to use for export",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Disable ONNX graph simplification",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic input shapes (normally keep OFF for this project)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export ONNX in FP16 when supported",
    )
    parser.add_argument(
        "--end2end",
        choices=["true", "false"],
        default=None,
        help="Override YOLO26 end-to-end/NMS-free export mode",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    end2end = None
    if args.end2end == "true":
        end2end = True
    elif args.end2end == "false":
        end2end = False

    export_yolo26(
        weights=args.weights,
        output=args.output,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=not args.no_simplify,
        dynamic=args.dynamic,
        half=args.half,
        end2end=end2end,
    )
