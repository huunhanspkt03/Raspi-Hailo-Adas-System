#!/usr/bin/env python3
"""
Split a YOLO26 ONNX model into HEF-compatible (neural) and CPU postprocessing parts.

Based on the Hailo reference architecture: the neural portion runs on the Hailo
accelerator, while post-processing (NMS, box decode) runs on the CPU via ONNX
Runtime. This script extracts the postprocessing subgraph from the full ONNX.

The split happens at the Conv cutoff nodes listed in
``config/hailo/yolo26_raw_end_nodes.txt``.

Usage:
    python scripts/extract_yolo26_postprocessing.py \
        --input src/models/yolo26_adas_2404.onnx \
        --output src/models/yolo26_adas_2404_postprocess.onnx

    # If Hailo compilation absorbs sigmoid activations:
    python scripts/extract_yolo26_postprocessing.py \
        --input src/models/yolo26_adas_2404.onnx \
        --output src/models/yolo26_adas_2404_postprocess.onnx \
        --skip-sigmoid
"""
from __future__ import annotations

import argparse
from pathlib import Path

END_NODES_FILE = Path("config/hailo/yolo26_raw_end_nodes.txt")


def _load_end_nodes() -> list[str]:
    if not END_NODES_FILE.exists():
        raise FileNotFoundError(f"End-nodes file not found: {END_NODES_FILE}")
    nodes = [ln.strip() for ln in END_NODES_FILE.read_text("utf-8").splitlines() if ln.strip()]
    if not nodes:
        raise ValueError(f"No end nodes defined in {END_NODES_FILE}")
    return nodes


def _find_conv_output_names(model, end_node_names: list[str]) -> list[str]:
    output_names = []
    for node in model.graph.node:
        for end_name in end_node_names:
            if node.name == end_name or any(end_name in o for o in node.output):
                output_names.extend(node.output)
    if not output_names:
        available = [n.name for n in model.graph.node if n.op_type == "Conv"][-12:]
        raise ValueError(
            f"Could not find cutoff nodes in ONNX graph.\n"
            f"Requested: {end_node_names}\n"
            f"Last Conv nodes: {available}"
        )
    return output_names


def _remove_sigmoid_before_outputs(model, cutoff_outputs: list[str]):
    """Remove Sigmoid nodes feeding into cutoff outputs when --skip-sigmoid."""

    nodes_to_remove = []
    for node in model.graph.node:
        if node.op_type != "Sigmoid":
            continue
        if node.output[0] not in cutoff_outputs:
            continue
        # Rewire: point consumers of sigmoid output to sigmoid input
        sigmoid_in = node.input[0]
        sigmoid_out = node.output[0]
        for other in model.graph.node:
            for i, inp in enumerate(other.input):
                if inp == sigmoid_out:
                    other.input[i] = sigmoid_in
        for out in model.graph.output:
            if out.name == sigmoid_out:
                out.name = sigmoid_in
        nodes_to_remove.append(node)

    for node in nodes_to_remove:
        model.graph.node.remove(node)
    return len(nodes_to_remove)


def extract_postprocessing(
    input_path: str,
    output_path: str,
    skip_sigmoid: bool = False,
) -> None:
    try:
        import onnx
        from onnx.utils import extract_model
    except ImportError as exc:
        raise ImportError("onnx package required: pip install onnx") from exc

    input_p = Path(input_path)
    output_p = Path(output_path)
    if not input_p.exists():
        raise FileNotFoundError(f"Input ONNX not found: {input_p}")
    output_p.parent.mkdir(parents=True, exist_ok=True)

    end_nodes = _load_end_nodes()
    print(f"Cutoff nodes ({len(end_nodes)}):")
    for n in end_nodes:
        print(f"  {n}")

    model = onnx.load(str(input_p))
    cutoff_outputs = _find_conv_output_names(model, end_nodes)
    print(f"\nCutoff output tensors: {cutoff_outputs}")

    if skip_sigmoid:
        removed = _remove_sigmoid_before_outputs(model, cutoff_outputs)
        print(f"Removed {removed} Sigmoid nodes (--skip-sigmoid)")

    # Extract postprocessing subgraph: from cutoff outputs to final model outputs
    final_output_names = [o.name for o in model.graph.output]
    print(f"Final outputs: {final_output_names}")

    # Save modified full model to temp, then extract subgraph
    tmp_path = str(output_p.with_suffix(".tmp.onnx"))
    onnx.save(model, tmp_path)

    try:
        extract_model(tmp_path, str(output_p), cutoff_outputs, final_output_names)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    print(f"\nExtracted postprocessing ONNX: {output_p}")
    print(f"  Inputs : {cutoff_outputs}")
    print(f"  Outputs: {final_output_names}")
    if skip_sigmoid:
        print("  Note: sigmoid was removed — Hailo HEF absorbs it during quantization")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract YOLO26 postprocessing from ONNX")
    p.add_argument("--input", required=True, help="Full YOLO26 ONNX model")
    p.add_argument("--output", required=True, help="Output postprocessing ONNX")
    p.add_argument("--skip-sigmoid", action="store_true",
                   help="Remove Sigmoid at cutoff (Hailo absorbs it)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_postprocessing(args.input, args.output, skip_sigmoid=args.skip_sigmoid)
