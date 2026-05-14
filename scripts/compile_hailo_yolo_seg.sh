#!/usr/bin/env bash
# Deprecated: superseded by scripts/compile_hailo_yolo26s_seg.sh.
# This script remains as a thin shim for older notebooks/tooling.
# For new work prefer compile_hailo_yolo26s_seg.sh, which has corrected
# default paths and a yolo26s variant.
# Run this inside a Hailo DFC environment on x86-64 Linux or WSL.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

HW_ARCH="${HW_ARCH:-hailo8}"
YOLO_SEG_VARIANT="${YOLO_SEG_VARIANT:-yolo26n}"
YOLO_SEG_ONNX="${YOLO_SEG_ONNX:-./src/models/yolo26n_seg.onnx}"
YOLO_SEG_CALIB_DIR="${YOLO_SEG_CALIB_DIR:-./calib_data/yolo26n_seg}"
YOLO_SEG_NET_NAME="${YOLO_SEG_NET_NAME:-yolo26n_seg}"
YOLO_SEG_START_NODE="${YOLO_SEG_START_NODE:-images}"
YOLO_SEG_END_NODES="${YOLO_SEG_END_NODES:-/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv /model.23/one2one_cv2.1/one2one_cv2.1.2/Conv /model.23/one2one_cv2.2/one2one_cv2.2.2/Conv /model.23/one2one_cv3.0/one2one_cv3.0.2/Conv /model.23/one2one_cv3.1/one2one_cv3.1.2/Conv /model.23/one2one_cv3.2/one2one_cv3.2.2/Conv /model.23/one2one_cv4.0/one2one_cv4.0.2/Conv /model.23/one2one_cv4.1/one2one_cv4.1.2/Conv /model.23/one2one_cv4.2/one2one_cv4.2.2/Conv /model.23/proto/cv3/act/Mul}"
YOLO_SEG_MODEL_SCRIPT="${YOLO_SEG_MODEL_SCRIPT:-./config/hailo/${YOLO_SEG_VARIANT}_seg_${HW_ARCH}.alls}"
MODELS_DIR="${MODELS_DIR:-./src/models}"
BUILD_DIR="${BUILD_DIR:-./outputs/hailo_build/yolo26_seg}"
KEEP_INTERMEDIATES="${KEEP_INTERMEDIATES:-0}"
ALLOW_RANDOM_CALIB="${ALLOW_RANDOM_CALIB:-0}"

if [ "${YOLO_SEG_MODEL_SCRIPT}" = "none" ] || [ "${YOLO_SEG_MODEL_SCRIPT}" = "NONE" ]; then
    YOLO_SEG_MODEL_SCRIPT=""
fi

read -r -a YOLO_SEG_END_NODE_ARRAY <<< "${YOLO_SEG_END_NODES}"

print_banner() {
    echo "=== Hailo DFC Compilation - YOLO26 Segmentation ==="
    echo "Workspace           : ${ROOT_DIR}"
    echo "Target HW           : ${HW_ARCH}"
    echo "Variant             : ${YOLO_SEG_VARIANT}"
    echo "YOLO-seg ONNX       : ${YOLO_SEG_ONNX}"
    echo "Net name            : ${YOLO_SEG_NET_NAME}"
    echo "Start node          : ${YOLO_SEG_START_NODE}"
    echo "End nodes           : ${YOLO_SEG_END_NODES}"
    echo "Model script        : ${YOLO_SEG_MODEL_SCRIPT:-<none>}"
    echo "Calibration dir     : ${YOLO_SEG_CALIB_DIR}"
    echo "Models dir          : ${MODELS_DIR}"
    echo "Build dir           : ${BUILD_DIR}"
    echo ""
}

ensure_hailo_cli() {
    if ! command -v hailo >/dev/null 2>&1; then
        echo "Error: 'hailo' CLI not found."
        echo "Run this script inside the Hailo Dataflow Compiler environment."
        exit 1
    fi
}

count_calibration_entries() {
    if [ ! -d "${YOLO_SEG_CALIB_DIR}" ]; then
        echo 0
        return
    fi

    find "${YOLO_SEG_CALIB_DIR}" -maxdepth 1 -type f -name "*.npy" | wc -l | tr -d '[:space:]'
}

first_calibration_entry() {
    if [ ! -d "${YOLO_SEG_CALIB_DIR}" ]; then
        return
    fi

    shopt -s nullglob
    local files=( "${YOLO_SEG_CALIB_DIR}"/*.npy )
    shopt -u nullglob

    if [ "${#files[@]}" -gt 0 ]; then
        printf '%s\n' "${files[0]}"
    fi
}

validate_inputs() {
    if [ ! -f "${YOLO_SEG_ONNX}" ]; then
        echo "Error: ONNX not found: ${YOLO_SEG_ONNX}"
        echo "Export it first, for example:"
        echo "  python scripts/export_yolo26s_seg.py"
        exit 1
    fi

    if [ -n "${YOLO_SEG_MODEL_SCRIPT}" ] && [ ! -f "${YOLO_SEG_MODEL_SCRIPT}" ]; then
        echo "Error: model script not found: ${YOLO_SEG_MODEL_SCRIPT}"
        exit 1
    fi

    if [ "${#YOLO_SEG_END_NODE_ARRAY[@]}" -eq 0 ]; then
        echo "Error: no YOLO-seg end nodes configured."
        exit 1
    fi

    python -c '
from pathlib import Path
import sys

import onnx

onnx_path = Path(sys.argv[1])
expected_end_nodes = sys.argv[2:]
model = onnx.load(str(onnx_path))
input_dims = [int(d.dim_value) for d in model.graph.input[0].type.tensor_type.shape.dim]
output_names = [o.name for o in model.graph.output]
node_names = {n.name for n in model.graph.node}
missing = [name for name in expected_end_nodes if name not in node_names]
print(f"ONNX input shape : {tuple(input_dims)}")
print(f"ONNX outputs     : {output_names}")
if missing:
    print(f"Error: requested parser end nodes are missing from ONNX graph nodes: {missing}")
    sys.exit(1)
print(f"All {len(expected_end_nodes)} end nodes located in ONNX graph.")
' "${YOLO_SEG_ONNX}" "${YOLO_SEG_END_NODE_ARRAY[@]}"

    mkdir -p "${BUILD_DIR}" "${MODELS_DIR}"
}

validate_calibration_shape() {
    local sample_path="$1"

    if [ -z "${sample_path}" ]; then
        return
    fi

    python -c '
from pathlib import Path
import sys

import numpy as np
import onnx

onnx_path = Path(sys.argv[1])
sample_path = Path(sys.argv[2])
model = onnx.load(str(onnx_path))
dims = [int(d.dim_value) for d in model.graph.input[0].type.tensor_type.shape.dim]
expected = (dims[2], dims[3], dims[1])
sample = np.load(sample_path)
actual = tuple(int(v) for v in sample.shape)
if actual != expected:
    print(f"Error: calibration sample shape {actual} does not match ONNX input {expected}.")
    print(f"Point YOLO_SEG_CALIB_DIR at a directory of RGB uint8 .npy samples with shape {expected}.")
    sys.exit(1)
print(f"Calibration sample shape matches ONNX input: {actual}")
' "${YOLO_SEG_ONNX}" "${sample_path}"
}

cleanup_artifacts() {
    if [ "${KEEP_INTERMEDIATES}" = "1" ]; then
        return
    fi

    rm -f \
        "${BUILD_DIR}/parsed/${YOLO_SEG_NET_NAME}.har" \
        "${BUILD_DIR}/optimized/${YOLO_SEG_NET_NAME}_optimized.har" \
        "${MODELS_DIR}/${YOLO_SEG_NET_NAME}_compiled.har"
}

optimize_har() {
    local har="$1"
    local har_opt="$2"
    local calib_count
    local sample_path

    calib_count="$(count_calibration_entries)"
    if [ "${calib_count}" -eq 0 ]; then
        if [ "${ALLOW_RANDOM_CALIB}" != "1" ]; then
            echo "Error: no calibration .npy files found in ${YOLO_SEG_CALIB_DIR}"
            echo "Point YOLO_SEG_CALIB_DIR to 640x640 RGB uint8 calibration samples,"
            echo "or rerun with ALLOW_RANDOM_CALIB=1 for a quick smoke-test compile."
            exit 1
        fi

        echo "Warning: calibration directory is missing or empty."
        echo "Falling back to --use-random-calib-set because ALLOW_RANDOM_CALIB=1."
        if [ -n "${YOLO_SEG_MODEL_SCRIPT}" ]; then
            hailo optimize "${har}" \
                --hw-arch "${HW_ARCH}" \
                --model-script "${YOLO_SEG_MODEL_SCRIPT}" \
                --use-random-calib-set \
                --output-har-path "${har_opt}"
        else
            hailo optimize "${har}" \
                --hw-arch "${HW_ARCH}" \
                --use-random-calib-set \
                --output-har-path "${har_opt}"
        fi
        return
    fi

    echo "Calibration entries: ${calib_count}"
    sample_path="$(first_calibration_entry)"
    validate_calibration_shape "${sample_path}"
    if [ -n "${YOLO_SEG_MODEL_SCRIPT}" ]; then
        hailo optimize "${har}" \
            --hw-arch "${HW_ARCH}" \
            --model-script "${YOLO_SEG_MODEL_SCRIPT}" \
            --calib-set-path "${YOLO_SEG_CALIB_DIR}" \
            --output-har-path "${har_opt}"
    else
        hailo optimize "${har}" \
            --hw-arch "${HW_ARCH}" \
            --calib-set-path "${YOLO_SEG_CALIB_DIR}" \
            --output-har-path "${har_opt}"
    fi
}

compile_yolo_seg() {
    local parsed_dir="${BUILD_DIR}/parsed"
    local quant_dir="${BUILD_DIR}/optimized"
    local har="${parsed_dir}/${YOLO_SEG_NET_NAME}.har"
    local har_opt="${quant_dir}/${YOLO_SEG_NET_NAME}_optimized.har"
    local hef="${MODELS_DIR}/${YOLO_SEG_NET_NAME}.hef"

    mkdir -p "${parsed_dir}" "${quant_dir}"

    echo "--- Compiling: ${YOLO_SEG_NET_NAME} ---"
    echo "[1/4] Parsing ONNX -> HAR"
    hailo parser onnx "${YOLO_SEG_ONNX}" \
        --hw-arch "${HW_ARCH}" \
        --net-name "${YOLO_SEG_NET_NAME}" \
        --start-node-names "${YOLO_SEG_START_NODE}" \
        --end-node-names "${YOLO_SEG_END_NODE_ARRAY[@]}" \
        --har-path "${har}"

    echo "[2/4] Optimizing / quantizing"
    optimize_har "${har}" "${har_opt}"

    echo "[3/4] Compiling optimized HAR -> HEF"
    if [ -n "${YOLO_SEG_MODEL_SCRIPT}" ]; then
        hailo compiler "${har_opt}" \
            --hw-arch "${HW_ARCH}" \
            --model-script "${YOLO_SEG_MODEL_SCRIPT}" \
            --output-dir "${MODELS_DIR}"
    else
        hailo compiler "${har_opt}" \
            --hw-arch "${HW_ARCH}" \
            --output-dir "${MODELS_DIR}"
    fi

    echo "[4/4] Validating output"
    if [ ! -f "${hef}" ]; then
        echo "Error: expected HEF not found after compile: ${hef}"
        exit 1
    fi

    echo "Done: ${hef}"
}

print_banner
ensure_hailo_cli
validate_inputs
cleanup_artifacts
compile_yolo_seg

if [ "${KEEP_INTERMEDIATES}" != "1" ]; then
    rm -f "${BUILD_DIR}/parsed/${YOLO_SEG_NET_NAME}.har" "${BUILD_DIR}/optimized/${YOLO_SEG_NET_NAME}_optimized.har"
fi

echo ""
echo "Next steps:"
echo "  1. ${MODELS_DIR}/${YOLO_SEG_NET_NAME}.hef was built as a legacy single-model artifact."
echo "  2. The maintained benchmark path is the unified runtime:"
echo "     python src/main.py --config config/base.yaml --bench --video <video.mp4>"
