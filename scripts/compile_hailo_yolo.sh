#!/usr/bin/env bash
# Compile the custom YOLO26 ONNX model to Hailo HEF.
# Must run on x86-64 Linux inside the Hailo AI Software Suite / DFC environment.
#
# Example (Docker):
#   docker run -it --rm -v "$(pwd)":/workspace hailo_ai_sw_suite_2025-04
#   cd /workspace
#   bash scripts/compile_hailo_yolo.sh

set -euo pipefail

HW_ARCH="${HW_ARCH:-hailo8}"
YOLO_VARIANT="${YOLO_VARIANT:-yolo26n}"
YOLO_CALIB_DIR="${YOLO_CALIB_DIR:-./calib_data/yolo26}"
YOLO_NET_NAME="${YOLO_NET_NAME:-}"
YOLO_ONNX="${YOLO_ONNX:-}"
YOLO_MODEL_SCRIPT="${YOLO_MODEL_SCRIPT:-./config/hailo/${YOLO_VARIANT}_${HW_ARCH}.alls}"
YOLO_END_NODES_FILE="${YOLO_END_NODES_FILE:-./config/hailo/yolo26_raw_end_nodes.txt}"
MODELS_DIR="${MODELS_DIR:-./src/models}"
BUILD_DIR="${BUILD_DIR:-./outputs/hailo_build/yolo26}"
KEEP_INTERMEDIATES="${KEEP_INTERMEDIATES:-0}"

DEFAULT_YOLO_ONNX_CANDIDATES=(
    "./src/models/yolo26_adas_2404.onnx"
    "./checkpoints/Object_Detection/yolo2404.onnx"
    "./src/models/yolo26_adas_1404.onnx"
    "./checkpoints/Object_Detection/yolo26_1404.onnx"
    "./checkpoints/Object_Detection/best.onnx"
)

resolve_onnx_path() {
    if [ -n "${YOLO_ONNX}" ] && [ -f "${YOLO_ONNX}" ]; then
        return
    fi

    local requested="${YOLO_ONNX:-}"
    if [ -n "${requested}" ] && [ ! -f "${requested}" ]; then
        echo "Info: requested ONNX was not found: ${requested}"
    fi

    local candidate
    for candidate in "${DEFAULT_YOLO_ONNX_CANDIDATES[@]}"; do
        if [ -f "${candidate}" ]; then
            if [ -n "${requested}" ] && [ "${candidate}" != "${requested}" ]; then
                echo "Info: falling back to ONNX: ${candidate}"
            fi
            YOLO_ONNX="${candidate}"
            return
        fi
    done
}

resolve_net_name() {
    if [ -n "${YOLO_NET_NAME}" ]; then
        return
    fi
    if [ -n "${YOLO_ONNX}" ]; then
        YOLO_NET_NAME="$(basename "${YOLO_ONNX}")"
        YOLO_NET_NAME="${YOLO_NET_NAME%.onnx}"
        return
    fi
    YOLO_NET_NAME="yolo26_adas_2404"
}

resolve_onnx_path
resolve_net_name

echo "=== Hailo DFC Compilation - YOLO26 ==="
echo "Target HW      : ${HW_ARCH}"
echo "Variant        : ${YOLO_VARIANT}"
echo "Net name       : ${YOLO_NET_NAME}"
echo "YOLO ONNX      : ${YOLO_ONNX}"
echo "Model script   : ${YOLO_MODEL_SCRIPT}"
echo "End nodes file : ${YOLO_END_NODES_FILE}"
echo "Models dir     : ${MODELS_DIR}"
echo "Build dir      : ${BUILD_DIR}"
echo "YOLO calib dir : ${YOLO_CALIB_DIR}"
echo ""

if ! command -v hailo >/dev/null 2>&1; then
    echo "Error: 'hailo' CLI not found."
    echo "Run this script inside the Hailo AI Software Suite / DFC environment on x86-64."
    exit 1
fi

mkdir -p "${BUILD_DIR}" "${MODELS_DIR}"

if [ ! -f "${YOLO_END_NODES_FILE}" ]; then
    echo "Error: raw end-nodes file not found: ${YOLO_END_NODES_FILE}"
    exit 1
fi

mapfile -t YOLO_END_NODES < "${YOLO_END_NODES_FILE}"
if [ "${#YOLO_END_NODES[@]}" -eq 0 ]; then
    echo "Error: no raw end nodes defined in ${YOLO_END_NODES_FILE}"
    exit 1
fi

cleanup_artifacts() {
    if [ "${KEEP_INTERMEDIATES}" = "1" ]; then
        return
    fi
    rm -f \
        "${BUILD_DIR}/parsed/${YOLO_NET_NAME}.har" \
        "${BUILD_DIR}/optimized/${YOLO_NET_NAME}_optimized.har" \
        "${MODELS_DIR}/${YOLO_NET_NAME}_compiled.har"
}

compile_yolo26() {
    local parsed_dir="${BUILD_DIR}/parsed"
    local quant_dir="${BUILD_DIR}/optimized"
    local har="${parsed_dir}/${YOLO_NET_NAME}.har"
    local har_opt="${quant_dir}/${YOLO_NET_NAME}_optimized.har"
    local hef="${MODELS_DIR}/${YOLO_NET_NAME}.hef"

    resolve_onnx_path
    resolve_net_name

    if [ ! -f "${YOLO_ONNX}" ]; then
        echo "Error: ONNX not found: ${YOLO_ONNX}"
        echo "Export it first with:"
        echo "  python scripts/export_yolo26.py --weights checkpoints/Object_Detection/yolo2404.pt"
        exit 1
    fi
    if [ ! -f "${YOLO_MODEL_SCRIPT}" ]; then
        echo "Error: Hailo model script not found: ${YOLO_MODEL_SCRIPT}"
        exit 1
    fi

    mkdir -p "${parsed_dir}" "${quant_dir}"

    echo "--- Compiling: ${YOLO_NET_NAME} ---"
    echo "Raw end nodes:"
    printf '  %s\n' "${YOLO_END_NODES[@]}"

    echo "[1/4] Parsing ONNX -> HAR"
    hailo parser onnx "${YOLO_ONNX}" \
        --hw-arch "${HW_ARCH}" \
        --net-name "${YOLO_NET_NAME}" \
        --start-node-names "images" \
        --end-node-names "${YOLO_END_NODES[@]}" \
        --har-path "${har}"

    echo "[2/4] Optimizing / quantizing"
    if [ -d "${YOLO_CALIB_DIR}" ]; then
        hailo optimize "${har}" \
            --hw-arch "${HW_ARCH}" \
            --model-script "${YOLO_MODEL_SCRIPT}" \
            --calib-set-path "${YOLO_CALIB_DIR}" \
            --output-har-path "${har_opt}"
    else
        echo "Warning: calibration directory not found: ${YOLO_CALIB_DIR}"
        echo "Proceeding without explicit calibration set."
        hailo optimize "${har}" \
            --hw-arch "${HW_ARCH}" \
            --model-script "${YOLO_MODEL_SCRIPT}" \
            --output-har-path "${har_opt}"
    fi

    echo "[3/4] Compiling optimized HAR -> HEF"
    hailo compiler "${har_opt}" \
        --hw-arch "${HW_ARCH}" \
        --model-script "${YOLO_MODEL_SCRIPT}" \
        --output-dir "${MODELS_DIR}"

    echo "[4/4] Validating output"
    if [ ! -f "${hef}" ]; then
        echo "Error: expected HEF not found after compile: ${hef}"
        exit 1
    fi

    echo "Done: ${hef}"
}

cleanup_artifacts
compile_yolo26

if [ "${KEEP_INTERMEDIATES}" != "1" ]; then
    rm -f "${BUILD_DIR}/parsed/${YOLO_NET_NAME}.har" "${BUILD_DIR}/optimized/${YOLO_NET_NAME}_optimized.har"
fi

echo ""
echo "Recommended calibration build:"
echo "  python scripts/build_calib_data.py --model yolo26"
echo ""
echo "Artifacts kept in src/models:"
echo "  ${MODELS_DIR}/${YOLO_NET_NAME}.hef"
