#!/usr/bin/env bash
# Compile src/models/yolo26_adas_2404.onnx to src/models/yolo26_adas_2404.hef.
#
# Run from WSL inside a Hailo DFC / AI Software Suite environment:
#   cd /mnt/d/adas-v2
#   bash scripts/compile_yolo26_adas_2404_hef_wsl.sh
#
# Useful overrides:
#   HW_ARCH=hailo8l bash scripts/compile_yolo26_adas_2404_hef_wsl.sh
#   FINETUNE=0 bash scripts/compile_yolo26_adas_2404_hef_wsl.sh
#   BUILD_DIR=/home/$USER/hailo_build/yolo26_adas_2404 bash scripts/compile_yolo26_adas_2404_hef_wsl.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

NET_NAME="${NET_NAME:-yolo26_adas_2404}"
HW_ARCH="${HW_ARCH:-hailo8}"
START_NODE="${START_NODE:-images}"
ONNX_PATH="${ONNX_PATH:-./src/models/${NET_NAME}.onnx}"
CALIB_DIR="${CALIB_DIR:-./calib_data/yolo26}"
OUTPUT_DIR="${OUTPUT_DIR:-./src/models}"

# Keep DFC temporary files on the WSL ext4 filesystem by default. This is usually
# faster and less fragile than building under /mnt/d.
BUILD_DIR="${BUILD_DIR:-${HOME}/hailo_build/${NET_NAME}_${HW_ARCH}_wsl32}"
PARSED_DIR="${BUILD_DIR}/parsed"
OPT_DIR="${BUILD_DIR}/optimized"
LOG_DIR="${BUILD_DIR}/logs"
MODEL_SCRIPT="${BUILD_DIR}/${NET_NAME}_${HW_ARCH}_quality.alls"
PARSED_HAR="${PARSED_DIR}/${NET_NAME}.har"
OPT_HAR="${OPT_DIR}/${NET_NAME}_optimized.har"
EXPECTED_HEF="${OUTPUT_DIR}/${NET_NAME}.hef"

CALIBSET_SIZE="${CALIBSET_SIZE:-1024}"
FINETUNE="${FINETUNE:-1}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-4}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-2}"
FINETUNE_LR="${FINETUNE_LR:-0.000025}"
KEEP_INTERMEDIATES="${KEEP_INTERMEDIATES:-1}"

END_NODES=(
    "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv"
    "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv"
    "/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv"
    "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv"
    "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv"
    "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv"
)

print_config() {
    echo "=== Hailo DFC compile: ${NET_NAME} ==="
    echo "Workspace      : ${ROOT_DIR}"
    echo "DFC target     : ${HW_ARCH}"
    echo "ONNX           : ${ONNX_PATH}"
    echo "HEF output     : ${EXPECTED_HEF}"
    echo "Calibration    : ${CALIB_DIR}"
    echo "Calibset size  : ${CALIBSET_SIZE}"
    echo "Finetune       : ${FINETUNE} (epochs=${FINETUNE_EPOCHS}, batch=${FINETUNE_BATCH_SIZE})"
    echo "Build dir      : ${BUILD_DIR}"
    echo "Model script   : ${MODEL_SCRIPT}"
    echo ""
}

check_environment() {
    if ! command -v hailo >/dev/null 2>&1; then
        echo "Error: 'hailo' CLI not found. Activate the Hailo DFC environment first."
        exit 1
    fi

    if ! command -v python3 >/dev/null 2>&1; then
        echo "Error: python3 not found in this WSL environment."
        exit 1
    fi

    local mem_kb
    mem_kb="$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
    if [ "${mem_kb}" -gt 0 ]; then
        local mem_gb=$((mem_kb / 1024 / 1024))
        echo "WSL visible RAM : ${mem_gb} GB"
        if [ "${mem_gb}" -lt 28 ]; then
            echo "Warning: DFC sees less than 28GB RAM. For best compile quality, set WSL memory near 32GB."
        fi
    fi

    export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"

    mkdir -p "${PARSED_DIR}" "${OPT_DIR}" "${LOG_DIR}" "${OUTPUT_DIR}"
}

validate_inputs() {
    if [ ! -f "${ONNX_PATH}" ]; then
        echo "Error: ONNX not found: ${ONNX_PATH}"
        exit 1
    fi

    if [ ! -d "${CALIB_DIR}" ]; then
        echo "Error: calibration directory not found: ${CALIB_DIR}"
        echo "Expected RGB uint8 .npy samples shaped 640x640x3."
        exit 1
    fi

    python3 - "${ONNX_PATH}" "${CALIB_DIR}" "${START_NODE}" "${END_NODES[@]}" <<'PY'
import glob
import sys
from pathlib import Path

import numpy as np
import onnx

onnx_path = Path(sys.argv[1])
calib_dir = Path(sys.argv[2])
start_node = sys.argv[3]
end_nodes = sys.argv[4:]

model = onnx.load(str(onnx_path))
inputs = list(model.graph.input)
if len(inputs) != 1:
    raise SystemExit(f"Error: expected one ONNX input, found {len(inputs)}")

inp = inputs[0]
shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
if inp.name != start_node:
    raise SystemExit(f"Error: expected input name {start_node!r}, found {inp.name!r}")
if shape != [1, 3, 640, 640]:
    raise SystemExit(f"Error: expected ONNX input [1, 3, 640, 640], found {shape}")

node_names = {node.name for node in model.graph.node}
missing = [name for name in end_nodes if name not in node_names]
if missing:
    raise SystemExit(f"Error: missing parser end nodes: {missing}")

calib_files = sorted(glob.glob(str(calib_dir / "*.npy")))
if not calib_files:
    raise SystemExit(f"Error: no .npy calibration files found in {calib_dir}")

sample = np.load(calib_files[0])
if sample.shape != (640, 640, 3) or sample.dtype != np.uint8:
    raise SystemExit(
        f"Error: bad calibration sample {calib_files[0]}: "
        f"shape={sample.shape}, dtype={sample.dtype}; expected (640, 640, 3) uint8"
    )

print(f"ONNX input      : {inp.name} {shape}")
print(f"Raw end nodes   : {len(end_nodes)} OK")
print(f"Calibration set : {len(calib_files)} .npy files; sample shape {sample.shape} {sample.dtype}")
PY
}

write_model_script() {
    cat > "${MODEL_SCRIPT}" <<ALLS
# Auto-generated by scripts/compile_yolo26_adas_2404_hef_wsl.sh.
# YOLO26 ADAS uses raw detection heads; sigmoid/NMS stays in runtime code.

# Runtime feeds RGB uint8 NHWC calibration samples and Hailo normalizes to 0..1.
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])

# Highest compiler search level available in DFC for better placement/perf.
performance_param(compiler_optimization_level=max)
model_optimization_flavor(optimization_level=2)

# 1024 is a good quality/reliability point for 32GB WSL; raise via CALIBSET_SIZE if needed.
model_optimization_config(calibration, calibset_size=${CALIBSET_SIZE})

# Reduce quantization outlier damage without forcing lossy compression.
pre_quantization_optimization(activation_clipping, layers={*}, mode=percentile, clipping_values=[0.01, 99.99])
pre_quantization_optimization(weights_clipping, layers={*}, mode=percentile, clipping_values=[0.01, 99.99])
ALLS

    if [ "${FINETUNE}" = "1" ]; then
        cat >> "${MODEL_SCRIPT}" <<ALLS

# Enabled for the 32GB-RAM WSL build path. Disable with FINETUNE=0 if DFC runs out of memory.
post_quantization_optimization(finetune, policy=enabled, learning_rate=${FINETUNE_LR}, epochs=${FINETUNE_EPOCHS}, batch_size=${FINETUNE_BATCH_SIZE})
ALLS
    else
        cat >> "${MODEL_SCRIPT}" <<'ALLS'

post_quantization_optimization(finetune, policy=disabled)
ALLS
    fi

    cat >> "${MODEL_SCRIPT}" <<'ALLS'

# Matches Hailo YOLO reference scripts and avoids a problematic legacy allocator path.
allocator_param(width_splitter_defuse=disabled)
ALLS
}

compile_model() {
    echo ""
    echo "[1/4] Parse ONNX -> HAR"
    hailo parser onnx "${ONNX_PATH}" \
        --hw-arch "${HW_ARCH}" \
        --net-name "${NET_NAME}" \
        --start-node-names "${START_NODE}" \
        --end-node-names "${END_NODES[@]}" \
        --har-path "${PARSED_HAR}" \
        2>&1 | tee "${LOG_DIR}/01_parser.log"

    echo ""
    echo "[2/4] Optimize / quantize HAR"
    hailo optimize "${PARSED_HAR}" \
        --hw-arch "${HW_ARCH}" \
        --model-script "${MODEL_SCRIPT}" \
        --calib-set-path "${CALIB_DIR}" \
        --output-har-path "${OPT_HAR}" \
        2>&1 | tee "${LOG_DIR}/02_optimize.log"

    echo ""
    echo "[3/4] Compile optimized HAR -> HEF"
    hailo compiler "${OPT_HAR}" \
        --hw-arch "${HW_ARCH}" \
        --model-script "${MODEL_SCRIPT}" \
        --output-dir "${OUTPUT_DIR}" \
        2>&1 | tee "${LOG_DIR}/03_compiler.log"

    echo ""
    echo "[4/4] Validate HEF"
    if [ ! -f "${EXPECTED_HEF}" ]; then
        echo "Error: expected HEF was not created: ${EXPECTED_HEF}"
        exit 1
    fi

    ls -lh "${EXPECTED_HEF}"

    if [ "${KEEP_INTERMEDIATES}" != "1" ]; then
        rm -f "${PARSED_HAR}" "${OPT_HAR}"
    fi
}

print_config
check_environment
validate_inputs
write_model_script
compile_model

echo ""
echo "Done: ${EXPECTED_HEF}"
echo "Logs: ${LOG_DIR}"
