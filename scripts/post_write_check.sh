#!/usr/bin/env bash
set -euo pipefail

file="${CLAUDE_TOOL_FILE:-}"
if [[ -z "${file}" ]]; then
    exit 0
fi
normalized="${file//\\//}"

warn() {
    echo "[post-write-check] $1" >&2
}

run_quiet() {
    if ! "$@" >/dev/null 2>&1; then
        warn "Command failed: $*"
        return 1
    fi
    return 0
}

case "${normalized}" in
    *.py)
        run_quiet ruff check --fix "${file}" || true
        ;;
    *.json)
        if [[ "${normalized}" == *".claude/"* ]]; then
            run_quiet python -c "import json,sys; json.load(open(sys.argv[1], encoding='utf-8'))" "${file}" || true
        fi
        ;;
esac

case "${normalized}" in
    *src/features/lane_detection.py|*src/features/lane_center_path.py|*src/features/controller.py|*src/pipeline/preprocessor.py|*src/pipeline_loop.py)
        warn "Lane Python path changed. Settle behavior here first, then port/audit parity with cpp_runtime/src/deeplab_postprocess.cpp and related C++ preprocess/controller code."
        run_quiet python -m unittest tests.test_lane_bounds tests.test_lane_corridor || true
        ;;
    *src/features/object_detection.py|*src/features/detection_filters.py|*src/features/yolo26_raw_head.py|*src/features/behavior_logic.py)
        warn "Detection Python path changed. Settle behavior here first, then port/audit parity with cpp_runtime/src/yolo26s_postprocess.cpp."
        ;;
    *cpp_runtime/*.cpp|*cpp_runtime/*.hpp|*cpp_runtime/CMakeLists.txt)
        run_quiet cmake -S cpp_runtime -B cpp_runtime/build -DUSE_HAILO=OFF || true
        ;;
    *scripts/deploy_pi_runtime.sh|*cpp_runtime/README.md)
        warn "Deploy flow changed. Re-check Pi runtime payload, model paths, and run commands."
        ;;
esac

exit 0
