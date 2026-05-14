#!/usr/bin/env bash
# Dispatcher for Hailo compilation scripts.
#
# Usage:
#   bash scripts/compile_hailo.sh yolo
#   bash scripts/compile_hailo.sh yolo-seg
#   bash scripts/compile_hailo.sh all
#
# Example (Docker):
#   docker run -it --rm -v "$(pwd)":/workspace hailo/hailo8_ai_sw_suite_2025-10
#   cd /workspace
#   bash scripts/compile_hailo.sh yolo

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-yolo}"

case "${TARGET}" in
    yolo)
        bash "${ROOT_DIR}/scripts/compile_hailo_yolo.sh"
        ;;
    yolo-seg|yoloseg|seg-yolo|lane)
        bash "${ROOT_DIR}/scripts/compile_hailo_yolo_seg.sh"
        ;;
    all)
        bash "${ROOT_DIR}/scripts/compile_hailo_yolo.sh"
        echo ""
        bash "${ROOT_DIR}/scripts/compile_hailo_yolo_seg.sh"
        ;;
    *)
        echo "Usage: bash scripts/compile_hailo.sh [yolo|yolo-seg|all]"
        exit 1
        ;;
esac
