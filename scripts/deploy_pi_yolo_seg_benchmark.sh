#!/usr/bin/env bash
# Deprecated shim kept only to point callers to the maintained runtime deploy path.

set -euo pipefail

PI_HOST="${1:-pi@<pi-ip>}"
REMOTE_DIR="${2:-~/adas-v2}"

echo "scripts/deploy_pi_yolo_seg_benchmark.sh is deprecated."
echo "The maintained benchmark path is the unified runtime pipeline"
echo "(one camera frame -> DeepLab 512 + YOLO26 640 -> same output frame)."
echo ""
echo "Use instead:"
echo "  bash scripts/deploy_pi_runtime.sh ${PI_HOST} '${REMOTE_DIR}'"
echo ""
echo "Then run on the Pi:"
echo "  cd ${REMOTE_DIR}"
echo "  python src/main.py --config config/full_test.yaml --bench --video <video.mp4>"
exit 1
