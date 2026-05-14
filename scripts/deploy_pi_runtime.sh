#!/usr/bin/env bash
# Copy only the files required to build and run the C++ ADAS runtime on Raspberry Pi.
#
# Usage:
#   bash scripts/deploy_pi_runtime.sh pi@<pi-ip>
#   bash scripts/deploy_pi_runtime.sh pi@<pi-ip> '~/adas_project'
#   bash scripts/deploy_pi_runtime.sh pi@<pi-ip> /home/pi/adas_project

set -euo pipefail

PI_HOST="${1:?Usage: bash scripts/deploy_pi_runtime.sh pi@<pi-ip> [remote_dir]}"
REMOTE_DIR="${2:-~/adas_project}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEEPLAB_HEF="${ROOT_DIR}/src/models/deeplab_v3_mobilenet_v2_2304.hef"
YOLO_HEF=""
CPP_RUNTIME_DIR="${ROOT_DIR}/cpp_runtime"

resolve_yolo_hef() {
  local candidate
  for candidate in \
    "${ROOT_DIR}/src/models/yolo26_adas_2404.hef" \
    "${ROOT_DIR}/src/models/yolo26_adas_1404.hef"
  do
    if [ -f "${candidate}" ]; then
      YOLO_HEF="${candidate}"
      return
    fi
  done
  YOLO_HEF="${ROOT_DIR}/src/models/yolo26_adas_2404.hef"
}

resolve_yolo_hef

case "${REMOTE_DIR}" in
  /home/nhan/*)
    echo "Error: remote_dir expanded to a local WSL home path: ${REMOTE_DIR}" >&2
    echo "Pass the remote home path quoted, for example:" >&2
    echo "  bash scripts/deploy_pi_runtime.sh ${PI_HOST} '~/adas'" >&2
    echo "or use an explicit Pi path such as:" >&2
    echo "  bash scripts/deploy_pi_runtime.sh ${PI_HOST} /home/pi/adas" >&2
    exit 1
    ;;
esac

require_file() {
  local path="$1"
  if [ ! -e "${path}" ]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

require_file "${CPP_RUNTIME_DIR}/CMakeLists.txt"
require_file "${CPP_RUNTIME_DIR}/src/main.cpp"
require_file "${CPP_RUNTIME_DIR}/include/adas/camera_source.hpp"
require_file "${CPP_RUNTIME_DIR}/include/adas/controller.hpp"
require_file "${CPP_RUNTIME_DIR}/include/adas/deeplab_postprocess.hpp"
require_file "${CPP_RUNTIME_DIR}/include/adas/hailo_backend.hpp"
require_file "${CPP_RUNTIME_DIR}/include/adas/inference_backend.hpp"
require_file "${CPP_RUNTIME_DIR}/include/adas/lane_center_path.hpp"
require_file "${CPP_RUNTIME_DIR}/include/adas/object_localizer.hpp"
require_file "${CPP_RUNTIME_DIR}/include/adas/preprocess.hpp"
require_file "${CPP_RUNTIME_DIR}/include/adas/visualization.hpp"
require_file "${CPP_RUNTIME_DIR}/include/adas/yolo26s_postprocess.hpp"
require_file "${CPP_RUNTIME_DIR}/src/camera_source.cpp"
require_file "${CPP_RUNTIME_DIR}/src/controller.cpp"
require_file "${CPP_RUNTIME_DIR}/src/deeplab_postprocess.cpp"
require_file "${CPP_RUNTIME_DIR}/src/lane_center_path.cpp"
require_file "${CPP_RUNTIME_DIR}/src/hailo_backend.cpp"
require_file "${CPP_RUNTIME_DIR}/src/object_localizer.cpp"
require_file "${CPP_RUNTIME_DIR}/src/preprocess.cpp"
require_file "${CPP_RUNTIME_DIR}/src/visualization.cpp"
require_file "${CPP_RUNTIME_DIR}/src/yolo26s_postprocess.cpp"
require_file "${DEEPLAB_HEF}"
require_file "${YOLO_HEF}"

ssh "${PI_HOST}" "mkdir -p \
  ${REMOTE_DIR}/cpp_runtime \
  ${REMOTE_DIR}/src/models \
  ${REMOTE_DIR}/outputs"

scp -r \
  "${CPP_RUNTIME_DIR}/CMakeLists.txt" \
  "${CPP_RUNTIME_DIR}/include" \
  "${CPP_RUNTIME_DIR}/src" \
  "${PI_HOST}:${REMOTE_DIR}/cpp_runtime/"

scp \
  "${DEEPLAB_HEF}" \
  "${YOLO_HEF}" \
  "${PI_HOST}:${REMOTE_DIR}/src/models/"

echo "Deployed Pi runtime bundle to ${PI_HOST}:${REMOTE_DIR}"
echo "Copied only:"
echo "  - cpp_runtime/CMakeLists.txt"
echo "  - cpp_runtime/include/"
echo "  - cpp_runtime/src/"
echo "  - src/models/$(basename "${DEEPLAB_HEF}")"
echo "  - src/models/$(basename "${YOLO_HEF}")"
echo
echo "Build on Pi:"
echo "  cd ${REMOTE_DIR}"
echo "  sudo apt install libopencv-dev hailort-dev"
echo "  cmake -S cpp_runtime -B cpp_runtime/build -DUSE_HAILO=ON"
echo "  cmake --build cpp_runtime/build -j"
echo
echo "Run on Pi:"
echo "  ./cpp_runtime/build/adas_runtime --mode lane --camera --display"
echo "  ./cpp_runtime/build/adas_runtime --mode full --camera --display"
echo "  ./cpp_runtime/build/adas_runtime --mode full --video /path/to/video.mp4 --display"
echo
echo "Notes:"
echo "  --display only shows the live OpenCV window."
echo "  No video file is written unless you add --save-video /path/to/out.mp4"
echo "  or, for lane-only overlays, --save-mask-video /path/to/out.mp4."
echo "  If you SSH into the Pi without X11/desktop access, the display window will not appear."
