#!/usr/bin/env bash
# Prepare Raspberry Pi after a fresh OS install for this ADAS project.
#
# Usage:
#   bash scripts/pi_prepare_runtime.sh
#   bash scripts/pi_prepare_runtime.sh --skip-upgrade
#   bash scripts/pi_prepare_runtime.sh --build-cpp
#
# What it does:
#   1) apt update + full-upgrade (unless --skip-upgrade)
#   2) project install flow via scripts/install.sh (unless --skip-install)
#   3) verify Hailo + camera + required HEF files (unless --skip-checks)
#   4) optional C++ runtime build (with --build-cpp)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKIP_UPGRADE=false
SKIP_INSTALL=false
SKIP_CHECKS=false
BUILD_CPP=false

usage() {
    cat <<'EOF'
Usage: bash scripts/pi_prepare_runtime.sh [options]

Options:
  --skip-upgrade   Skip apt update/full-upgrade
  --skip-install   Skip running scripts/install.sh
  --skip-checks    Skip post-install verification checks
  --build-cpp      Build C++ runtime (cpp_runtime) after setup
  --help           Show this help
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-upgrade)
            SKIP_UPGRADE=true
            ;;
        --skip-install)
            SKIP_INSTALL=true
            ;;
        --skip-checks)
            SKIP_CHECKS=true
            ;;
        --build-cpp)
            BUILD_CPP=true
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
    shift
done

cd "${ROOT_DIR}"

resolve_yolo_hef() {
    local candidate
    for candidate in \
        "src/models/yolo26_adas_2404.hef" \
        "src/models/yolo26_adas_1404.hef"
    do
        if [ -f "${candidate}" ]; then
            printf '%s\n' "${candidate}"
            return
        fi
    done
    printf '%s\n' "src/models/yolo26_adas_2404.hef"
}

YOLO_HEF="$(resolve_yolo_hef)"

echo "=== ADAS Pi runtime preparation ==="
echo "Repo: ${ROOT_DIR}"
echo ""

if [ "${SKIP_UPGRADE}" = false ]; then
    echo "[1/4] Updating Raspberry Pi OS packages..."
    sudo apt update
    sudo apt full-upgrade -y
else
    echo "[1/4] Skipped OS package upgrade (--skip-upgrade)."
fi

if [ "${SKIP_INSTALL}" = false ]; then
    echo "[2/4] Running project install script..."
    bash scripts/install.sh
else
    echo "[2/4] Skipped project install (--skip-install)."
fi

if [ "${SKIP_CHECKS}" = false ]; then
    echo "[3/4] Verifying runtime prerequisites..."

    if [ -e /dev/hailo0 ]; then
        echo "  ✓ /dev/hailo0 detected"
    else
        echo "  ⚠ /dev/hailo0 not found"
    fi

    if command -v hailortcli >/dev/null 2>&1; then
        echo "  - hailortcli fw-control identify"
        hailortcli fw-control identify || true
    else
        echo "  ⚠ hailortcli not found in PATH"
    fi

    if command -v rpicam-hello >/dev/null 2>&1; then
        echo "  - rpicam-hello --list-cameras"
        rpicam-hello --list-cameras || true
    else
        echo "  ⚠ rpicam-hello not found"
    fi

    REQUIRED_HEFS=(
        "src/models/deeplab_v3_mobilenet_v2_2304.hef"
        "${YOLO_HEF}"
    )
    for hef in "${REQUIRED_HEFS[@]}"; do
        if [ -f "${hef}" ]; then
            echo "  ✓ ${hef}"
        else
            echo "  ✗ Missing required model: ${hef}" >&2
            exit 1
        fi
    done
else
    echo "[3/4] Skipped verification checks (--skip-checks)."
fi

if [ "${BUILD_CPP}" = true ]; then
    echo "[4/4] Building optional C++ runtime..."
    sudo apt install -y libopencv-dev hailort-dev
    cmake -S cpp_runtime -B cpp_runtime/build -DUSE_HAILO=ON
    cmake --build cpp_runtime/build -j"$(nproc)"
else
    echo "[4/4] C++ runtime build skipped (pass --build-cpp to enable)."
fi

echo ""
if [ -f /var/run/reboot-required ]; then
    echo "⚠ Reboot is required before running the pipeline:"
    echo "  sudo reboot"
    echo ""
fi

echo "Next step (smoke tests):"
echo "  bash scripts/pi_smoke_test.sh --mode lane"
echo "  bash scripts/pi_smoke_test.sh --mode full"
echo "  bash scripts/pi_smoke_test.sh --mode full --camera usb"
echo ""
echo "Done."
