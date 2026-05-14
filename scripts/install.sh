#!/usr/bin/env bash
# Install runtime dependencies for the RC Lane-Centering project
# on Raspberry Pi 5 + Raspberry Pi AI HAT+ 26 TOPS (Hailo-8).
#
# Run:
#   bash scripts/install.sh
#
# Notes:
# - This is for Raspberry Pi OS Bookworm/Trixie 64-bit on Raspberry Pi 5
# - AI HAT+ 26 TOPS uses Hailo-8
# - On Pi, run the real pipeline with base config, not the x86 dev config

set -euo pipefail

echo "=== RC Lane-Centering Setup — Raspberry Pi 5 + AI HAT+ 26 TOPS (Hailo-8) ==="
echo ""

echo "[1/6] Updating apt package lists..."
sudo apt update

echo "[2/6] Installing system packages..."
sudo apt install -y \
    python3-pip python3-venv python3-dev \
    dkms \
    libgstreamer1.0-dev gstreamer1.0-tools \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-libcamera \
    libopencv-dev python3-opencv \
    libcamera-dev rpicam-apps \
    i2c-tools

echo "[3/6] Installing Hailo runtime stack from Raspberry Pi apt repo..."
sudo apt install -y hailo-all

echo "[4/6] Creating Python virtual environment..."
python3 -m venv --system-site-packages .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

echo "[5/6] Installing Python packages for the project..."
# Keep OpenCV from apt/system packages to avoid GUI/headless conflicts.
python -m pip install \
    numpy \
    loguru \
    pyyaml \
    ruff

echo "[6/6] Verifying camera and Hailo environment..."

echo "- Checking Python Hailo bindings..."
if python - <<'PY'
try:
    import hailo  # noqa: F401
    print("hailo import OK")
except Exception:
    try:
        import hailo_platform  # noqa: F401
        print("hailo_platform import OK")
    except Exception as e:
        raise SystemExit(f"Hailo Python bindings not available in current Python env: {e}")
PY
then
    echo "✓ Hailo Python bindings detected"
else
    echo "✗ Hailo Python bindings not available"
    echo "  If packages were just installed, reboot first, then re-run this script."
    exit 1
fi

echo "- Checking device node..."
if [ -e /dev/hailo0 ]; then
    echo "✓ Hailo device detected at /dev/hailo0"
else
    echo "⚠ /dev/hailo0 not found"
    echo "  Reboot after installing hailo-all, then check again."
fi

echo "- Running hailortcli identify..."
if command -v hailortcli >/dev/null 2>&1; then
    hailortcli fw-control identify || true
else
    echo "⚠ hailortcli not found in PATH"
fi

echo "- Checking camera enumeration..."
if command -v rpicam-hello >/dev/null 2>&1; then
    rpicam-hello --list-cameras || true
else
    echo "⚠ rpicam-hello not found"
fi

echo ""
echo "=== Setup complete ==="
echo "Activate venv:"
echo "  source .venv/bin/activate"
echo ""
echo "Recommended next checks:"
echo "  python -c \"import cv2; print(cv2.__version__)\""
echo "  python -c \"import hailo; print('hailo import OK')\""
echo ""
echo "Run on Raspberry Pi:"
echo "  python src/main.py --config config/base.yaml --display"
echo ""
echo "Display-only note:"
echo "  --display opens the live window and does not save video by itself."
echo "  Add --save-video outputs/run.mp4 only when you want to record output."
echo ""
echo "Do NOT use --onnx on Pi unless you intentionally want CPU fallback instead of Hailo."
