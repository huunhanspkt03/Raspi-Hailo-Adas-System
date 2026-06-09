# adas_cpp — parallel C++ runtime

This is a **new, parallel** C++ implementation of the ADAS pipeline for
Raspberry Pi 5 + Hailo-8L. The Python runtime under `../src` remains the
reference/fallback; nothing in `../src` is changed by this work.

## Status (milestone 1)

| Subsystem        | Status        | Source-of-truth (Python)                          |
|------------------|---------------|---------------------------------------------------|
| CMake build      | ✅ done       | —                                                 |
| Capture (CSI + video) | ✅ done   | `src/pipeline/camera.py`, `camera_video.py`       |
| HailoRT C++ wrap | ✅ done       | `src/pipeline/hailo_backend.py`                   |
| Lane preprocess  | ✅ done       | `src/pipeline/preprocessor.py::preprocess_lane_semantic` |
| STDC1 decode     | ✅ done       | `src/features/lane_semantic_postprocess.py`       |
| Lane geometry    | ✅ done       | `src/features/lane_path.py::mask_centerline`      |
| Steering         | ✅ basic      | `src/features/lane_target_resolver.py` (subset)   |
| Overlay          | ✅ basic      | `src/pipeline/output_renderer.py` (subset)        |
| Threaded runtime | ✅ done       | `src/pipeline/threaded_runtime.py`                |
| YOLO26 decode    | ✅ done       | `src/features/object_detection.py`, `yolo26_raw_head.py` |
| Object distance  | ✅ basic      | `src/features/object_localizer*.py` (road-vehicle subset) |
| Object tracker   | ⚠ identity    | `src/features/object_tracker.py`                  |
| Behavior         | ⚠ basic       | `src/features/behavior_logic.py`                  |
| Fusion (full)    | ⚠ partial     | `src/pipeline/fusion_worker.py`                   |
| RC actuator      | ⏳ **stub**   | `src/features/control_runner.py`                  |
| Multiprocess render (Stage A parity) | ❌ not planned | `src/pipeline/mp_render.py` |

The lane path is feature-complete enough to run on Pi against a CSI camera
or a video file, draw the lane area + red centerline + yellow corridor, and
report a steering scalar. YOLO26 raw-head decode is also wired for
`--mode detection` and `--mode full`; tracking, behavior smoothing, and RC
actuation remain intentionally lightweight/TODO ports.

## Layout

```
cpp/
├── CMakeLists.txt
├── README.md                 ← you are here
├── include/adas/             ← public headers (types, capture, hailo,
│   ├── types.hpp                lane_decode, lane_geometry, overlay,
│   ├── latest_queue.hpp         object_detection, behavior (partial),
│   ├── capture.hpp              fusion (partial), runtime, cli, ...)
│   ├── hailo_inference.hpp
│   ├── lane_decode.hpp
│   ├── lane_geometry.hpp
│   ├── overlay.hpp
│   ├── object_detection.hpp
│   ├── behavior.hpp          ← partial behavior policy
│   ├── fusion.hpp            ← partial safety/control port
│   ├── runtime.hpp
│   └── cli.hpp
└── src/                      ← implementations + main + partial ports
```

## Build (Raspberry Pi 5, Bookworm aarch64)

Prerequisites:

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config \
                    libopencv-dev libgstreamer1.0-dev \
                    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
                    gstreamer1.0-libcamera gstreamer1.0-plugins-bad
# HailoRT + Hailo PCIe driver + libcamera integration
sudo apt install -y hailo-all
sudo reboot     # required after hailo-all install
```

Build:

```bash
cd cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
# Binary lives at: cpp/build/adas_cpp
```

On dev machines without `/dev/hailo0`:

```bash
# Skip Hailo entirely — useful to validate that the code compiles and
# capture/overlay paths work; inference returns empty outputs.
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DADAS_USE_HAILO=OFF
cmake --build build -j
```

## Run commands

All commands are run from the repository root (one level above `cpp/`)
so the default HEF / video paths under `src/models/` resolve.

### Video lane-only (recommended first run)

```bash
./cpp/build/adas_cpp \
    --video test4_720p30.mp4 \
    --mode lane \
    --display-width 960
```

### Video full mode

Uses the default YOLO HEF at `src/models/yolo26n_object.hef`, runs object
inference every second captured frame by default, and fuses the freshest
object packet whose `frame_id <= lane.frame_id`.

```bash
./cpp/build/adas_cpp \
    --video test4_720p30.mp4 \
    --mode full \
    --display-width 960
```

#### Full mode — model scheduling

In full mode both lane (STDC1) and YOLO26 run on the same Hailo-8L. The
`--model-schedule` flag picks how the two models share the device:

```
--model-schedule parallel   (default; backwards compatible)
--model-schedule alternate
```

- `parallel` — lane is enqueued for every captured frame; YOLO is
  enqueued every Nth frame (`--object-every-n`). The two models may
  infer concurrently on Hailo, which is the source of the contention
  observed at `--object-every-n 1` (lane/object infer rise to
  ~34–36 ms).
- `alternate` — each captured frame is enqueued to exactly one Hailo
  branch. Lane and YOLO take strict turns (frame 1 → lane, frame 2 →
  YOLO, frame 3 → lane, ...). Only one model runs per frame, so Hailo
  contention is eliminated and the per-inference latency drops back to
  the single-model numbers (~10 ms object, ~21 ms lane).

  In this mode the object worker emits a render packet using the most
  recent cached `LanePacket`, so the display still updates on every
  captured frame even though each individual perception output is at
  half the source rate. `--object-every-n` is ignored. The HUD shows a
  small "alternate" icon (two crossed arrows) instead of the parallel
  icon (two up arrows).

  Limitation: lane and object outputs each refresh at half the capture
  FPS. Front-object selection on the object-driven render packets uses
  the cached corridor (1 frame stale at 30 fps capture, ~33 ms).

Example:

```bash
./cpp/build/adas_cpp \
    --video test4_720p30.mp4 \
    --mode full --model-schedule alternate \
    --display-width 960
```

#### Operator HUD (Python-parity)

The renderer mirrors the Python operator UI from
`src/pipeline/postprocessor.py`:

- Per-detection bounding boxes are colored by warning zone (red=STOP,
  orange=CAUTION, green=MONITOR) when the foot point sits inside the
  visible corridor's bands; a small zone-coded warning glyph
  (octagon / triangle / speedometer) is drawn near the bbox top-right.
- A translucent **warning HUD** panel renders over the lower windshield/hood
  area when a
  front object falls inside a warning band: large zone icon + the
  STOP / CAUTION / MONITOR label, with an optional distance line when
  the C++ Detection has distance data populated.
- A **lane-keep arrow** panel renders in the lower-right hood area with a vector
  arrow
  pointing straight / left / right, driven by corridor curvature
  first and steering as the fallback.

All icons are drawn with OpenCV primitives — no PNG decode, no
per-frame asset loading. Toggle individually via
`OverlayConfig::show_warning_hud` and `OverlayConfig::show_lane_keep_icon`
(no CLI flags yet; both default to true).

### Video detection-only mode

```bash
./cpp/build/adas_cpp \
    --video test4_720p30.mp4 \
    --mode detection \
    --display-width 960
```

### Live CSI camera, full mode

```bash
./cpp/build/adas_cpp \
    --camera csi \
    --width 1280 --height 720 --fps 30 \
    --mode full \
    --display-width 960
```

If the Pi display/render path is the bottleneck, keep perception live but
render fewer preview frames:

```bash
./cpp/build/adas_cpp \
    --camera csi \
    --width 1280 --height 720 --fps 30 \
    --mode full \
    --display-width 960 \
    --display-every-n 2
```

`--save-video` forces `--display-every-n 1` so recorded outputs keep every
annotated frame.

### Headless smoke / dry-run

The `--dry-run [N]` flag runs `N` frames without opening a window or
calling `cv::imshow`, then exits. Useful for CI / first build sanity:

```bash
./cpp/build/adas_cpp --video test4_720p30.mp4 --mode lane --dry-run 30
# Expected: stderr lines for capture/lane/render counts and final
#   [runtime] stop | captured=… lane=… rendered=… elapsed=…s
```

### Save annotated video

```bash
./cpp/build/adas_cpp \
    --video test4_720p30.mp4 --mode lane --no-display \
    --save-video outputs/cpp_lane_test4.mp4
```

## Validation — Python vs C++ parity on the same video

The acceptance test for milestone 1 is **side-by-side replay parity** on
a deterministic video. We don't need bit-exact equality (HailoRT and
OpenCV resize at the C++/Python boundary aren't bit-exact); we need
visually identical lane geometry and steering within a small tolerance.

### A. Generate annotated outputs from both runtimes

```bash
# Python reference (existing official command, swap --camera for --video):
.\.venv\Scripts\python.exe .\src\main.py \
    --config config/stdc1_lane_hailo.yaml --mode lane --onnx \
    --video test4_720p30.mp4 --no-display \
    --save-video outputs/py_lane_test4.mp4 \
    --metrics-out outputs/py_lane_test4_metrics.jsonl

# C++ runtime, same input video:
./cpp/build/adas_cpp \
    --video test4_720p30.mp4 --mode lane --no-display \
    --save-video outputs/cpp_lane_test4.mp4
```

### B. Eyeball the videos side by side

Use any media player to overlay or alt-tab between
`outputs/py_lane_test4.mp4` and `outputs/cpp_lane_test4.mp4`. Look for:

- **Lane area fill**: same vertical extent and width per row, no
  obvious lateral shift, no extra phantom rows.
- **Red centerline**: same direction of curvature each frame, anchor at
  roughly the same bottom-screen x.
- **Yellow corridor**: parallel to red center, same half-width
  envelope.

### C. Quantitative parity (lane area IoU + steering RMSE)

There's a helper script (Python; lives at the repo root because it
drives both runtimes) at `scripts/compare_lane_cpp_vs_py.py`. It:

1. extracts per-frame `lane_area` masks from the Python metrics JSONL
   and from a parallel JSONL emitted by C++ via `--metrics-out`
   (TODO: add `--metrics-out` to `cpp/src/cli.cpp` once Python parity
   is the active blocker — currently the comparison runs from the
   saved videos via OpenCV).
2. computes per-frame IoU and mean lateral error of the fitted
   centerline.
3. reports the median IoU and 95th-percentile lateral error.

**Pass criteria for milestone 1:**

| Metric                       | Target                      |
|------------------------------|-----------------------------|
| Lane-area IoU (median frame) | ≥ 0.85 vs Python ONNX path  |
| Lane-area IoU (5th pct)      | ≥ 0.70                      |
| Steering RMSE (per frame)    | ≤ 0.10 normalised units     |
| Frame rate, headless, Pi 5   | ≥ Python `--perf` baseline  |

If we land below these the most likely cause is the simple per-row
centroid + quadratic fit in `lane_geometry.cpp` diverging from the
boundary-tracker + ego-anchored path used by the Python reference. The
C++ overlay is intentionally simpler in this milestone; full parity
needs the purple-boundary tracker + ego-anchored path which are not
yet ported.

## Implementation contract

What is **fully implemented** and considered correct for milestone 1:

1. CMake build with `find_package(OpenCV)` + manual HailoRT discovery.
2. Capture via OpenCV+GStreamer for both CSI (libcamera) and video
   files (decodebin → BGR appsink). USB fallback via V4L2.
3. HailoRT C++ VDevice owner with per-model `InferVStreams` pipelines.
   Per-model `std::mutex` around
   `infer()` mirrors the Python backend's per-model lock.
4. STDC1 lane preprocess: stretch resize to 640×640, RGB uint8 (no
   `/255` on device — matches `preprocess_lane_semantic`).
5. STDC1 decode: per-pixel argmax on NHWC or NCHW logits, nearest-
   neighbor resize to source size, lane_area + line_union masks.
6. Quadratic centerline fit by row centroids in the lower 55% of the
   frame, with weighted normal-equations solve.
7. Basic overlay: translucent lane area, red centerline polyline,
   yellow corridor edges, optional dotted/solid line markings, HUD.
8. YOLO26 preprocess + raw-head decode for the 6-tensor Hailo output,
   with class sigmoid, box decode, confidence threshold, and top-k.
9. Threaded runtime: capture → lane/object → fusion/render. Each edge is a
   drop-oldest `LatestQueue`, identical semantics to
   `src/pipeline/latest_queue.py`.
10. CLI flags listed in `--help`; `--dry-run [N]` runs headless and
   exits, useful as a smoke test.

What is **explicitly TODO** or still partial:

1. ByteTracker port from `src/features/object_tracker.py`; current tracker
   is identity-only and keeps `track_id=-1`.
2. Full behavior ladder/smoother from `src/features/behavior_logic.py`.
   Current behavior is Go/Observe from lane validity and Stop when a
   `front_object` is selected.
3. Full safety supervisor / control behavior from
   `src/pipeline/fusion_worker.py`.
4. RC actuator wrapper: port `src/features/control_runner.py`.
5. Remaining lane-parity stability gates: full validate_pair logic,
   dynamic-corridor weighting/smoothing, and exact Python metrics JSONL.
6. `--metrics-out` flag in `cli.cpp` so the parity script can
   diff JSONL outputs directly (instead of going through video).

## Risks and next steps

- **HailoRT API drift.** This code is written against the public
  `hailo/hailort.hpp` shape (VDevice, Hef, InferVStreams). If the
  installed `hailo-all` version on the target Pi exposes only the
  newer Async `InferModel` API, `cpp/src/hailo_inference.cpp` needs
  a small rewrite. Easy to spot at compile time.
- **`libcamerasrc` element naming** can vary by libcamera package
  version (sometimes `libcamerasrc`, sometimes `v4l2src` with a
  `/dev/video*` bridge). If the CSI pipeline fails on a given Pi
  install, try `--camera usb` against the libcamera-V4L2 shim.
- **Lane parity gap.** The purple/ego-anchored chain is present, but
  some Python stability gates and dynamic-corridor weighting are still
  approximated. Expect the largest differences during one-sided or
  partial lane detections.
- **No multiprocess render.** The Python Stage A `--mp-render`
  pattern is not needed here — the C++ runtime is GIL-free, so
  the threaded model is enough for milestone 1.

## Rollback

Nothing to roll back: this whole tree is new and does not modify the
Python runtime. To stop using it, just don't build / don't run
`adas_cpp`. The Python pipeline behaves exactly as before.
