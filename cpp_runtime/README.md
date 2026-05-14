# adas_cpp_runtime

C++17 runtime for the ADAS lane-centering pipeline on Raspberry Pi 5 + Hailo-8 . This is a thin deploy-only runtime — training, export and HEF
compilation stay in Python.

Supports exactly **two** models, nothing else:

| Branch             | Model                          | Input size | Format          |
|--------------------|--------------------------------|------------|-----------------|
| Lane segmentation  | DeepLabV3 + MobileNetV2        | 512x512    | RGB uint8       |
| Object detection   | YOLO26s                        | 640x640    | RGB uint8 (letterbox, pad=114) |

Three runtime modes: `lane`, `detection`, `full`.

## DeepLab Model Artifacts

DeepLab now has two distinct ONNX artifacts on the Python/export side:

| Artifact | Default path | Purpose |
|---|---|---|
| Host ONNX | `src/models/deeplab_v3_mobilenet_v2.onnx` | Windows/x86 dev inference, parity checks, debug overlays (`NHWC -> ResizeBilinear:0`) |
| Hailo ONNX (low-res) | `src/models/deeplab_v3_mobilenet_v2_hailo.onnx` | Hailo compile baseline with `1x6x64x64` logits |
| Hailo ONNX (full-res) | `src/models/deeplab_v3_mobilenet_v2_hailo_fullres.onnx` | Hailo compile experiment with in-graph bilinear upsample to preserve fine lines better |
| Hailo ONNX (published-like) | `src/models/deeplab_v3_mobilenet_v2_hailo_published_like.onnx` | `512x512 NHWC -> ResizeBilinear:0` graph shaped to resemble Hailo-published DeepLab host exports while matching this repo |
| Hailo HEF (current default) | `src/models/deeplab_v3_mobilenet_v2_2304.hef` | Current Python and C++ runtime default |
| Hailo HEF (published-like) | `src/models/deeplab_v3_mobilenet_v2_published_like.hef` | Optional published-like ONNX contract compile |

Recommended export flow from the repo root:

```bash
python scripts/export_deeplabv3_mobilenetv2.py --variant all \
    --weights checkpoints/DeeplabV3_mobilenetV2/best.pth --imgsz 512
python scripts/build_calib_deeplabv3_mobilenetv2.py --imgsz 512 --clean
bash scripts/compile_hailo_deeplab.sh
```

The compile script now defaults to `src/models/deeplab_v3_mobilenet_v2_hailo_fullres.onnx`.
To fall back to the smaller low-resolution contract:

```bash
DEEPLAB_OUTPUT_CONTRACT=lowres bash scripts/compile_hailo_deeplab.sh
```

To try the Hailo-published-like DeepLab contract:

```bash
python scripts/build_calib_deeplabv3_mobilenetv2.py --imgsz 512 --dst calib_data/deeplab_512 --clean
DEEPLAB_OUTPUT_CONTRACT=published_like bash scripts/compile_hailo_deeplab.sh
```

For the most aggressive quality-oriented DeepLab compile profile currently in
this repo, use:

```bash
DEEPLAB_OUTPUT_CONTRACT=published_like \
DEEPLAB_MODEL_SCRIPT=./config/hailo/deeplab_v3_mobilenet_v2_fullquality.alls \
bash scripts/compile_hailo_deeplab.sh
```

If that still is not enough and the host has more headroom, use the
workspace-tuned ultra profile:

```bash
DEEPLAB_OUTPUT_CONTRACT=published_like \
DEEPLAB_MODEL_SCRIPT=./config/hailo/deeplab_v3_mobilenet_v2_ultraquality.alls \
bash scripts/compile_hailo_deeplab.sh
```

## Layout

```
cpp_runtime/
├── CMakeLists.txt
├── README.md
├── include/adas/
│   ├── camera_source.hpp
│   ├── controller.hpp
│   ├── deeplab_postprocess.hpp
│   ├── hailo_backend.hpp
│   ├── inference_backend.hpp
│   ├── preprocess.hpp
│   ├── visualization.hpp
│   └── yolo26s_postprocess.hpp
└── src/
    ├── camera_source.cpp
    ├── controller.cpp
    ├── deeplab_postprocess.cpp
    ├── hailo_backend.cpp
    ├── main.cpp
    ├── preprocess.cpp
    ├── visualization.cpp
    └── yolo26s_postprocess.cpp
```

## Dependencies

- CMake ≥ 3.16
- C++17 compiler
- OpenCV 4.x (core, imgproc, imgcodecs, videoio, highgui)
- **Optional on Pi**: HailoRT (`hailort-dev` package, provides `hailo/hailort.hpp`
  and `libhailort.so`).

On a dev machine without HailoRT the project still builds with
`-DUSE_HAILO=OFF`. In that mode `HailoBackend` is a stub that returns empty
inference output, so the camera/video source, preprocess, visualization and
steering code paths can all be exercised.

## Build — Windows dev machine (no HailoRT)

`opencv-python` from pip is not enough for this C++ build. CMake needs the
OpenCV C++ development package, including headers, libraries, and
`OpenCVConfig.cmake`.

With vcpkg:

```powershell
git clone https://github.com/microsoft/vcpkg C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
C:\vcpkg\vcpkg.exe install opencv4:x64-windows

cmake -S cpp_runtime -B cpp_runtime/build -DUSE_HAILO=OFF `
  -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build cpp_runtime/build --config Release
```

With a prebuilt OpenCV package, point CMake at the directory that contains
`OpenCVConfig.cmake`:

```powershell
cmake -S cpp_runtime -B cpp_runtime/build -DUSE_HAILO=OFF `
  -DOpenCV_DIR=C:/opencv/build/x64/vc16/lib
cmake --build cpp_runtime/build --config Release
```

## Build — Linux/dev machine (no HailoRT)

```bash
sudo apt install libopencv-dev
cmake -S cpp_runtime -B cpp_runtime/build -DUSE_HAILO=OFF
cmake --build cpp_runtime/build -j
```

## Build — Raspberry Pi 5 (with HailoRT)

```bash
sudo apt install libopencv-dev hailort-dev
cmake -S cpp_runtime -B cpp_runtime/build -DUSE_HAILO=ON
cmake --build cpp_runtime/build -j
```

The build looks for `hailo/hailort.hpp` and `libhailort` in the standard
system paths (`/usr/include`, `/usr/lib`, `/usr/lib/aarch64-linux-gnu`). If
HailoRT is installed elsewhere, pass the paths:

```bash
cmake -S cpp_runtime -B cpp_runtime/build -DUSE_HAILO=ON \
      -DHAILORT_INCLUDE_DIR=/opt/hailo/include \
      -DHAILORT_LIBRARY=/opt/hailo/lib/libhailort.so
```

## Run

Binary lives at `cpp_runtime/build/adas_runtime`. All paths below are
relative to the repo root, so the default HEF locations
(`src/models/*.hef`) match the existing Python pipeline.

Lane only — DeepLabV3 512x512:

```bash
./cpp_runtime/build/adas_runtime --mode lane \
    --video test1.mp4 --display \
    --deeplab-hef src/models/deeplab_v3_mobilenet_v2_2304.hef
```

Detection only — YOLO26s 640x640:

```bash
./cpp_runtime/build/adas_runtime --mode detection \
    --video test1.mp4 --display \
    --yolo26s-hef src/models/yolo26_adas_2404.hef
```

Full pipeline (both branches):

```bash
./cpp_runtime/build/adas_runtime --mode full \
    --video test1.mp4 --display \
    --deeplab-hef src/models/deeplab_v3_mobilenet_v2_2304.hef \
    --yolo26s-hef src/models/yolo26_adas_2404.hef
```

Full pipeline with video paced to the source playback speed:

```bash
./cpp_runtime/build/adas_runtime --mode full \
    --video test1.mp4 --display --realtime-video \
    --deeplab-hef src/models/deeplab_v3_mobilenet_v2_2304.hef \
    --yolo26s-hef src/models/yolo26_adas_2404.hef
```

Live camera on the Pi (CSI or USB via V4L2):

```bash
./cpp_runtime/build/adas_runtime --mode full --camera \
    --width 1280 --height 720 --fps 30 --display
```

`--display` only opens the live OpenCV window. It does not write any MP4 unless
you also pass `--save-video` or `--save-mask-video`.

If you launch the runtime over SSH without a Pi desktop session or X11
forwarding, the OpenCV window will not appear.

Record an output video:

```bash
./cpp_runtime/build/adas_runtime --mode full --video test1.mp4 \
    --save-video outputs/cpp_full.mp4
```

Record only the lane segmentation overlay (no HUD, no detections — useful for
verifying the DeepLab HEF output visually on the Pi):

```bash
./cpp_runtime/build/adas_runtime --mode lane --video test1.mp4 \
    --save-mask-video outputs/cpp_lane_mask.mp4
```

Debug run with every knob turned up (load diagnostics + per-frame Hailo stats
every 30 frames + per-class pixel counts every 60 frames + stdout stats every
30 frames):

```bash
ADAS_LOG_EVERY_N=30 ADAS_DEEPLAB_DEBUG=1 \
./cpp_runtime/build/adas_runtime --mode lane --video test1.mp4 \
    --stats-every 30 --save-mask-video outputs/cpp_lane_mask.mp4
```

## CLI flags

| Flag             | Description                                         |
|------------------|-----------------------------------------------------|
| `--mode`         | `lane`, `detection` or `full`                       |
| `--video PATH`   | Read frames from a video file                       |
| `--camera`       | Read from the first V4L2 camera                     |
| `--deeplab-hef`  | Path to DeepLabV3 HEF (default: `src/models/deeplab_v3_mobilenet_v2_2304.hef`) |
| `--yolo26s-hef`  | Path to YOLO26s HEF (default: prefer `src/models/yolo26_adas_2404.hef`, fallback `src/models/yolo26_adas_1404.hef`)  |
| `--display`      | Show an OpenCV window only; does not save output by itself |
| `--realtime-video` | Pace `--video` playback to the source timestamps when possible |
| `--save-video`   | Write an annotated MP4 (HUD + lane + detections) when explicitly requested |
| `--save-mask-video` | Write a raw-frame + colorized-lane MP4 (no HUD, no detections) when explicitly requested |
| `--stats-every N` | stdout `[stats]` line cadence (default 60, 0 disables) |
| `--width/--height/--fps` | Camera request parameters                   |

### Logging env vars

Runtime logging is env-gated so production runs stay quiet:

| Env var                 | Effect                                                                                   |
|-------------------------|------------------------------------------------------------------------------------------|
| `ADAS_LOG_EVERY_N=<N>`  | `HailoBackend` prints `[hailo] [tag] frame=F avg_infer=X fps=Y last=Z` + raw-output head stats every N frames (per model). |
| `ADAS_DEEPLAB_DEBUG=1`  | `DeeplabPostprocessor` prints `[deeplab] frame=F mask=WxH classes=C pixels: background(0)=... dotted_yellow(1)=...` every 60 frames. |

Load-time diagnostics (HailoRT version, HEF path, per-stream name/shape/frame_size,
load wall-time) are always on and fire once per model load. The DeepLab tensor
shape also logs once per run so you can verify the HEF's raw output layout.

## Data flow

```
CameraSource (BGR)
  ├── Preprocessor::preprocessDeeplab512  → HailoBackend::infer(Deeplab)
  │     → DeeplabPostprocessor
  │         → argmax class mask → lane_area → morphology
  │         → per-row centerline → Controller (steering)
  │
  └── Preprocessor::preprocessYolo26s640  → HailoBackend::infer(Yolo26s)
        → Yolo26sPostprocessor (end2end OR raw-head)
        → letterbox-inverse → NMS → Detection list

Visualizer → lane overlay + detections + HUD → (display / save_video)
```

## What is implemented

- Full video/camera IO loop with loop-on-EOF for videos.
- Deterministic 512x512 DeepLab preprocess and 640x640 letterboxed YOLO26s
  preprocess matching `src/pipeline/preprocessor.py`.
- DeepLab postprocess:
  - NHWC / NCHW auto-detection,
  - per-pixel argmax → class mask,
  - morphological open+close on the lane_area channel,
  - a simple per-row centerline (bottom-weighted) that feeds the steering
    controller.
- YOLO26s postprocess supports **both** output shapes:
  - End-to-end Ultralytics-style `(1,N,6)` / `(N,6)` → direct decode.
  - Hailo raw-head 6 tensors (3 box, 3 class) → ported from
    `src/features/yolo26_raw_head.py` (anchor grid, `dist2bbox`, stride
    scaling, sigmoid, class-wise NMS).
- `Controller`: pixel error → normalized steering + EMA smoothing.
- `Visualizer`: lane area fill, centerline, control point, bounding boxes,
  HUD (mode, FPS, infer ms, steering, pixel error).
- HailoRT C++ backend guarded by `ADAS_USE_HAILO`:
  - `-DUSE_HAILO=ON`: loads HEF files via `Hef` + `VDevice::configure` and
    runs through input/output vstreams.
  - `-DUSE_HAILO=OFF`: stub that prints a warning and returns empty tensors,
    so the rest of the runtime still builds and runs on a dev machine.

## Differences from the Python pipeline (intentional scope cuts)

- The DeepLab input is **fixed at 512x512** here, matching the current Python
  configs and Hailo compile flow in this repo.
- Lane centerline uses a lightweight per-row centroid instead of the BEV
  quadratic fit in `src/features/lane_center_path.py`. The steering signal
  is stable enough for an MVP; port the BEV fit later if curved-road
  behaviour needs to match Python exactly.
- Behaviour logic (`behavior_logic.py`) is not ported — `Controller` only
  computes steering. Throttle/stop decisions are a TODO.
- RC hardware output is a no-op — `main.cpp` computes the steering value
  but does not forward it to any actuator. Plug RC output in next to the
  call to `controller.compute(...)`.

## Validation status

- CMake requires an OpenCV C++ SDK. If configure fails with missing
  `OpenCVConfig.cmake`, install OpenCV development files or pass
  `-DOpenCV_DIR=...` as shown above.
- `-DUSE_HAILO=OFF` is intended to build on a dev machine once OpenCV is
  available.
- The HailoRT-linked build has **not** been compiled on an actual Pi from this
  checkout — see TODO below.

## TODO / not yet tested

- **HailoRT build on the Pi**: link and smoke-test `-DUSE_HAILO=ON` on a
  real Raspberry Pi 5. The code targets the `Hef` + vstream API; if your
  HailoRT version ships only the newer `InferModel` API, the `LoadedModel`
  struct in `hailo_backend.cpp` will need to be swapped over.
- **YOLO26s raw-head tensor ordering**: the raw-head path groups the 6
  tensors by channel count (4 → box, 11 → class) and then sorts by
  spatial area to map them to strides 8/16/32. The Python reference also
  looks at tensor names (`one2one_cv2_*`, `one2one_cv3_*`). If the HEF
  exposes names, add the same regex match for robustness.
- **Behaviour logic + RC output**: port `behavior_logic.py` and wire a
  real RC driver into `main.cpp`.
- **Threaded capture / decoupled display**: the C++ runtime is
  currently synchronous.
