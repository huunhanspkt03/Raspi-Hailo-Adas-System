# adas_cpp - Raspberry Pi 5 + Hailo ADAS runtime

`adas_cpp` is the C++17 runtime for the Raspberry Pi 5 + Hailo-8L ADAS
pipeline. The Python implementation under `../src` remains the reference and
development fallback.

The current C++ build supports one production pipeline configuration:

```text
mode            full
lane mode       python_parity
model schedule  parallel
```

Lane-only, detection-only, image lane geometry, and the alternate Hailo
schedule have been removed. The CLI rejects those old values rather than
silently running a different pipeline.

## Current status

| Subsystem | Status | Current behavior |
|---|---|---|
| CMake/OpenCV build | Implemented | C++17 executable `adas_cpp` |
| CSI, USB, video capture | Implemented | OpenCV + GStreamer/V4L2 |
| HailoRT wrapper | Implemented | Separate STDC1 and YOLO26 pipelines |
| STDC1 lane decode | Implemented | Four-class semantic mask decode |
| Lane geometry | Implemented | Boundary tracker, red center, strict lane gate, corridor |
| Steering | Implemented | Red-center shape to normalized `[-1, 1]` command |
| YOLO26 object decode | Implemented | Six raw heads, confidence filter, top-k |
| Object tracking | Implemented, lightweight | Previous-frame greedy IoU association |
| Monocular distance | Implemented | Size and ground-plane cue fusion with track smoothing |
| Lane/object fusion | Implemented | Freshest compatible object packet fused into each lane packet |
| Longitudinal control | Implemented | Distance-based following plus speed-sign throttle ceiling |
| Safety governor | Implemented | STOP/CAUTION hysteresis and stale-data supervision |
| GPS | Implemented | UART NMEA reader; speed affects the following caution distance |
| RC actuator | Implemented | Servo + forward-only BTS7960, watchdog, log-only mode |
| Operator overlay | Implemented | Lane, detections, LDWS/LKAS, FCWS, GPS, speed-limit HUD |
| Automated C++ tests | Not implemented | Build and Pi replay/live validation are currently required |

The tracker is intentionally lighter than ByteTrack, and some lane-area
weighting and stability gates are approximations of the Python reference.

## Runtime architecture

```text
capture thread
  |-- every frame --------> lane queue --> STDC1 --> lane geometry --+
  `-- every Nth frame ----> object queue --> YOLO26 --> distance ----+--> fusion
                                                                      |
                                                                      v
                                                            behavior/render thread
                                                              |-- overlay/display
                                                              |-- video writer
                                                              `-- safety --> RC
```

All inter-stage queues are bounded `LatestQueue` instances with capacity one.
When a consumer falls behind, the oldest queued item is dropped. This keeps
the system close to real time instead of accumulating latency.

The lane branch drives fusion and rendering. YOLO runs every
`--object-every-n` captured frames, default `2`. Fusion accepts the newest
object packet satisfying:

```text
object.frame_id <= lane.frame_id
lane.frame_id - object.frame_id <= 8
```

Values of `--object-every-n` above `8` are clamped to `8`, because they would
otherwise exceed the fusion stale-packet tolerance.

## Build

### Raspberry Pi 5, Bookworm, aarch64

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config \
    libopencv-dev libgstreamer1.0-dev \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-libcamera gstreamer1.0-plugins-bad

sudo apt install -y hailo-all
sudo reboot
```

From the repository root:

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j
```

The executable is `cpp/build/adas_cpp`.

### Development build without Hailo

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DADAS_USE_HAILO=OFF
cmake --build cpp/build -j
```

The non-Hailo backend is a compile/smoke-test stub. It does not produce lane
or object inference results and is not a product acceptance path.

## Run

Run commands from the repository root so the default model paths resolve:

```text
src/models/stdc1_lane_640.hef
src/models/yolo26n_object.hef
```

### Video replay

```bash
./cpp/build/adas_cpp \
    --video video_test3.mp4 \
    --mode full \
    --lane-mode python_parity \
    --model-schedule parallel \
    --display-width 960
```

### Live CSI camera

```bash
./cpp/build/adas_cpp \
    --camera csi \
    --width 1280 --height 720 --fps 30 \
    --mode full \
    --lane-mode python_parity \
    --model-schedule parallel \
    --display-width 1280
```

### Operator preset

`--operator-mode` applies:

```text
1280x720 at 30 FPS
display width 1280
GPS enabled
RC enabled
quiet logs enabled
```

It does not choose the input source:

```bash
./cpp/build/adas_cpp --camera csi --operator-mode
```

This command arms the RC layer. During bench testing, add `--rc-log-only`:

```bash
./cpp/build/adas_cpp \
    --camera csi \
    --operator-mode \
    --rc-log-only
```

### Headless smoke test

```bash
./cpp/build/adas_cpp \
    --video video_test3.mp4 \
    --mode full \
    --dry-run 30
```

`--dry-run` disables the display and exits after the requested number of
rendered packets.

### Save annotated video

```bash
./cpp/build/adas_cpp \
    --video video_test3.mp4 \
    --mode full \
    --no-display \
    --save-video outputs/cpp_full_video_test3.mp4
```

Saving video forces `--display-every-n 1` so every rendered packet is written.

### Profiling and evaluation logs

```bash
./cpp/build/adas_cpp \
    --video video_test3.mp4 \
    --mode full \
    --no-display \
    --profile \
    --profile-log-every 100 \
    --eval-log
```

`--profile` records per-stage EMA/max timing. `--eval-log` prints frame
counts, average FPS, queue drops, stale fusion packets, and GPS statistics at
shutdown.

## Perception and control behavior

### Lane and steering

STDC1 receives a stretch-resized `640x640` RGB `uint8` frame. Its four-class
output is decoded into:

- `lane_area`: drivable lane mask.
- `line_union`: dotted and solid lane-marking mask.

The parity tracker fits left/right boundaries, applies temporal EMA, and can
hold a missing side for up to eight lane packets. It then builds:

- A red, camera-anchored center path.
- A yellow display/drive corridor.
- A narrower strict lane gate used to select the FCWS front vehicle.

Steering is normalized to `[-1, 1]`. Positive values steer right. It is based
on the red path's shape between the near point and lookahead point, not merely
the path's absolute image offset.

If dynamic geometry cannot be resolved, the UI receives a fixed straight
guidance template with confidence `0`. That fallback remains visible but is
not treated as a valid lane lock by the safety layer.

### Object detection and distance

YOLO26 recognizes:

```text
bus, car, motorcycle, truck,
speed_limit_50, speed_limit_60, speed_limit_70, speed_limit_80
```

Vehicles are associated between consecutive object frames using same-class
IoU matching. Distance combines:

- Class-specific physical-size priors.
- Bbox-foot ground-plane geometry.
- A near-field guardrail.
- Per-track confidence-adaptive temporal smoothing.

Only vehicles whose bbox bottom-center lies inside the strict lane gate can
become `front_object`. The nearest valid vehicle is selected by estimated
forward distance; bbox area is the fallback when range is unavailable.

Distance calibration can be changed without rebuilding:

```text
ADAS_CALIB_FOOT_Y
ADAS_CALIB_Z_M
ADAS_CAMERA_TO_BUMPER_M
ADAS_DISTANCE_SCALE
ADAS_DISTANCE_OFFSET_M
```

### Following controller

The selected in-lane `front_object` controls longitudinal throttle:

| Condition | Result |
|---|---|
| Display distance `<= 3 m` | Immediate STOP, throttle `0` |
| Between `3 m` and caution distance | CAUTION, linear throttle ramp |
| Beyond caution distance | MONITOR, cruise throttle |
| Front object without usable distance | Conservative throttle `0.5` |
| No front object | Base GO throttle, still subject to the sign ceiling |

The caution distance is:

- `10 m` when GPS is unavailable or ego speed is at most `40 km/h`.
- `15 m` when valid GPS ego speed is above `40 km/h`.

Throttle changes are slew-limited, except an immediate hard stop.

Speed-limit detections set a maximum throttle:

```text
50 km/h -> 0.60
60 km/h -> 0.80
70 km/h -> 0.80
80 km/h -> 1.00
```

Before seeing a sign, the default ceiling is `0.80`/60 km/h. A detected sign
is adopted immediately and held until a different speed-limit sign is seen.

Relative speed, estimated front-vehicle speed, and TTC are display/log data.
The current motor controller is distance-based; TTC does not command throttle.

## Safety terminology and behavior

### Packet

A packet is a typed result passed between pipeline stages, such as
`FramePacket`, `LanePacket`, `ObjectPacket`, or `BehaviorPacket`. A packet is
not guaranteed to correspond to every camera frame because capacity-one
queues may drop old data.

### Stale

Stale means data is too old to be safely paired with the current result. For
example, an object packet more than eight frame IDs behind the lane packet is
not fused.

### SafetyGovernor

`SafetyGovernor` is the final software supervisor before RC commands:

- STOP engages immediately and remains held for five subsequent non-STOP
  behavior packets.
- CAUTION is held for three subsequent non-CAUTION packets.
- Invalid lane geometry is tolerated for eight packets.
- Lane loss alone does not stop the motor. After the grace period it becomes
  unsafe only when a selected front object is within `4 m`.
- More than ten consecutive behavior packets without fresh object data sets
  `safe=false`.
- Any `safe=false` result forces throttle to `0`.

These values count processed behavior packets, not wall-clock time.

### RC watchdog

`RcActuator::command()` only stores the latest desired command. A separate
50 Hz watchdog thread performs GPIO/PWM writes.

If no new command arrives for `0.5 s`, or if `safe=false`, the watchdog sets
motor PWM to zero. On shutdown it also centers the servo and drops both
BTS7960 enable pins.

The actuator is forward-only:

```text
servo        GPIO13
BTS7960 RPWM GPIO18
BTS7960 LPWM GPIO23, always LOW
BTS7960 R_EN GPIO27
BTS7960 L_EN GPIO22
```

## Validation

Minimum validation after a code change:

1. Build the C++ executable.
2. Run `--help` and confirm the documented CLI flags.
3. Run a video `--dry-run` on the Pi with both HEFs loaded.
4. Replay with the live overlay visible and inspect lane, object selection,
   distance, FCWS, steering, and throttle.
5. Before connecting the motor driver, repeat with `--rc-log-only`.
6. Run the official live CSI command with the display enabled for product
   acceptance.

Headless FPS alone is diagnostic evidence, not acceptance of the operator
pipeline.

## Known limitations and risks

- The C++ tracker is greedy previous-frame IoU association, not ByteTrack.
- Lane-area weighting and some Python stability gates remain approximated.
- Monocular distance depends strongly on camera height, field of view, pitch,
  bumper offset, and bbox-foot quality. Calibrate it on the installed camera.
- `--loop-video` rewinds capture without resetting all temporal trackers and
  controllers; state may carry across the replay boundary.
- Speed-limit state is held indefinitely until another sign is detected.
- The RC backend uses Linux sysfs PWM/GPIO and is pinned to
  `/sys/class/pwm/pwmchip0`; verify the target Pi mapping before hardware use.
- HailoRT C++ API differences between package versions may require wrapper
  changes at compile time.
- There are currently no automated C++ unit/integration tests.

## Layout

```text
cpp/
|-- CMakeLists.txt
|-- README.md
|-- include/adas/
|   |-- runtime.hpp
|   |-- types.hpp
|   |-- latest_queue.hpp
|   |-- hailo_inference.hpp
|   |-- lane_decode.hpp
|   |-- lane_parity.hpp
|   |-- lane_parity_render.hpp
|   |-- object_detection.hpp
|   |-- object_distance.hpp
|   |-- behavior.hpp
|   |-- gps.hpp
|   `-- rc_actuator.hpp
`-- src/
    |-- main.cpp
    |-- runtime.cpp
    |-- lane_*.cpp
    |-- object_*.cpp
    |-- fusion.cpp
    |-- behavior.cpp
    |-- overlay.cpp
    |-- gps.cpp
    `-- rc_actuator.cpp
```
