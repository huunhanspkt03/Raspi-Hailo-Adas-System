# Edge-AI ADAS for Raspberry Pi 5 + Hailo

[![C++](https://img.shields.io/badge/C%2B%2B-17-004482?style=flat-square&logo=cplusplus&logoColor=white&labelColor=00599C)](https://isocpp.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Vision-3B8C3B?style=flat-square&logo=opencv&logoColor=white&labelColor=5C3EE8)](https://opencv.org/)
[![GStreamer](https://img.shields.io/badge/GStreamer-Camera%20Pipeline-D71920?style=flat-square&logo=gstreamer&logoColor=white&labelColor=FF3131)](https://gstreamer.freedesktop.org/)
[![CMake](https://img.shields.io/badge/CMake-3.16%2B-043F70?style=flat-square&logo=cmake&logoColor=white&labelColor=064F8C)](https://cmake.org/)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-5-8A1538?style=flat-square&logo=raspberrypi&logoColor=white&labelColor=C51A4A)](https://www.raspberrypi.com/)
[![Hailo](https://img.shields.io/badge/Hailo-8%20AI%20Accelerator-007EA7?style=flat-square&labelColor=00AEEF)](https://hailo.ai/)
[![Linux](https://img.shields.io/badge/Linux-Runtime-E0A800?style=flat-square&logo=linux&logoColor=black&labelColor=FCC624)](https://www.kernel.org/)
[![STDC1](https://img.shields.io/badge/STDC1-Lane%20Segmentation-1B5E20?style=flat-square&labelColor=2E7D32)](https://github.com/MichaelFan01/STDC-Seg)
[![YOLO26n](https://img.shields.io/badge/YOLO26n-Object%20Detection-E65100?style=flat-square&labelColor=F57C00)](https://github.com/ultralytics/ultralytics)
[![ADAS](https://img.shields.io/badge/ADAS-FCWS%20%7C%20LDWS%20%7C%20LKAS-0D47A1?style=flat-square&labelColor=1565C0)](https://www.nhtsa.gov/vehicle-safety/driver-assistance-technologies)
[![Claude](https://img.shields.io/badge/Claude-AI%20Assisted%20Development-B85C38?style=flat-square&labelColor=D97757)](https://www.anthropic.com/claude)

Example C++ runtime for real-time ADAS on Raspberry Pi 5 using the Hailo AI accelerator.

Lane perception uses an **STDC1 semantic segmentation** model compiled as HEF for Hailo. The runtime decodes the lane mask into lane area, lane-line cues, a stabilized centerline, strict in-lane gate, and steering guidance.

Object perception uses a **YOLO26n raw-head detector** compiled as HEF for Hailo. The runtime decodes vehicles and speed-limit signs, tracks detections, estimates monocular distance, selects the in-lane front object, and renders FCWS status.

The full system combines **FCWS**, **LDWS/LKAS**, GPS speed display, video replay, live CSI/USB camera input, and optional RC vehicle actuation through servo steering and a BTS7960 forward-only motor driver.

##  Contents

1. [Key Features](#key-features)
2. [Hardware Platform](#hardware-platform)
3. [Requirements](#requirements)
4. [Examples](#examples)
5. [Architecture](#architecture)
6. [Models](#models)
7. [Project Layout](#project-layout)
8. [Results](#results)
9. [License](#license)

## Key Features

- **Real-time lane segmentation** using STDC1 on Hailo HEF.
- **Lane geometry and steering guidance** with stabilized centerline and strict in-lane gate.
- **YOLO26n object detection** for vehicles and speed-limit signs.
- **Forward Collision Warning System (FCWS)** with front-object distance, TTC, and risk display.
- **Lane Departure Warning / Lane Keeping Assist (LDWS/LKAS)** with left/right guidance.
- **GPS HUD** for ego speed, satellite count, and fix status.
- **RC vehicle control path** for servo steering and forward-only motor throttle.

## Hardware Platform

| Component | Usage |
| --- | --- |
| Raspberry Pi 5 | Main edge runtime and camera pipeline |
| Hailo-8L AI Accelerator | Lane segmentation and object detection inference |
| CSI / USB Camera | Live road input |
| U-Blox NEO-M10 GPS | Ego speed and position display |
| Servo | Steering actuator for RC vehicle validation |
| BTS7960 Motor Driver | Forward-only throttle control |

## Requirements

### Raspberry Pi 5 runtime

- Raspberry Pi 5
- Hailo AI HAT / Hailo-8 runtime
- HailoRT headers and library, usually installed through `hailo-all`
- OpenCV with GStreamer support
- CMake 3.16+
- C++17 compiler
- Optional U-Blox NEO-M10 GPS over UART
- Optional servo + BTS7960 motor driver for RC actuation

### Development machine

- CMake 3.16+
- C++17 compiler
- OpenCV development package
- Optional Python tools from `requirements.txt` for training/export/reference workflows

## Examples

Build on Raspberry Pi or a Linux development machine:

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j
```

Build without HailoRT for development inspection:

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DADAS_USE_HAILO=OFF
cmake --build cpp/build -j
```

Run with the Raspberry Pi CSI camera:

```bash
./cpp/build/adas_cpp --operator-mode --camera csi
```

## Architecture

The runtime is a parallel full-mode C++ pipeline:

```text
Capture
  |-- lane queue   -> STDC1 Hailo inference -> lane decode -> lane geometry
  |-- object queue -> YOLO26 Hailo inference -> raw-head decode -> tracking -> distance

Lane packet + latest object packet
  -> fusion
  -> behavior and follow controller
  -> overlay / display / optional video save
  -> optional RC actuator command
```

Lane inference runs on every captured frame. Object inference runs every `N` frames through `--object-every-n`, and fusion reuses the latest fresh object packet.

## Models

Default model paths used by the C++ runtime:

| Task | Model |
| --- | --- |
| Lane segmentation | `src/models/stdc1_lane_640.hef` |
| Object detection | `src/models/yolo26n_object.hef` |

## Project Layout

```text
cpp/
  CMakeLists.txt
  include/adas/      C++ headers for capture, inference, fusion, behavior, RC, GPS, overlay
  src/               C++ runtime implementation

assets/              README screenshots and UI examples
Outputs/             Local generated video outputs
checkpoints/         Training checkpoints and model artifacts
Hailo_dataflow_compiler/
                     Local Hailo compiler installer/log artifacts
requirements.txt     Python-side training/export/reference dependencies
```

## Results

| Function | Output |
| --- | --- |
| FCWS | Front object distance, TTC, warning level, and selected object box |
| LDWS / LKAS | Lane keeping state, turn direction, and steering guidance |
| Speed Limit Sign | Detected speed limit and held throttle ceiling |
| GPS HUD | Ego speed, satellite count, and fix age |
| Replay Output | Annotated video showing the full ADAS overlay |

### Replay Output

<p align="center">
  <video src="Outputs/Video%20Result.mp4" controls width="780">
    <a href="Outputs/Video%20Result.mp4">Watch the replay output video</a>
  </video>
</p>

> Note: the embedded player works only if `Outputs/Video Result.mp4` is committed or replaced with a GitHub-hosted release/demo asset URL.

### Forward Collision Warning System

<p align="center">
  <img src="assets/Forward%20Collision%20Warning%20System.jpg" alt="Forward Collision Warning System output" width="780">
</p>

### Lane Keeping Assist / Lane Departure Warning

<p align="center">
  <img src="assets/Lane%20Keeping%20.jpg" alt="Lane keeping output" width="780">
</p>

### Speed Limit Sign Display

<p align="center">
  <img src="assets/Speed%20Limit%20Sign.jpg" alt="Speed limit sign output" width="780">
</p>

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
