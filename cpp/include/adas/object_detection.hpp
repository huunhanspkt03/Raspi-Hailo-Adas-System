#pragma once

// C++ port of the YOLO26 raw-head decode used on Pi/Hailo. Mirrors:
//   src/features/object_detection.py   (decode_yolo26)
//   src/features/yolo26_raw_head.py    (raw-head: dist2bbox + top-k)
//   src/features/detection_filters.py  (CLASS_ID_TO_NAME, leading-vehicle)
//
// Hailo HEF output: 3 box heads + 3 class heads (NHWC float32 after the
// FLOAT32-output vstream params). One-to-one heads are NMS-free; we
// take per-anchor argmax over classes, threshold, sort, and top-k.
//
// Letterbox preprocess matches pipeline.preprocessor.preprocess_yolo26
// (640x640 RGB uint8, pad value 114). Boxes are scaled back from model
// coords to source-frame coords using the letterbox parameters.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "adas/hailo_inference.hpp"
#include "adas/lane_geometry.hpp"
#include "adas/types.hpp"

namespace adas {

constexpr int kYolo26InputSize = 640;
constexpr std::uint8_t kYolo26PadValue = 114;
constexpr int kYolo26NumClasses = 8;
constexpr int kYolo26RegMax = 16;
constexpr int kYolo26MaxDet = 300;

// Class label table — mirrors features/detection_filters.py::CLASS_ID_TO_NAME.
const char* yolo26_class_name(int class_id);
bool yolo26_is_vehicle(int class_id);
// Traffic-sign grouping. Mirrors features/detection_filters.py
// SPEED_LIMIT_LABELS / resolve_speed_sign. The only sign classes currently
// in the model label table are speed_limit_{50,60,70,80}; other sign
// families (directional, prohibition) are empty here as in Python.
bool yolo26_is_traffic_sign(int class_id);
// Returns 0 when the class is not a speed-limit sign, else the limit in
// km/h (50/60/70/80).
int  yolo26_speed_limit_kmh(int class_id);
// Highest-confidence speed-limit sign label in the batch, or nullptr.
// Mirrors detection_filters.resolve_speed_sign for log-only consumers.
const char* yolo26_resolve_speed_sign(const std::vector<Detection>& dets);
// Upper-bound throttle (0.0..1.0) for the given class id. Returns 1.0 for
// classes that are not speed-limit signs, so the value is always safe to
// pass through std::min(throttle, cap). Mapping (user-specified; the table
// in object_detection.cpp::kSpeedLimitThrottleCap is the source of truth):
//   speed_limit_50 -> 0.60
//   speed_limit_60 -> 0.80
//   speed_limit_70 -> 0.80
//   speed_limit_80 -> 1.00
float yolo26_speed_limit_throttle_cap(int class_id);

struct ObjectDecodeConfig {
    float conf_thresh{0.50f};
    int   yolo_input_size{kYolo26InputSize};
    int   pad_value{kYolo26PadValue};
    int   max_det{kYolo26MaxDet};
};

// Letterbox + RGB + flatten. Returns a contiguous uint8 buffer of size
// (input_size*input_size*3) suitable for Hailo input.
std::vector<std::uint8_t> preprocess_yolo26(const cv::Mat& frame_bgr,
                                            int input_size = kYolo26InputSize,
                                            std::uint8_t pad_value = kYolo26PadValue);
void preprocess_yolo26(const cv::Mat& frame_bgr,
                       int input_size,
                       std::uint8_t pad_value,
                       std::vector<std::uint8_t>* out,
                       cv::Mat* resize_scratch);

// Decode the 6-tensor raw-head output into source-frame Detections.
std::vector<Detection> decode_yolo26(const InferOutputs& raw,
                                     int frame_w, int frame_h,
                                     const ObjectDecodeConfig& cfg);

// Lightweight greedy IoU tracker: assigns a stable track_id to each detection
// by matching it (same class, max IoU above a threshold) against the previous
// frame's detections, so downstream per-track temporal smoothing has a key to
// associate on. Not a full ByteTracker (no motion model, no coast across
// misses) — a missed frame simply starts a fresh id, which only re-anchors the
// smoother. Stateful and single-threaded (the object worker thread only).
std::vector<Detection> update_tracker(std::vector<Detection> dets);

// Clear tracker state (next frame's detections all get fresh ids). Call on a
// video/stream switch so ids/poses do not leak across a cut.
void reset_object_tracker();

// Pick the closest in-lane vehicle using LaneGeometry::strict_lane_* as the
// object gate. Lane membership is decided by the bbox bottom-center point in
// the inner band of the strict lane. The wider rendered corridor is
// intentionally not a fallback for FCWS.
std::optional<Detection> pick_front_object(const std::vector<Detection>& dets,
                                           const LaneGeometry& lane_geom,
                                           int frame_w, int frame_h,
                                           FrameId debug_frame_id = 0);
// (pick_speed_object was removed: it ran the identical strict gate, so
// fusion now assigns speed_object = front_object after one gate pass.)

// ---- raw-head helpers exposed for unit-style inspection / reuse ----
struct RawHeadTriplet {
    const InferTensor* box_head{nullptr};
    const InferTensor* cls_head{nullptr};
    int stride{0};
    int grid_h{0};
    int grid_w{0};
};

// Group the 6 output tensors into stride-ordered (box,cls) triplets.
// Returns false if the layout is not the expected raw-head 6-tensor set.
bool group_yolo26_raw_heads(const InferOutputs& raw,
                            std::vector<RawHeadTriplet>* out_triplets);

}  // namespace adas
