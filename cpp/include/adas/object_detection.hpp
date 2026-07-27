#pragma once

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

//   speed_limit_50 -> 0.60
//   speed_limit_60 -> 0.80
//   speed_limit_70 -> 0.80
//   speed_limit_80 -> 1.00
float yolo26_speed_limit_throttle_cap(int class_id);

struct ObjectDecodeConfig {
    float conf_thresh{0.50f};
    float nms_iou_thresh{0.60f};  // <=0 disables duplicate suppression
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
std::vector<Detection> update_tracker(std::vector<Detection> dets);

// Clear tracker state (next frame's detections all get fresh ids). Call on a
// video/stream switch so ids/poses do not leak across a cut.
void reset_object_tracker();

class FrontObjectNearLaneMemory {
public:
    // Max fused frames the remembered width stays usable after the current gate
    // goes invalid (within the requested 10..15 window).
    static constexpr int kMaxHeldFrames = 12;

    void store(float near_width_px, float near_center_x) {
        near_width_px_ = near_width_px;
        near_center_x_ = near_center_x;
        age_   = 0;
        valid_ = true;
    }
    // Advance one fused frame without a valid current gate; expire past window.
    void tick_invalid() {
        if (!valid_) return;
        if (++age_ > kMaxHeldFrames) valid_ = false;
    }
    bool  usable() const { return valid_ && near_width_px_ > 1.0f; }
    int   age() const { return age_; }
    float near_width_px() const { return near_width_px_; }
    float near_center_x() const { return near_center_x_; }
    void  reset() {
        near_width_px_ = 0.0f; near_center_x_ = 0.0f; age_ = 0; valid_ = false;
    }

private:
    float near_width_px_{0.0f};
    float near_center_x_{0.0f};
    int   age_{0};
    bool  valid_{false};
};

std::optional<Detection> pick_front_object(const std::vector<Detection>& dets,
                                           const LaneGeometry& lane_geom,
                                           int frame_w, int frame_h,
                                           FrontObjectNearLaneMemory& near_mem,
                                           FrameId debug_frame_id = 0);
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
