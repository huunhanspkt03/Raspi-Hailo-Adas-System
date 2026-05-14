#pragma once

#include "adas/inference_backend.hpp"

#include <opencv2/opencv.hpp>

#include <array>
#include <memory>

namespace adas {

class LaneCenterEstimator;  // forward-declare to avoid include cycle

// Mirrors the Python lane class IDs in src/features/lane_detection.py.
constexpr int LANE_CLASS_BACKGROUND = 0;
constexpr int LANE_CLASS_DOTTED_Y   = 1;
constexpr int LANE_CLASS_DOTTED_W   = 2;
constexpr int LANE_CLASS_LANE_AREA  = 3;
constexpr int LANE_CLASS_SOLID_W    = 4;
constexpr int LANE_CLASS_SOLID_Y    = 5;
constexpr int LANE_NUM_CLASSES      = 6;

struct LaneResult {
    bool valid = false;
    cv::Mat class_mask;                                  // CV_8UC1, per-pixel class id
    cv::Mat lane_area_mask;                              // CV_8UC1 {0,1}, cleaned (drives centerline)
    std::array<cv::Mat, LANE_NUM_CLASSES> class_masks;   // CV_8UC1 {0,255}, indexed by class id
    std::vector<cv::Point> fitted_path;
    int control_x = -1;
    int control_y = -1;
    float confidence = 0.0f;
};

class DeeplabPostprocessor {
public:
    // Set ADAS_DEEPLAB_DEBUG=1 in the environment, or pass debug=true,
    // to log per-class pixel counts every 60 frames.
    explicit DeeplabPostprocessor(bool debug = false);
    ~DeeplabPostprocessor();
    DeeplabPostprocessor(const DeeplabPostprocessor&) = delete;
    DeeplabPostprocessor& operator=(const DeeplabPostprocessor&) = delete;

    LaneResult process(const InferenceOutput& infer_out, int frame_w, int frame_h);

private:
    bool debug_ = false;
    int  debug_frame_ = 0;
    bool logged_shape_ = false;
    std::unique_ptr<LaneCenterEstimator> estimator_;
};

}  // namespace adas
