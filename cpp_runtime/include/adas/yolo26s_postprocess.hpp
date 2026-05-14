#pragma once

#include "adas/inference_backend.hpp"
#include "adas/preprocess.hpp"

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace adas {

// 11 classes — matches src/features/detection_filters.py
constexpr int YOLO26S_NUM_CLASSES = 11;
constexpr int YOLO26S_MAX_DET     = 300;

struct Detection {
    cv::Rect box;          // image-space bounding box
    float score = 0.0f;
    int class_id = -1;
    std::string label;
    int bottom_center_x = -1;
    int bottom_center_y = -1;
    bool pose_valid = false;
    bool in_path = false;
    float distance_m = -1.0f;
    float x_m = 0.0f;
    float z_m = 0.0f;
    float bearing_deg = 0.0f;
};

class Yolo26sPostprocessor {
public:
    // conf_thresh mirrors the Python default of 0.30.
    explicit Yolo26sPostprocessor(float conf_thresh = 0.30f,
                                  float nms_iou_thresh = 0.45f);

    std::vector<Detection> process(const InferenceOutput& infer_out,
                                   const LetterboxInfo& lb,
                                   int orig_w,
                                   int orig_h);

    static const std::string& className(int class_id);

private:
    std::vector<Detection> decodeEnd2End(const Tensor& t,
                                         const LetterboxInfo& lb,
                                         int orig_w,
                                         int orig_h) const;

    std::vector<Detection> decodeRawHead(const InferenceOutput& out,
                                         const LetterboxInfo& lb,
                                         int orig_w,
                                         int orig_h) const;

    float conf_thresh_;
    float nms_iou_thresh_;
};

}  // namespace adas
