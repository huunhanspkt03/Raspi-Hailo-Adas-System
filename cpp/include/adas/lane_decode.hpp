#pragma once
#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

#include "adas/hailo_inference.hpp"
#include "adas/types.hpp"   // lane_class constants live here

namespace adas {

constexpr int kStdc1InputSize = 640;

struct LaneDecodeResult {
    cv::Mat lane_area;   // HxW uint8 (0 or 1)
    cv::Mat line_union;  // HxW uint8 (0 or 1): dotted | solid
};
void preprocess_lane_semantic(const cv::Mat& frame_bgr, int size,
                              std::vector<std::uint8_t>* out);
                              
bool decode_stdc1(const InferOutputs& raw, int frame_h, int frame_w,
                  LaneDecodeResult* out);

}  // namespace adas
