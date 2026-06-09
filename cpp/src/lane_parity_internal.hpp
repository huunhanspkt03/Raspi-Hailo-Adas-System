#pragma once

// Internal types shared between lane_parity_components.cpp and
// lane_parity_tracker.cpp. NOT part of the public adas/ API — both
// consumers live in this directory.

#include <vector>

#include <opencv2/core.hpp>

#include "adas/lane_parity.hpp"

namespace adas {
namespace lane_parity {

struct SideFit {
    LaneCoeffs               coeffs;
    std::vector<cv::Point2f> pts_img;
    float                    confidence{0.0f};
    int                      n_pixels{0};
    int                      status{0};  // 0=EMPTY, 1=WEAK, 2=OK
};

struct FitLaneBoundariesResult {
    SideFit left;
    SideFit right;
    int     side_split_x_used{0};
};

// Defined in lane_parity_components.cpp.
FitLaneBoundariesResult fit_lane_boundaries_impl(
    const cv::Mat& line_union, const cv::Mat& lane_area,
    int frame_h, int frame_w, const float* prev_split_x);

}  // namespace lane_parity
}  // namespace adas
