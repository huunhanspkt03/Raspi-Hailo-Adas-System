#pragma once

#include "adas/yolo26s_postprocess.hpp"

#include <vector>

namespace adas {

struct CameraIntrinsics {
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
};

struct FrontObject {
    bool valid = false;
    Detection det;
};

class ObjectLocalizer {
public:
    // Defaults: wide-angle camera prior, 1280x720, dashcam mount
    explicit ObjectLocalizer(float hfov_deg = 73.0f,
                             float camera_height_m = 1.0f,
                             float pitch_deg = 1.0f,
                             float center_gate_ratio = 0.18f,
                             float min_bottom_ratio = 0.45f);

    FrontObject localize(std::vector<Detection>& detections,
                         int frame_w,
                         int frame_h) const;

private:
    CameraIntrinsics makeApproxIntrinsics(int frame_w, int frame_h) const;
    bool estimatePose(Detection& det,
                      const CameraIntrinsics& intr,
                      int frame_w,
                      int frame_h) const;

    float hfov_deg_;
    float camera_height_m_;
    float pitch_rad_;
    float center_gate_ratio_;
    float min_bottom_ratio_;
};

}  // namespace adas
