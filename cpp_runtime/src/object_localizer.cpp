#include "adas/object_localizer.hpp"

#include <algorithm>
#include <cmath>

namespace adas {

namespace {

struct RefDims {
    float height_m;
    float width_m;
};

bool lookupRefDims(int class_id, RefDims& dims) {
    switch (class_id) {
        case 0:  dims = {3.2f, 2.5f}; return true;  // bus
        case 1:  dims = {1.5f, 1.8f}; return true;  // car
        case 2:  dims = {0.6f, 0.3f}; return true;  // green_light
        case 3:  dims = {1.0f, 0.8f}; return true;  // motorcycle
        case 4:  dims = {0.6f, 0.3f}; return true;  // red_light
        case 10: dims = {3.4f, 2.5f}; return true;  // truck
        default: return false;
    }
}

constexpr float kMinDepth = 0.3f;
constexpr float kMaxDepth = 80.0f;
constexpr float kTrafficLightMountHeightM = 5.5f;
constexpr float kTrafficLightMinDistanceM = 3.0f;
constexpr float kTrafficLightMaxDistanceM = 80.0f;
constexpr float kTrafficLightStableDyPx = 40.0f;

bool isTrafficLightClass(int class_id) {
    return class_id == 2 || class_id == 4;
}

}  // namespace

ObjectLocalizer::ObjectLocalizer(float hfov_deg,
                                 float camera_height_m,
                                 float pitch_deg,
                                 float center_gate_ratio,
                                 float min_bottom_ratio)
    : hfov_deg_(hfov_deg),
      camera_height_m_(camera_height_m),
      pitch_rad_(pitch_deg * static_cast<float>(CV_PI) / 180.0f),
      center_gate_ratio_(center_gate_ratio),
      min_bottom_ratio_(min_bottom_ratio) {}

CameraIntrinsics ObjectLocalizer::makeApproxIntrinsics(int frame_w, int frame_h) const {
    const float half_fov_rad = hfov_deg_ * 0.5f * static_cast<float>(CV_PI) / 180.0f;
    const float fx = 0.5f * static_cast<float>(frame_w) / std::tan(std::max(half_fov_rad, 1e-3f));
    CameraIntrinsics intr;
    intr.fx = fx;
    intr.fy = fx;
    intr.cx = 0.5f * static_cast<float>(frame_w - 1);
    intr.cy = 0.5f * static_cast<float>(frame_h - 1);
    return intr;
}

bool ObjectLocalizer::estimatePose(Detection& det,
                                   const CameraIntrinsics& intr,
                                   int frame_w,
                                   int frame_h) const {
    det.bottom_center_x = det.box.x + det.box.width / 2;
    det.bottom_center_y = det.box.y + det.box.height;
    det.pose_valid = false;
    det.in_path = false;
    det.distance_m = -1.0f;
    det.x_m = 0.0f;
    det.z_m = 0.0f;
    det.bearing_deg = 0.0f;

    RefDims dims{};
    if (!lookupRefDims(det.class_id, dims)) return false;
    if (det.box.width <= 1 || det.box.height <= 1) return false;

    const float bbox_h = static_cast<float>(det.box.height);
    const bool is_light = isTrafficLightClass(det.class_id);

    if (is_light) {
        const float center_u = static_cast<float>(det.box.x) +
                               static_cast<float>(det.box.width) * 0.5f;
        const float center_v = static_cast<float>(det.box.y) +
                               static_cast<float>(det.box.height) * 0.5f;
        const float dy = intr.cy - center_v;
        const float height_diff = kTrafficLightMountHeightM - camera_height_m_;
        if (dy <= 8.0f || height_diff <= 0.1f) return false;
        if (dy < kTrafficLightStableDyPx) return false;

        float z = height_diff * intr.fy / dy;
        if (z > kTrafficLightMaxDistanceM) return false;
        z = std::clamp(z, kTrafficLightMinDistanceM, kTrafficLightMaxDistanceM);
        if (!std::isfinite(z) || z <= 0.0f) return false;

        const float x = ((center_u - intr.cx) / std::max(intr.fx, 1e-3f)) * z;
        det.bottom_center_x = static_cast<int>(std::round(center_u));
        det.bottom_center_y = static_cast<int>(std::round(center_v));
        det.pose_valid = true;
        det.in_path = false;
        det.x_m = x;
        det.z_m = z;
        det.distance_m = std::sqrt(x * x + z * z);
        det.bearing_deg = std::atan2(x, z) * 180.0f / static_cast<float>(CV_PI);
        return true;
    }

    float v = static_cast<float>(det.bottom_center_y);
    // Vehicle-CV-ADAS-style single-camera distance:
    // z = real_object_height * focal_px / bbox_pixel_height.
    float z = dims.height_m * intr.fy / bbox_h;
    const float bottom_ratio = static_cast<float>(det.bottom_center_y) /
                               std::max(static_cast<float>(frame_h), 1.0f);
    if (bottom_ratio >= 0.78f) {
        const float t = std::clamp((bottom_ratio - 0.78f) / (0.95f - 0.78f), 0.0f, 1.0f);
        const float bottom_cap = 8.0f * (1.0f - t) + 0.6f * t;
        z = std::min(z, std::max(kMinDepth, bottom_cap));
    }
    z = std::clamp(z, kMinDepth, kMaxDepth);

    if (!std::isfinite(z) || z <= 0.0f) return false;

    const float box_left = static_cast<float>(det.box.x);
    const float box_right = static_cast<float>(det.box.x + det.box.width);
    float u = static_cast<float>(det.bottom_center_x);
    if (box_right < intr.cx) {
        u = box_right;
    } else if (box_left > intr.cx) {
        u = box_left;
    } else {
        u = intr.cx;
    }
    det.bottom_center_x = static_cast<int>(std::round(u));
    const float x = ((u - intr.cx) / std::max(intr.fx, 1e-3f)) * z;

    det.pose_valid = true;
    det.x_m = x;
    det.z_m = z;
    det.distance_m = std::sqrt(x * x + z * z);
    det.bearing_deg = std::atan2(x, z) * 180.0f / static_cast<float>(CV_PI);

    const float center_gate_px = center_gate_ratio_ * static_cast<float>(frame_w);
    const float min_bottom_y = min_bottom_ratio_ * static_cast<float>(frame_h);
    det.in_path = box_right >= intr.cx - center_gate_px &&
                  box_left <= intr.cx + center_gate_px &&
                  v >= min_bottom_y;
    return true;
}

FrontObject ObjectLocalizer::localize(std::vector<Detection>& detections,
                                      int frame_w,
                                      int frame_h) const {
    FrontObject front;
    const CameraIntrinsics intr = makeApproxIntrinsics(frame_w, frame_h);

    for (auto& det : detections) {
        if (!estimatePose(det, intr, frame_w, frame_h)) continue;
        if (!det.in_path) continue;
        if (!front.valid || det.z_m < front.det.z_m) {
            front.valid = true;
            front.det = det;
        }
    }
    return front;
}

}  // namespace adas
