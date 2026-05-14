#include "adas/visualization.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace adas {

namespace {

// BGR colors per lane class id. Index matches LANE_CLASS_* constants.
// Tuned so all four line classes are visually distinct on road footage.
const cv::Scalar kLaneColors[LANE_NUM_CLASSES] = {
    cv::Scalar(  0,   0,   0),  // 0 background — unused
    cv::Scalar(  0, 165, 255),  // 1 dotted_y   — orange
    cv::Scalar(255, 255,   0),  // 2 dotted_w   — cyan
    cv::Scalar(  0, 180,   0),  // 3 lane_area  — green (translucent fill)
    cv::Scalar(255, 255, 255),  // 4 solid_w    — white
    cv::Scalar(  0, 255, 255),  // 5 solid_y    — yellow
};

// Mirror the Python render-visibility thresholds in
// src/features/lane_safety.py. Below kLaneHideConf the overlay is
// entirely suppressed so a decayed tracker state is not drawn as a
// confident lane; between kLaneHideConf and kLaneWeakConf the fill
// alpha and line thickness taper linearly. These literals MUST stay
// equal to RENDER_HIDE_CONFIDENCE / RENDER_WEAK_CONFIDENCE.
constexpr float kLaneHideConf = 0.18f;
constexpr float kLaneWeakConf = 0.35f;
constexpr float kLaneFillAlphaFull = 0.35f;

const cv::Mat& markingDilateKernel() {
    static const cv::Mat kernel =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    return kernel;
}

float laneAlphaScale(float confidence) {
    const float c = std::clamp(confidence, 0.0f, 1.0f);
    if (c < kLaneHideConf) return 0.0f;
    if (c >= kLaneWeakConf) return 1.0f;
    const float span = kLaneWeakConf - kLaneHideConf;
    return (c - kLaneHideConf) / std::max(span, 1e-3f);
}

}  // namespace

void Visualizer::drawLane(cv::Mat& frame_bgr, const LaneResult& lane) {
    const float alpha_scale = laneAlphaScale(lane.confidence);
    const bool hide_fill = alpha_scale <= 0.0f;

    // 1) Translucent fill for the drivable lane area. Dimmed by
    //    alpha_scale so a low-confidence frame reads as tentative.
    if (!hide_fill && !lane.lane_area_mask.empty()) {
        cv::Mat overlay = frame_bgr.clone();
        overlay.setTo(kLaneColors[LANE_CLASS_LANE_AREA], lane.lane_area_mask);
        const double fill_alpha = kLaneFillAlphaFull * alpha_scale;
        cv::addWeighted(overlay, fill_alpha, frame_bgr, 1.0 - fill_alpha, 0, frame_bgr);
    }

    // 2) Direct overwrite for thin line classes — single pass, no blend,
    //    no extra clones. Line classes are segmentation evidence, so they
    //    render at any confidence (Python keeps markings on hide_fill too).
    for (int id = 0; id < LANE_NUM_CLASSES; ++id) {
        if (id == LANE_CLASS_BACKGROUND || id == LANE_CLASS_LANE_AREA) continue;
        const cv::Mat& m = lane.class_masks[id];
        if (m.empty()) continue;
        cv::Mat thick_mask;
        cv::dilate(m, thick_mask, markingDilateKernel(), cv::Point(-1, -1), 1);
        frame_bgr.setTo(kLaneColors[id], thick_mask);
    }

    if (hide_fill) return;  // nothing else should paint a confident lane

    // 3) Centerline + control point, thickness scaled by confidence.
    const int center_th  = std::max(1, static_cast<int>(std::round(3.0f * alpha_scale)));
    const int circle_r   = std::max(3, static_cast<int>(std::round(8.0f * alpha_scale)));
    const int control_th = std::max(1, static_cast<int>(std::round(2.0f * alpha_scale)));
    if (!lane.fitted_path.empty()) {
        for (size_t i = 1; i < lane.fitted_path.size(); ++i) {
            cv::line(frame_bgr, lane.fitted_path[i - 1], lane.fitted_path[i],
                     cv::Scalar(255, 255, 255), center_th + 2);
            cv::line(frame_bgr, lane.fitted_path[i - 1], lane.fitted_path[i],
                     cv::Scalar(0, 255, 255), center_th);
        }
    }
    if (lane.valid && lane.control_x >= 0) {
        cv::circle(frame_bgr, cv::Point(lane.control_x, lane.control_y),
                   circle_r + 2, cv::Scalar(255, 255, 255), -1);
        cv::circle(frame_bgr, cv::Point(lane.control_x, lane.control_y),
                   circle_r, cv::Scalar(0, 0, 255), -1);
        cv::line(frame_bgr,
                 cv::Point(frame_bgr.cols / 2, frame_bgr.rows),
                 cv::Point(lane.control_x, lane.control_y),
                 cv::Scalar(255, 255, 255), control_th + 2);
        cv::line(frame_bgr,
                 cv::Point(frame_bgr.cols / 2, frame_bgr.rows),
                 cv::Point(lane.control_x, lane.control_y),
                 cv::Scalar(255, 0, 0), control_th);
    }
}

void Visualizer::drawDetections(cv::Mat& frame_bgr, const std::vector<Detection>& dets) {
    for (const auto& d : dets) {
        const cv::Scalar color = d.in_path ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 255, 0);
        cv::rectangle(frame_bgr, d.box, color, 2);
        if (d.bottom_center_x >= 0 && d.bottom_center_y >= 0) {
            cv::circle(frame_bgr, cv::Point(d.bottom_center_x, d.bottom_center_y),
                       4, color, cv::FILLED);
        }
        char buf[128];
        if (d.pose_valid) {
            std::snprintf(buf, sizeof(buf), "%s %.2f %.1fm",
                          d.label.c_str(), d.score, d.distance_m);
        } else {
            std::snprintf(buf, sizeof(buf), "%s %.2f", d.label.c_str(), d.score);
        }
        int base = 0;
        cv::Size ts = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
        cv::rectangle(frame_bgr,
                      cv::Point(d.box.x, d.box.y - ts.height - 4),
                      cv::Point(d.box.x + ts.width + 4, d.box.y),
                      color, cv::FILLED);
        cv::putText(frame_bgr, buf, cv::Point(d.box.x + 2, d.box.y - 3),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

void Visualizer::drawHud(cv::Mat& frame_bgr,
                         double fps,
                         double infer_ms,
                         const SteeringOutput& steer,
                         const char* mode,
                         const FrontObject& front) {
    char line[192];
    std::snprintf(line, sizeof(line),
                  "mode=%s  fps=%.1f  infer=%.1fms  steer=%+.2f  err=%dpx",
                  mode, fps, infer_ms, steer.smoothed, steer.error_px);
    cv::rectangle(frame_bgr, cv::Point(0, 0),
                  cv::Point(frame_bgr.cols, 48), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(frame_bgr, line, cv::Point(8, 18),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

    char line2[192];
    if (front.valid) {
        std::snprintf(line2, sizeof(line2),
                      "front=%s  z=%.1fm  x=%+.1fm  bearing=%+.1fdeg",
                      front.det.label.c_str(),
                      front.det.z_m,
                      front.det.x_m,
                      front.det.bearing_deg);
    } else {
        std::snprintf(line2, sizeof(line2),
                      "front=none  heuristic=center-band until deeplab lane gating is available");
    }
    cv::putText(frame_bgr, line2, cv::Point(8, 40),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(180, 255, 255), 1);
}

}  // namespace adas
