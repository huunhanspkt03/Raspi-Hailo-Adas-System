#pragma once


#include <string>

#include <opencv2/core.hpp>

#include "adas/types.hpp"

namespace adas {

struct OverlayConfig {
    bool show_hud{true};
    bool hailo_active{true};
    int  display_width{960};
    //   warning_hud  — translucent lower windshield/hood panel with zone icon +
    //                  STOP / CAUTION / MONITOR label and optional distance
    //                  for the current front object
    //   lane_keep    — lower-right panel with a straight / left / right
    //                  arrow icon driven by corridor curvature, with
    //                  steering as the fallback signal
    //   per-detection warning icons drawn next to bbox top-right when
    //                  the detection sits inside the corridor zones.
    // All icons are drawn with OpenCV primitives — no per-frame image
    // decode or PNG asset loading.
    bool show_warning_hud{true};
    bool show_lane_keep_icon{true};
};

// Draws lane / corridor / detections / HUD onto a copy of pkt.frame and
// returns the rendered image (BGR uint8, source-frame size).
cv::Mat render_overlay(const BehaviorPacket& pkt,
                       float display_fps,
                       const OverlayConfig& cfg);

// Resize for display, preserving aspect ratio.
cv::Mat resize_for_display(const cv::Mat& src, int target_w);

}  // namespace adas
