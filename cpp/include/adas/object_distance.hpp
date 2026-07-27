#pragma once
#include <cstdint>
#include <vector>

#include "adas/types.hpp"

namespace adas {

// Camera defaults match the Pi Cam Module 3 Wide capture used for this
// project (rpicam-vid --mode 1536:864:10:P --width 1280 --height 864).
// Resolutions are *not* hard-coded — the focal length is derived from
// hfov_deg and the current frame_w at annotate time.
struct DistanceCameraConfig {
    float hfov_deg{97.0f};           // effective horizontal FOV after CSI crop
    float vfov_deg{-1.0f};           // <=0 means "use fx for fy"
    float camera_height_m{1.35f};    // used by off-axis ground-plane correction
    float pitch_deg{-1.5f};          // negative = camera pointed down
    float camera_to_front_bumper_m{0.20f};
    float vehicle_width_m{1.8f};
    float display_distance_offset_m{0.0f};
    float calib_foot_y{-1.0f};
    float calib_z_m{-1.0f};
};
void annotate_distances(std::vector<Detection>& dets,
                        int frame_w, int frame_h,
                        const DistanceCameraConfig& cfg,
                        std::int64_t frame_id = -1);

void apply_distance_calibration_env(DistanceCameraConfig& cfg);

float ground_plane_row_for_z(float z_cam, int frame_w, int frame_h,
                             const DistanceCameraConfig& cfg);

// Clear per-track temporal pose-smoothing state (e.g. on a video/stream
// switch) so distances do not carry over a cut. Pair with
// reset_object_tracker() in object_detection.hpp.
void reset_distance_smoother();

}  // namespace adas
