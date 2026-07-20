#include "adas/fusion.hpp"

#include <cstdlib>
#include <iostream>

#include "adas/behavior.hpp"
#include "adas/object_detection.hpp"

// Lane/object fusion for the current C++ milestone. The object branch is
// available when YOLO is enabled; tracker and behavior policy are still
// lightweight compared with the Python reference. The matching rule is
//   object.frame_id <= lane.frame_id  AND  (lane.frame_id - object.frame_id)
//                                          <= kStaleFrameTolerance
// — see src/pipeline/fusion_worker.py::emit_fused for the reference.

namespace adas {

namespace {
bool debug_fcws_gate() {
    static const bool enabled = std::getenv("ADAS_DEBUG_FCWS_GATE") != nullptr;
    return enabled;
}

void log_selected_front(FrameId frame_id, FrameId object_frame_id,
                        const std::optional<Detection>& front_object) {
    if (!debug_fcws_gate()) return;
    std::cerr << "[fcws_select] frame=" << static_cast<unsigned long long>(frame_id)
              << " object_frame=" << static_cast<unsigned long long>(object_frame_id);
    if (!front_object) {
        std::cerr << " selected=0\n";
        return;
    }
    const Detection& d = *front_object;
    const cv::Rect2f& bb = d.bbox_img;
    std::cerr << " selected=1"
              << " class=" << yolo26_class_name(d.class_id)
              << " tid=" << d.track_id
              << " bc=(" << (bb.x + bb.width * 0.5f) << ","
              << (bb.y + bb.height) << ")"
              << " x_m=" << d.x_m
              << " z_m=" << d.z_m
              << " display=" << d.display_distance_m
              << " dist_valid=" << (d.distance_display_valid ? 1 : 0)
              << "\n";
}
}  // namespace

BehaviorPacket fuse(const LanePacket& lane, const ObjectPacket* objects,
                    FrontObjectNearLaneMemory& near_mem) {
    BehaviorPacket bp;
    bp.frame_id = lane.frame_id;
    bp.timestamp = lane.timestamp;
    bp.frame = lane.source_frame;
    bp.lane = lane;
    if (objects) {
        ObjectPacket merged = *objects;  // single mutable copy (see fusion.hpp)
        // front_object is strict-lane-gate dependent, so resolve it here
        // where lane geometry is in scope. Mirrors the Python fusion_worker
        // flow. speed_object shares the SAME strict gate (the looser-gate
        // variant was removed), so it is the same selection — assigned once
        // instead of re-running the gate scan.
        int frame_w = lane.source_frame.cols;
        int frame_h = lane.source_frame.rows;
        merged.front_object = pick_front_object(merged.detections,
                                                lane.geometry,
                                                frame_w, frame_h,
                                                near_mem,
                                                lane.frame_id);
        merged.speed_object = merged.front_object;
        log_selected_front(lane.frame_id, merged.frame_id, merged.front_object);
        bp.objects = std::move(merged);
    }
    auto state = decide_behavior(bp.lane, bp.objects);
    bp.behavior_action = state.action;
    bp.behavior_reason = state.reason;
    bp.control.steering = lane.steering;
    bp.control.throttle = state.throttle;
    bp.control.rc_enabled = false;
    bp.control.safe = true;
    return bp;
}

// (fuse_object_driven was removed with the alternate scheduler: in the
// parallel schedule the lane worker drives every render packet, so the
// object worker only caches its packet for the lane-side fusion above.)

}  // namespace adas
