#pragma once

// Threaded runtime: capture → lane | object → fusion → render/display.
//
// Mirrors src/pipeline/threaded_runtime.py shape so cross-language reviewers
// can map components quickly. The scheduler is ALWAYS full-mode parallel:
// every captured frame goes to the lane branch, and every Nth frame
// (object_every_n) additionally to the object branch — both models may
// infer concurrently on Hailo (the installed HailoRT scheduler arbitrates).
// The lane worker drives every render packet; fusion pairs each lane packet
// with the latest object packet whose frame_id <= lane.frame_id within
// kStaleFrameTolerance.

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <mutex>

#include "adas/behavior.hpp"
#include "adas/capture.hpp"
#include "adas/front_speed.hpp"
#include "adas/gps.hpp"
#include "adas/hailo_inference.hpp"
#include "adas/lane_parity.hpp"
#include "adas/latest_queue.hpp"
#include "adas/lane_geometry.hpp"
#include "adas/object_detection.hpp"
#include "adas/overlay.hpp"
#include "adas/rc_actuator.hpp"
#include "adas/stage_timer.hpp"
#include "adas/types.hpp"

namespace adas {

// This build supports exactly ONE pipeline configuration — the operator
// command used on the Pi:
//   --mode full --lane-mode python_parity --model-schedule parallel
//   [--object-every-n N]
// Lane-only / detection-only modes, the image lane-geometry path, and the
// alternate (odd/even) Hailo schedule were removed on request. The CLI
// still accepts the mode flags above (validation only); any other value is
// rejected with a clear error.
struct RuntimeConfig {
    CaptureConfig capture;
    std::string lane_hef{"src/models/stdc1_lane_640.hef"};
    std::string yolo_hef{"src/models/yolo26n_object.hef"};
    bool show_window{true};
    bool save_video{false};
    std::string save_video_path;
    OverlayConfig overlay;
    ObjectDecodeConfig object_decode;
    int object_every_n{2};           // throttle: run YOLO every Nth captured frame
    int display_every_n{1};          // render/show every Nth packet; save-video forces 1
    int profile_log_every{100};      // periodic render_fps/drops summary
    bool profile_enabled{false};     // --profile: per-stage EMA/max timing
    // --quiet: drop periodic / repeated stderr noise so realtime driving
    // sessions don't spam the operator terminal. Startup banner +
    // fatal errors + source-exhausted still print. UI overlays are
    // unaffected.
    bool quiet_logs{false};
    // --eval-log: print a copy-paste-friendly [eval] statistics block to
    // stdout at shutdown (frame counts, FPS, drops, GPS speed). Pipeline
    // behavior is unchanged; prints even under --quiet.
    bool eval_log{false};
    bool dry_run{false};             // run a few frames headlessly + exit
    int  dry_run_frames{30};
    // GPS (U-Blox NEO-M10 over UART). Disabled by default. Runs in its
    // own thread, never used for steering — speed/position/logging only.
    GpsConfig gps;
    // RC actuator (servo + BTS7960 forward-only). Disabled by default;
    // with --rc-enable, motor duty follows behavior/safety commands.
    RcConfig rc;
};

class Runtime {
public:
    explicit Runtime(RuntimeConfig cfg);
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    int run();
    void request_stop();

private:
    void capture_loop();
    void lane_loop();
    void object_loop();
    void render_loop();
    // Emit the [eval] statistics block to stdout (called from run() after
    // all worker threads have joined). No-op behavior change; --eval-log only.
    void print_eval_summary(double duration_s) const;

    // Most recent object packet keyed by lane.frame_id via fusion rule
    // object.frame_id <= lane.frame_id (within tolerance). Held as a
    // shared_ptr<const>: the object worker publishes one immutable packet
    // every Nth frame and the lane worker reads it every frame, so sharing
    // by refcount avoids a deep copy of the detections vector per lane frame.
    std::shared_ptr<const ObjectPacket> latest_object_locked() const;
    void store_object_locked(ObjectPacket pkt);

    RuntimeConfig cfg_;
    std::unique_ptr<Capture> cam_;
    std::unique_ptr<HailoInference> infer_;
    std::atomic<bool> stop_{false};

    LatestQueue<FramePacket> lane_q_{1};
    LatestQueue<FramePacket> object_q_{1};
    LatestQueue<BehaviorPacket> render_q_{1};
    LatestQueue<cv::Mat> display_q_{1};

    mutable std::mutex object_mu_;
    std::shared_ptr<const ObjectPacket> latest_object_;

    std::vector<std::thread> threads_;
    FrameId last_frame_id_{0};

    std::atomic<std::uint64_t> rendered_{0};
    std::atomic<std::uint64_t> lane_processed_{0};
    std::atomic<std::uint64_t> object_processed_{0};
    std::atomic<std::uint64_t> capture_count_{0};
    // Lane frames where an object packet existed but was outside the fusion
    // stale tolerance (counted for --eval-log; one relaxed atomic add).
    std::atomic<std::uint64_t> fusion_stale_{0};

    // --eval-log accumulators. Touched ONLY by the render thread; read in
    // run() after join(), so the join barrier makes them safe without locks.
    double        eval_render_fps_min_{0.0};   // min positive per-second render FPS
    bool          eval_render_fps_seen_{false};
    double        eval_gps_sum_kmh_{0.0};
    double        eval_gps_min_kmh_{0.0};
    double        eval_gps_max_kmh_{0.0};
    std::uint64_t eval_gps_count_{0};

    StageTimer stage_timer_;  // disabled unless cfg_.profile_enabled

    // Python-parity tracker. Owned via raw pointer because the type is
    // forward-declared in lane_parity.hpp (impl lives in the .cpp).
    // Constructed in run() — python_parity is the only lane path.
    LaneBoundaryTracker* parity_tracker_{nullptr};

    // Optional GPS reader. Constructed only when cfg_.gps.enabled.
    std::unique_ptr<GpsReader> gps_;
    // Optional RC actuator. Constructed only when cfg_.rc.enabled.
    std::unique_ptr<RcActuator> rc_;

    // Behavior smoother + fail-closed safety supervisor. Used only by the
    // render loop (single-threaded) just before issuing RC commands.
    SafetyGovernor safety_governor_;

    // Display-only speed-limit sign log: holds the last logged label so
    // repeated frames of the same sign don't spam stderr. Set by
    // object_loop. Empty string means "no sign currently visible". Signs
    // never gate motor/RC — see CLAUDE.md and Step 5 of the sign change.
    std::string last_logged_sign_;

    // Front-vehicle relative-speed estimator. Owned by Runtime, called
    // exclusively from render_loop (single-threaded) so it needs no
    // internal locking. Annotates bp.objects->front_object only;
    // display/log only.
    FrontSpeedEstimator front_speed_;

    // Authoritative longitudinal throttle controller (TTC-primary, distance
    // fallback, near hard stop). Render-loop only; runs after front_speed_
    // (needs ttc_*) and before safety_governor_ (which stays the final
    // authority). Overrides the front-object throttle from decide_behavior.
    FollowController follow_controller_;
};

}  // namespace adas
