#include "adas/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "adas/fusion.hpp"
#include "adas/lane_decode.hpp"
#include "adas/lane_geometry.hpp"
#include "adas/object_detection.hpp"
#include "adas/object_distance.hpp"
#include "adas/overlay.hpp"

namespace adas {

Runtime::Runtime(RuntimeConfig cfg)
    : cfg_(std::move(cfg)),
      stage_timer_(cfg_.profile_enabled) {}
Runtime::~Runtime() {
    request_stop();
    for (auto& t : threads_) if (t.joinable()) t.join();
    // Stop RC before GPS — guarantees motor=stop / servo=neutral hits
    // the wire before anything else tears down (incl. on signal exit).
    if (rc_) rc_->stop();
    rc_.reset();
    if (gps_) gps_->stop();
    gps_.reset();
    destroy_lane_boundary_tracker(parity_tracker_);
    parity_tracker_ = nullptr;
}

void Runtime::request_stop() {
    stop_.store(true);
    lane_q_.close();
    object_q_.close();
    render_q_.close();
    display_q_.close();
}

std::shared_ptr<const ObjectPacket> Runtime::latest_object_locked() const {
    std::lock_guard<std::mutex> lock(object_mu_);
    return latest_object_;  // shared_ptr copy — refcount bump, no deep copy
}

void Runtime::store_object_locked(ObjectPacket pkt) {
    auto shared = std::make_shared<const ObjectPacket>(std::move(pkt));
    std::lock_guard<std::mutex> lock(object_mu_);
    latest_object_ = std::move(shared);  // build off-lock; only swap under it
}

int Runtime::run() {
    // Mirror the --quiet gate into the lane_parity TU so its periodic /
    // repeated diagnostics fold under the same flag. Fatal first-time
    // misses + the startup banner are unaffected.
    set_lane_parity_quiet_logs(cfg_.quiet_logs);
    cam_ = std::make_unique<Capture>(cfg_.capture);
    if (!cam_->open()) {
        std::cerr << "[runtime] capture failed to open\n";
        return 2;
    }
    infer_ = std::make_unique<HailoInference>();
    if (!infer_->load("lane_segmentation", cfg_.lane_hef)) {
        std::cerr << "[runtime] failed to load lane HEF: " << cfg_.lane_hef << "\n";
        return 3;
    }
    if (!infer_->load("yolo26", cfg_.yolo_hef)) {
        std::cerr << "[runtime] failed to load YOLO HEF: " << cfg_.yolo_hef << "\n";
        return 4;
    }

    // Fusion only pairs object packets within kStaleFrameTolerance lane
    // frames; a larger stride would make every object packet read as stale
    // (no front object -> SafetyGovernor object_stale stop). Clamp loudly.
    if (cfg_.object_every_n > kStaleFrameTolerance) {
        std::cerr << "[runtime] object_every_n=" << cfg_.object_every_n
                  << " exceeds fusion stale tolerance (" << kStaleFrameTolerance
                  << "); clamping to " << kStaleFrameTolerance << "\n";
        cfg_.object_every_n = kStaleFrameTolerance;
    }
    parity_tracker_ = make_lane_boundary_tracker();
    std::cerr << "[runtime] mode=full lane_mode=python_parity "
              << "model_schedule=parallel object_every_n=" << cfg_.object_every_n
              << " conf=" << cfg_.object_decode.conf_thresh << "\n";
    if (cfg_.display_every_n > 1 && !cfg_.save_video) {
        std::cerr << "[runtime] display_every_n=" << cfg_.display_every_n
                  << " (render decimation enabled)\n";
    }

    if (cfg_.gps.enabled) {
        gps_ = std::make_unique<GpsReader>();
        if (gps_->start(cfg_.gps)) {
            std::cerr << "[runtime] gps enabled port=" << cfg_.gps.port
                      << " baud=" << cfg_.gps.baud
                      << " stale_timeout=" << cfg_.gps.stale_timeout_s << "s\n";
        } else {
            std::cerr << "[runtime] gps requested but reader did not start "
                      << "(unsupported OS or disabled at runtime). "
                      << "Continuing without GPS.\n";
        }
    }
    if (cfg_.rc.enabled) {
        rc_ = std::make_unique<RcActuator>();
        if (!rc_->start(cfg_.rc)) {
            std::cerr << "[runtime] rc enable requested but actuator refused "
                      << "to arm. Continuing without RC.\n";
            rc_.reset();
        }
    }

    threads_.emplace_back(&Runtime::capture_loop, this);
    threads_.emplace_back(&Runtime::lane_loop, this);
    threads_.emplace_back(&Runtime::object_loop, this);
    threads_.emplace_back(&Runtime::render_loop, this);

    if (cfg_.show_window && !cfg_.dry_run) {
        cv::namedWindow("adas_cpp", cv::WINDOW_NORMAL);
        cv::resizeWindow("adas_cpp", cfg_.overlay.display_width,
                         cfg_.overlay.display_width * 9 / 16);
    }

    // Main thread either pumps cv::waitKey for display or just idles until stop.
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    while (!stop_.load()) {
        if (cfg_.show_window && !cfg_.dry_run) {
            std::optional<cv::Mat> display_opt;
            {
                ScopedStage st(stage_timer_, "display_wait");
                display_opt = display_q_.get(std::chrono::milliseconds(1));
            }
            if (display_opt) {
                ScopedStage st(stage_timer_, "display_show");
                cv::imshow("adas_cpp", *display_opt);
            }
            int k;
            {
                ScopedStage st(stage_timer_, "display_waitkey");
                k = cv::waitKey(1) & 0xFF;
            }
            if (k == 'q' || k == 27) { request_stop(); break; }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        const std::uint64_t done = rendered_.load();
        if (cfg_.dry_run && done >= static_cast<std::uint64_t>(cfg_.dry_run_frames)) {
            std::cerr << "[runtime] dry-run target reached (" << cfg_.dry_run_frames << ")\n";
            request_stop();
            break;
        }
    }

    for (auto& t : threads_) if (t.joinable()) t.join();
    if (cfg_.show_window) cv::destroyAllWindows();
    auto dt = std::chrono::duration<double>(clock::now() - t0).count();
    if (!cfg_.quiet_logs) {
        const auto fps = [dt](std::uint64_t count) {
            return dt > 0.0 ? static_cast<double>(count) / dt : 0.0;
        };
        const double capture_fps = fps(capture_count_.load());
        const double lane_fps = fps(lane_processed_.load());
        const double object_fps = fps(object_processed_.load());
        const double render_fps = fps(rendered_.load());
        std::cerr << "[runtime] stop | captured=" << capture_count_
                  << " lane=" << lane_processed_
                  << " object=" << object_processed_
                  << " rendered=" << rendered_
                  << " elapsed=" << dt << "s"
                  << " | fps capture=" << capture_fps
                  << " lane=" << lane_fps
                  << " object=" << object_fps
                  << " render=" << render_fps
                  << " | lane_dropped=" << lane_q_.dropped()
                  << " object_dropped=" << object_q_.dropped()
                  << " render_dropped=" << render_q_.dropped() << "\n";
        if (cfg_.profile_enabled) {
            std::cerr << "[runtime] final " << stage_timer_.format_summary() << "\n";
        }
    }
    // Evaluation summary — prints regardless of --quiet (it IS the requested
    // output). Threads are joined above, so reading the eval_* members and
    // counters here is race-free.
    if (cfg_.eval_log) {
        print_eval_summary(dt);
    }
    return 0;
}

void Runtime::print_eval_summary(double duration_s) const {
    const auto rate = [duration_s](std::uint64_t n) {
        return duration_s > 0.0 ? static_cast<double>(n) / duration_s : 0.0;
    };
    std::string src;
    switch (cfg_.capture.source) {
        case CaptureSource::VideoFile: src = cfg_.capture.video_path; break;
        case CaptureSource::CsiCamera: src = "live:csi"; break;
        case CaptureSource::UsbCamera: src = "live:usb"; break;
    }

    std::ostringstream os;
    os.setf(std::ios::fixed);
    os << "[eval] source=" << src << "\n";
    os << std::setprecision(2);
    os << "[eval] duration_sec=" << duration_s << "\n";
    os << "[eval] captured_frames=" << capture_count_.load() << "\n";
    os << "[eval] rendered_frames=" << rendered_.load() << "\n";
    os << "[eval] lane_inference_frames=" << lane_processed_.load() << "\n";
    os << "[eval] object_inference_frames=" << object_processed_.load() << "\n";
    os << "[eval] object_every_n_frames=" << cfg_.object_every_n << "\n";
    os << "[eval] dropped_frames="
       << (lane_q_.dropped() + object_q_.dropped() + render_q_.dropped()) << "\n";
    os << "[eval] dropped_lane=" << lane_q_.dropped()
       << " dropped_object=" << object_q_.dropped()
       << " dropped_render=" << render_q_.dropped() << "\n";
    os << "[eval] stale_packets=" << fusion_stale_.load() << "\n";
    os << "[eval] render_fps_avg=" << rate(rendered_.load()) << "\n";
    if (eval_render_fps_seen_)
        os << "[eval] render_fps_min=" << eval_render_fps_min_ << "\n";
    else
        os << "[eval] render_fps_min=n/a\n";   // ran < 1 s, no per-second sample
    os << "[eval] lane_fps_avg=" << rate(lane_processed_.load()) << "\n";
    os << "[eval] object_fps_avg=" << rate(object_processed_.load()) << "\n";

    // Best-effort video metadata (GStreamer decode often reports n/a).
    const double vframes = cam_ ? cam_->source_total_frames() : -1.0;
    const double vfps    = cam_ ? cam_->source_fps() : -1.0;
    if (vframes > 0.0)
        os << "[eval] video_total_frames=" << static_cast<long long>(vframes) << "\n";
    else
        os << "[eval] video_total_frames=n/a\n";
    os << "[eval] video_fps=" << vfps << "\n";

    // GPS speed (km/h) over valid fixes only.
    if (cfg_.gps.enabled && eval_gps_count_ > 0) {
        os << "[eval] gps_speed_avg_kmh="
           << (eval_gps_sum_kmh_ / static_cast<double>(eval_gps_count_)) << "\n";
        os << "[eval] gps_speed_min_kmh=" << eval_gps_min_kmh_ << "\n";
        os << "[eval] gps_speed_max_kmh=" << eval_gps_max_kmh_ << "\n";
        os << "[eval] gps_valid_samples=" << eval_gps_count_ << "\n";
    } else {
        os << "[eval] gps_speed_avg_kmh=n/a\n";
        os << "[eval] gps_speed_min_kmh=n/a\n";
        os << "[eval] gps_speed_max_kmh=n/a\n";
    }

    std::cout << os.str();
    std::cout.flush();
}

void Runtime::capture_loop() {
    // Replay backpressure: for file-backed sources, the decoder is faster
    // than the lane worker, so without throttling the capture thread will
    // race through thousands of frames into the drop-oldest queue (and the
    // OpenCV decode allocates a fresh BGR buffer every read). Live camera
    // sources keep their original behavior — drop is the right policy when
    // the producer is real-time.
    const bool is_video_file = cfg_.capture.source == CaptureSource::VideoFile;
    while (!stop_.load()) {
        if (is_video_file) {
            while (!stop_.load()) {
                // Lane is the per-frame consumer; pace the decoder on it.
                // The object branch is throttled by N and its drop-oldest
                // queue is the right policy when it lags.
                if (lane_q_.size() == 0) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (stop_.load()) break;
        }
        std::optional<cv::Mat> opt;
        {
            ScopedStage st(stage_timer_, "capture");
            opt = cam_->read_frame();
        }
        if (!opt) {
            if (cam_->is_exhausted()) {
                std::cerr << "[capture] source exhausted\n";
                request_stop();
                return;
            }
            continue;
        }
        FramePacket pkt;
        pkt.frame_id = ++last_frame_id_;
        pkt.timestamp = adas::now();
        pkt.frame = *opt;   // shallow ref
        pkt.frame_h = pkt.frame.rows;
        pkt.frame_w = pkt.frame.cols;
        // Parallel scheduler: lane every frame, object every Nth. Both
        // models may infer concurrently on Hailo; the lane worker drives
        // every render packet using the freshest cached object packet.
        const int every = cfg_.object_every_n > 0 ? cfg_.object_every_n : 1;
        if ((pkt.frame_id % static_cast<FrameId>(every)) == 0) {
            FramePacket obj_pkt = pkt;  // shallow refcount on cv::Mat
            object_q_.put(std::move(obj_pkt));
        }
        lane_q_.put(std::move(pkt));
        capture_count_.fetch_add(1, std::memory_order_relaxed);
    }
}

void Runtime::lane_loop() {
    // Hoisted across iterations so preprocess_lane_semantic can reuse the
    // capacity (no per-frame ~1.2 MB allocation for STDC1 at 640x640x3).
    // `raw` holds zero-copy views into the backend's persistent output
    // storage (see InferTensor's lifetime contract); the views are consumed
    // by decode_stdc1 before the next infer() of the same model.
    std::vector<std::uint8_t> rgb;
    InferOutputs raw;
    while (!stop_.load()) {
        auto opt = lane_q_.get(std::chrono::milliseconds(100));
        if (!opt) continue;
        ScopedStage total(stage_timer_, "lane_loop_total");
        const FramePacket& f = *opt;

        {
            ScopedStage st(stage_timer_, "preprocess_lane");
            preprocess_lane_semantic(f.frame, kStdc1InputSize, &rgb);
        }
        bool ok;
        {
            ScopedStage st(stage_timer_, "infer_lane");
            ok = infer_->infer("lane_segmentation", rgb, &raw);
        }
        LaneDecodeResult decoded;
        bool decode_ok;
        {
            ScopedStage st(stage_timer_, "decode_stdc1");
            decode_ok = ok && decode_stdc1(raw, f.frame_h, f.frame_w, &decoded);
        }

        LanePacket lp;
        lp.frame_id = f.frame_id;
        lp.timestamp = f.timestamp;
        lp.source_frame = f.frame;
        if (decode_ok) {
            lp.lane_area = decoded.lane_area;
            lp.line_union = decoded.line_union;
            {
                ScopedStage st(stage_timer_, "lane_geometry_parity");
                lp.geometry = compute_lane_geometry_python_parity(
                    parity_tracker_,
                    decoded.lane_area, decoded.line_union);
            }
            // Raw lookahead steering, then temporal stabilization so the servo
            // follows the same smoothed signal as the rendered red path (the
            // parity tracker already smooths the displayed centerline). Hold the
            // last valid command when the lane is unstable / fixed-guidance
            // fallback (geometry.valid==false -> 0 points -> hold).
            const float raw_steer =
                compute_steering(lp.geometry, f.frame_w, f.frame_h);
            const int valid_pts = lp.geometry.valid
                ? static_cast<int>(lp.geometry.fitted_pts_img.size()) : 0;
            lp.steering = steering_stabilizer_.update(
                raw_steer, valid_pts, lp.geometry.confidence);
        } else {
            // Full decode failure has no geometry to update from; keep the
            // last stabilized command instead of snapping the servo to neutral.
            lp.steering = steering_stabilizer_.value();
        }
        lane_processed_.fetch_add(1, std::memory_order_relaxed);
        // Fusion: pair the lane packet with the most recent object packet
        // whose frame_id <= lp.frame_id within kStaleFrameTolerance. The
        // cached packet is shared by refcount; fuse() makes the single
        // mutable copy it needs to stamp front_object/speed_object.
        BehaviorPacket bp;
        {
            ScopedStage st(stage_timer_, "fusion");
            std::shared_ptr<const ObjectPacket> latest = latest_object_locked();
            const ObjectPacket* obj_pair = nullptr;
            if (latest && latest->frame_id <= lp.frame_id &&
                (lp.frame_id - latest->frame_id) <= static_cast<FrameId>(kStaleFrameTolerance)) {
                obj_pair = latest.get();
            } else if (latest) {
                // An object packet existed but fell outside the stale window
                // (not warmup, which is latest == null) — count for --eval-log.
                fusion_stale_.fetch_add(1, std::memory_order_relaxed);
            }
            bp = fuse(lp, obj_pair, front_near_lane_mem_);
        }
        render_q_.put(std::move(bp));
        stage_timer_.frame_end();  // frame counter = lane-processed frames
    }
}

void Runtime::object_loop() {
    std::vector<std::uint8_t> rgb;
    cv::Mat resize_scratch;
    InferOutputs raw;  // zero-copy views; consumed before the next infer()
    while (!stop_.load()) {
        auto opt = object_q_.get(std::chrono::milliseconds(100));
        if (!opt) continue;
        ScopedStage total(stage_timer_, "object_loop_total");
        const FramePacket& f = *opt;

        {
            ScopedStage st(stage_timer_, "preprocess_object");
            preprocess_yolo26(f.frame, cfg_.object_decode.yolo_input_size,
                              static_cast<std::uint8_t>(cfg_.object_decode.pad_value),
                              &rgb, &resize_scratch);
        }

        bool ok;
        {
            ScopedStage st(stage_timer_, "infer_object");
            ok = infer_->infer("yolo26", rgb, &raw);
        }

        std::vector<Detection> dets;
        if (ok) {
            ScopedStage st(stage_timer_, "decode_object");
            dets = decode_yolo26(raw, f.frame_w, f.frame_h, cfg_.object_decode);
            dets = update_tracker(std::move(dets));
            // Monocular distance annotation runs before fusion so
            // pick_front_object() and the overlay can read pose_valid /
            // distance_m straight off the Detection.
            DistanceCameraConfig dcfg;  // Pi Cam Module 3 Wide defaults
            // ADAS_CALIB_FOOT_Y / ADAS_CALIB_Z_M override the ground-plane
            // calibration at runtime (parsed once, cached) — no rebuild needed.
            apply_distance_calibration_env(dcfg);
            annotate_distances(dets, f.frame_w, f.frame_h, dcfg,
                               static_cast<std::int64_t>(f.frame_id));

            // Log-only speed-limit sign reporting. Behavior/motor policy
            // ignores signs (pick_front_object filters yolo26_is_vehicle);
            // this is purely a stderr trace so the operator can see what
            // the model is reading. Logs only on a change of active sign,
            // so it folds nicely under --quiet too.
            if (!cfg_.quiet_logs) {
                const Detection* best_sign = nullptr;
                for (const auto& d : dets) {
                    if (!yolo26_is_traffic_sign(d.class_id)) continue;
                    if (best_sign == nullptr || d.score > best_sign->score) {
                        best_sign = &d;
                    }
                }
                std::string cur = best_sign ? best_sign->label : "";
                if (cur != last_logged_sign_) {
                    if (cur.empty()) {
                        std::cerr << "[sign] cleared (was "
                                  << last_logged_sign_ << ")\n";
                    } else {
                        std::cerr << "[sign] " << cur
                                  << " (" << yolo26_speed_limit_kmh(best_sign->class_id)
                                  << " km/h) — display/log only\n";
                    }
                    last_logged_sign_ = std::move(cur);
                }
            }
        }

        ObjectPacket op;
        op.frame_id = f.frame_id;
        op.timestamp = f.timestamp;
        op.detections = std::move(dets);

        // Parallel: the lane worker drives every render packet; just cache
        // so its fusion step can pair with the freshest object packet.
        store_object_locked(std::move(op));
        object_processed_.fetch_add(1, std::memory_order_relaxed);
    }
}

void Runtime::render_loop() {
    cv::VideoWriter writer;
    auto fps_t0 = std::chrono::steady_clock::now();
    std::uint64_t fps_count = 0;
    std::uint64_t packet_count = 0;
    float display_fps = 0.0f;
    while (!stop_.load()) {
        auto opt = render_q_.get(std::chrono::milliseconds(100));
        if (!opt) continue;
        BehaviorPacket bp = std::move(*opt);
        bp.control.rc_enabled = rc_ != nullptr;
        // Display-only GPS readout. Snapshot via latest() (non-blocking)
        // before front_speed_ so the fused front vehicle can be estimated
        // as ego_gps_speed + relative_speed when GPS has a valid fix.
        if (gps_) {
            const GpsSample g = gps_->latest();
            GpsHud h;
            h.valid = g.valid;
            h.speed_kmh = g.speed_kmh;
            h.satellites = g.satellites;
            h.age_s = g.age_s;
            h.source = g.source;
            bp.gps = h;
            if (cfg_.eval_log && h.valid) {
                if (eval_gps_count_ == 0) {
                    eval_gps_min_kmh_ = eval_gps_max_kmh_ = h.speed_kmh;
                } else {
                    eval_gps_min_kmh_ = std::min(eval_gps_min_kmh_, h.speed_kmh);
                    eval_gps_max_kmh_ = std::max(eval_gps_max_kmh_, h.speed_kmh);
                }
                eval_gps_sum_kmh_ += h.speed_kmh;
                ++eval_gps_count_;
            }
        }
        // Annotate the fused front_object with relative speed and, when GPS
        // is valid, estimated absolute front-vehicle speed BEFORE overlay.
        // The estimator is display/log only and never changes bp.control.*.
        front_speed_.update(bp);
        // Longitudinal throttle: authoritative TTC-primary controller for the
        // in-corridor front object. Runs AFTER front_speed_ (so ttc_* is
        // stamped) and uses GPS ego speed to pick the safe-TTC threshold; TTC
        // primary, distance fallback, near hard stop. Overrides decide_behavior's
        // provisional front-object throttle; no-op with no front object. The
        // SafetyGovernor below still owns the final word. Runs outside the rc_
        // guard so the overlay HUD reflects it too.
        follow_controller_.update(bp);
        // Forward steering/throttle to the RC actuator on every packet,
        // independent of display decimation — actuation safety can't
        // wait for the next rendered frame. The actuator itself is
        // non-blocking (it only stores state) and has an internal
        // watchdog that zeroes the motor on timeout.
        if (rc_) {
            // Smooth the behavior action (STOP/CAUTION hysteresis) and run the
            // fail-closed safety supervisor before actuating. The supervisor
            // forces throttle 0 on effective STOP, on sustained loss of lane
            // geometry, and (full mode) on sustained loss of fresh object
            // data; it also drops safe=false to those conditions. The
            // RcActuator watchdog remains the independent hardware safety
            // layer. bp.control/behavior_action are mutated in
            // place so the overlay HUD reflects what the motor actually does.
            safety_governor_.apply(bp, bp.objects.has_value());
            rc_->command(bp.control.steering, bp.control.throttle,
                         bp.control.safe);
        }
        ++packet_count;
        if (!cfg_.save_video && cfg_.display_every_n > 1 &&
            ((packet_count - 1) % static_cast<std::uint64_t>(cfg_.display_every_n) != 0)) {
            continue;
        }
        cv::Mat out;
        {
            ScopedStage st(stage_timer_, "render_overlay_draw");
            out = render_overlay(bp, display_fps, cfg_.overlay);
        }
        {
            ScopedStage st(stage_timer_, "resize_display");
            out = resize_for_display(out, cfg_.overlay.display_width);
        }

        if (cfg_.save_video && !cfg_.save_video_path.empty()) {
            if (!writer.isOpened()) {
                double output_fps = cam_->source_fps();
                writer.open(cfg_.save_video_path,
                            cv::VideoWriter::fourcc('m','p','4','v'),
                            output_fps, out.size());
                if (!writer.isOpened()) {
                    std::cerr << "[render] VideoWriter open failed: "
                              << cfg_.save_video_path << "\n";
                }
            }
            if (writer.isOpened()) writer.write(out);
        }
        if (cfg_.show_window && !cfg_.dry_run) {
            display_q_.put(std::move(out));
        }
        std::uint64_t rendered_now = rendered_.fetch_add(1, std::memory_order_relaxed) + 1;

        // Update display FPS once per second + periodic drop log.
        ++fps_count;
        auto now_t = std::chrono::steady_clock::now();
        auto dt = std::chrono::duration<double>(now_t - fps_t0).count();
        if (dt >= 1.0) {
            display_fps = static_cast<float>(fps_count / dt);
            fps_count = 0;
            fps_t0 = now_t;
            if (cfg_.eval_log && display_fps > 0.0f) {
                if (!eval_render_fps_seen_ ||
                    display_fps < eval_render_fps_min_) {
                    eval_render_fps_min_ = display_fps;
                    eval_render_fps_seen_ = true;
                }
            }
            if (!cfg_.quiet_logs && cfg_.profile_log_every > 0) {
                std::cerr << "[runtime] render_fps=" << display_fps
                          << " | lane_dropped=" << lane_q_.dropped()
                          << " | render_dropped=" << render_q_.dropped();
                if (gps_) {
                    GpsSample g = gps_->latest();
                    std::cerr << " | gps=" << g.source
                              << (g.valid ? " valid" : " invalid")
                              << " sats=" << g.satellites
                              << " kmh=" << g.speed_kmh
                              << " age=" << g.age_s << "s";
                }
                std::cerr << "\n";
            }
        }

        // Periodic stage-timer summary, keyed off rendered frame count.
        if (!cfg_.quiet_logs && cfg_.profile_enabled &&
            cfg_.profile_log_every > 0 &&
            rendered_now % static_cast<std::uint64_t>(cfg_.profile_log_every) == 0) {
            std::cerr << stage_timer_.format_summary() << "\n";
        }
    }
    if (writer.isOpened()) writer.release();
}

}  // namespace adas
