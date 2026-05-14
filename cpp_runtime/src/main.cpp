#include "adas/camera_source.hpp"
#include "adas/controller.hpp"
#include "adas/deeplab_postprocess.hpp"
#include "adas/hailo_backend.hpp"
#include "adas/object_localizer.hpp"
#include "adas/preprocess.hpp"
#include "adas/visualization.hpp"
#include "adas/yolo26s_postprocess.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <string>
#include <thread>

using namespace adas;

enum class Mode { Lane, Detection, Full };

struct Args {
    Mode mode = Mode::Full;
    std::string video;
    bool use_camera = false;
    int width = 1280, height = 720, fps = 30;
    std::string deeplab_hef  = "src/models/deeplab_v3_mobilenet_v2_2304.hef";
    std::string yolo26s_hef  = "src/models/yolo26_adas_2404.hef";
    bool yolo26s_hef_explicit = false;
    bool display = false;
    bool realtime_video = false;
    std::string save_video;
    std::string save_mask_video;   // lane-only color overlay on raw frame
    int stats_every_n = 60;        // stdout stats cadence (0 disables)
};

static void usage() {
    std::cout <<
        "adas_runtime --mode lane|detection|full [--video path | --camera]\n"
        "             [--deeplab-hef path] [--yolo26s-hef path]\n"
        "             [--display] [--realtime-video]\n"
        "             [--save-video path]          # write HUD view MP4 (annotated output)\n"
        "             [--save-mask-video path]     # write raw frame + colorized lane classes MP4\n"
        "             [--stats-every N]            # stdout fps/infer stats cadence (0=off)\n"
        "             [--width N] [--height N] [--fps N]\n"
        "\n"
        "Note: --display only shows the live window. It does not write files unless\n"
        "      --save-video or --save-mask-video is also provided.\n"
        "\n"
        "Env flags (logging):\n"
        "  ADAS_LOG_EVERY_N=<N>     HailoBackend: avg inference + raw output stats every N frames\n"
        "  ADAS_DEEPLAB_DEBUG=1     per-class pixel counts every 60 frames\n";
}

static bool fileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

static std::string resolvePreferredPath(
    const std::string& configured,
    std::initializer_list<const char*> candidates) {
    if (fileExists(configured)) return configured;
    for (const char* candidate : candidates) {
        if (fileExists(candidate)) return candidate;
    }
    return configured;
}

static bool parseArgs(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { std::cerr << name << " needs a value\n"; return nullptr; }
            return argv[++i];
        };
        if (k == "--mode") {
            const char* v = next("--mode"); if (!v) return false;
            std::string m = v;
            if      (m == "lane")      a.mode = Mode::Lane;
            else if (m == "detection") a.mode = Mode::Detection;
            else if (m == "full")      a.mode = Mode::Full;
            else { std::cerr << "Unknown mode: " << m << "\n"; return false; }
        }
        else if (k == "--video")       { auto v = next("--video");       if (!v) return false; a.video = v; }
        else if (k == "--camera")      { a.use_camera = true; }
        else if (k == "--deeplab-hef") { auto v = next("--deeplab-hef"); if (!v) return false; a.deeplab_hef = v; }
        else if (k == "--yolo26s-hef") {
            auto v = next("--yolo26s-hef"); if (!v) return false; a.yolo26s_hef = v;
            a.yolo26s_hef_explicit = true;
        }
        else if (k == "--display")     { a.display = true; }
        else if (k == "--realtime-video") { a.realtime_video = true; }
        else if (k == "--save-video")  { auto v = next("--save-video");  if (!v) return false; a.save_video = v; }
        else if (k == "--save-mask-video") { auto v = next("--save-mask-video"); if (!v) return false; a.save_mask_video = v; }
        else if (k == "--stats-every") { auto v = next("--stats-every"); if (!v) return false; a.stats_every_n = std::atoi(v); }
        else if (k == "--width")       { auto v = next("--width");       if (!v) return false; a.width  = std::atoi(v); }
        else if (k == "--height")      { auto v = next("--height");      if (!v) return false; a.height = std::atoi(v); }
        else if (k == "--fps")         { auto v = next("--fps");         if (!v) return false; a.fps    = std::atoi(v); }
        else if (k == "-h" || k == "--help") { usage(); std::exit(0); }
        else { std::cerr << "Unknown arg: " << k << "\n"; return false; }
    }
    if (!a.use_camera && a.video.empty()) {
        std::cerr << "Need --video <path> or --camera\n";
        return false;
    }
    return true;
}

static const char* modeName(Mode m) {
    switch (m) {
        case Mode::Lane:      return "lane";
        case Mode::Detection: return "detection";
        case Mode::Full:      return "full";
    }
    return "?";
}

static int countNonZeroSafe(const cv::Mat& m) {
    return m.empty() ? 0 : cv::countNonZero(m);
}

static int countLaneLinePixels(const LaneResult& lane) {
    int total = 0;
    total += countNonZeroSafe(lane.class_masks[LANE_CLASS_DOTTED_Y]);
    total += countNonZeroSafe(lane.class_masks[LANE_CLASS_DOTTED_W]);
    total += countNonZeroSafe(lane.class_masks[LANE_CLASS_SOLID_W]);
    total += countNonZeroSafe(lane.class_masks[LANE_CLASS_SOLID_Y]);
    return total;
}

int main(int argc, char** argv) {
    Args args;
    if (!parseArgs(argc, argv, args)) { usage(); return 1; }

    if (!args.yolo26s_hef_explicit) {
        args.yolo26s_hef = resolvePreferredPath(
            args.yolo26s_hef,
            {"src/models/yolo26_adas_2404.hef", "src/models/yolo26_adas_1404.hef"}
        );
    }

    CameraOptions copt;
    copt.use_camera = args.use_camera;
    copt.video_path = args.video;
    copt.width = args.width; copt.height = args.height; copt.fps = args.fps;
    CameraSource cam(copt);
    if (!cam.open()) return 2;

    HailoBackend backend;
    const bool want_lane = (args.mode != Mode::Detection);
    const bool want_det  = (args.mode != Mode::Lane);
    if (want_lane) backend.loadDeeplab(args.deeplab_hef);
    if (want_det)  backend.loadYolo26s(args.yolo26s_hef);

    DeeplabPostprocessor lane_post;
    Yolo26sPostprocessor det_post;
    ObjectLocalizer obj_localizer;
    Controller controller;
    Visualizer viz;

    cv::VideoWriter writer;
    cv::VideoWriter mask_writer;
    const bool save = !args.save_video.empty();
    const bool save_mask = !args.save_mask_video.empty();

    // Print the resolved config once so users can see what HEF is being used.
    std::cout << "[adas] mode=" << modeName(args.mode)
              << " deeplab=" << args.deeplab_hef
              << " yolo=" << args.yolo26s_hef
              << " display=" << (args.display ? "on" : "off")
              << " save_video=" << (save ? args.save_video : std::string("-"))
              << " save_mask_video=" << (save_mask ? args.save_mask_video : std::string("-"))
              << std::endl;
    if (args.display && !save && !save_mask) {
        std::cout << "[adas] display-only mode: showing a live window and writing no video files"
                  << std::endl;
    } else if (!args.display && (save || save_mask)) {
        std::cout << "[adas] headless save mode: writing video output without opening a display window"
                  << std::endl;
    }

    cv::Mat frame, deeplab_in, yolo_in;
    auto prev = std::chrono::steady_clock::now();
    double fps_ema = 0.0;
    bool playback_clock_started = false;
    auto playback_start = std::chrono::steady_clock::now();
    double playback_base_ms = 0.0;
    size_t video_frame_index = 0;

    while (true) {
        if (!cam.read(frame)) break;
        ++video_frame_index;
        const int W = frame.cols;
        const int H = frame.rows;

        auto t0 = std::chrono::steady_clock::now();

        LaneResult lane;
        std::vector<Detection> detections;
        FrontObject front_obj;
        LetterboxInfo lb;

        if (want_lane) {
            preprocessDeeplab512(frame, deeplab_in);
            auto out = backend.infer(ModelId::Deeplab,
                                     deeplab_in.data, DEEPLAB_INPUT, DEEPLAB_INPUT);
            lane = lane_post.process(out, W, H);
        }
        if (want_det) {
            lb = preprocessYolo26s640(frame, yolo_in);
            auto out = backend.infer(ModelId::Yolo26s,
                                     yolo_in.data, YOLO26S_INPUT, YOLO26S_INPUT);
            detections = det_post.process(out, lb, W, H);
            front_obj = obj_localizer.localize(detections, W, H);
        }

        auto t1 = std::chrono::steady_clock::now();
        double infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        SteeringOutput steer = controller.compute(
            lane.valid ? lane.control_x : -1, W);

        // Clone BEFORE drawing so the mask video is a clean lane-only overlay
        // (no detections, no HUD). Cheap — one memcpy per frame.
        cv::Mat mask_overlay;
        if (save_mask && want_lane) {
            mask_overlay = frame.clone();
            viz.drawLane(mask_overlay, lane);
        }

        if (want_lane) viz.drawLane(frame, lane);
        if (want_det)  viz.drawDetections(frame, detections);

        auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(now - prev).count();
        prev = now;
        double inst = dt > 0 ? 1.0 / dt : 0.0;
        fps_ema = fps_ema == 0.0 ? inst : 0.9 * fps_ema + 0.1 * inst;
        viz.drawHud(frame, fps_ema, infer_ms, steer, modeName(args.mode), front_obj);

        if (save) {
            if (!writer.isOpened()) {
                int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
                writer.open(args.save_video, fourcc,
                            cam.sourceFps() > 0 ? cam.sourceFps() : 30.0,
                            cv::Size(W, H));
            }
            if (writer.isOpened()) writer.write(frame);
        }

        if (save_mask && !mask_overlay.empty()) {
            if (!mask_writer.isOpened()) {
                int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
                mask_writer.open(args.save_mask_video, fourcc,
                                 cam.sourceFps() > 0 ? cam.sourceFps() : 30.0,
                                 cv::Size(W, H));
            }
            if (mask_writer.isOpened()) mask_writer.write(mask_overlay);
        }

        // Periodic stdout stats — matches the "detailed logging" requested for
        // Pi runs so FPS/infer time can be inspected without enabling verbose
        // Hailo logging. Default every 60 frames; --stats-every 0 disables.
        if (args.stats_every_n > 0 &&
            (video_frame_index % static_cast<size_t>(args.stats_every_n) == 0)) {
            std::cout << "[stats] frame=" << video_frame_index
                      << " fps=" << static_cast<int>(fps_ema + 0.5)
                      << " infer=" << static_cast<int>(infer_ms + 0.5) << "ms"
                      << " lane_valid=" << (lane.valid ? 1 : 0)
                      << " conf=" << lane.confidence
                      << " ctrl=(" << lane.control_x << "," << lane.control_y << ")"
                      << " area_px=" << countNonZeroSafe(lane.lane_area_mask)
                      << " line_px=" << countLaneLinePixels(lane)
                      << " dets=" << detections.size()
                      << std::endl;
        }

        int wait_ms = 1;
        if (!args.use_camera && args.realtime_video) {
            double target_ms = cam.sourcePosMs();
            if (target_ms < 0.0) {
                double fps = cam.sourceFps();
                if (fps > 0.0 && fps < 1000.0) {
                    target_ms = (static_cast<double>(video_frame_index) - 1.0) * 1000.0 / fps;
                }
            }
            if (target_ms >= 0.0) {
                if (!playback_clock_started) {
                    playback_clock_started = true;
                    playback_start = std::chrono::steady_clock::now();
                    playback_base_ms = target_ms;
                }
                double relative_target_ms = target_ms - playback_base_ms;
                double elapsed_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - playback_start).count();
                double remaining_ms = relative_target_ms - elapsed_ms;
                if (remaining_ms > 1.0) {
                    wait_ms = std::max(1, static_cast<int>(std::ceil(remaining_ms)));
                }
            }
        }

        if (args.display) {
            cv::imshow("adas_runtime", frame);
            int k = cv::waitKey(wait_ms) & 0xff;
            if (k == 'q' || k == 27) break;
        } else if (wait_ms > 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
        }
    }

    if (writer.isOpened()) writer.release();
    if (mask_writer.isOpened()) mask_writer.release();
    cam.release();
    if (args.display) cv::destroyAllWindows();
    return 0;
}
