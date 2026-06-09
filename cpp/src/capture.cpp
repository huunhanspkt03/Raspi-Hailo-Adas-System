#include "adas/capture.hpp"

#include <iostream>
#include <sstream>

#include <opencv2/imgproc.hpp>

namespace adas {

Capture::Capture(const CaptureConfig& cfg) : cfg_(cfg) {}
Capture::~Capture() { if (cap_.isOpened()) cap_.release(); }

std::string Capture::build_csi_pipeline(int w, int h, int fps) {
    // libcamera on Raspberry Pi 5 (Bookworm) → BGR appsink.
    std::ostringstream ss;
    ss << "libcamerasrc ! "
       << "video/x-raw,format=BGR,width=" << w << ",height=" << h << ",framerate=" << fps << "/1 ! "
       << "videoconvert ! "
       << "appsink drop=true max-buffers=1 sync=false";
    return ss.str();
}

std::string Capture::build_video_pipeline(const std::string& path) {
    // OpenH264 decode path — matches the Python replay choice that
    // sidesteps the FFmpeg crash documented in PROJECT_STATE.md.
    std::ostringstream ss;
    ss << "filesrc location=\"" << path << "\" ! decodebin ! "
       << "videoconvert ! video/x-raw,format=BGR ! appsink sync=false";
    return ss.str();
}

bool Capture::open() {
    bool ok = false;
    switch (cfg_.source) {
        case CaptureSource::CsiCamera: {
            auto pipeline = build_csi_pipeline(cfg_.width, cfg_.height, cfg_.fps);
            ok = cap_.open(pipeline, cv::CAP_GSTREAMER);
            if (!ok) {
                std::cerr << "[capture] CSI GStreamer pipeline failed: " << pipeline << "\n";
            }
            break;
        }
        case CaptureSource::UsbCamera: {
            ok = cap_.open(cfg_.usb_device_index, cv::CAP_V4L2);
            if (ok) {
                cap_.set(cv::CAP_PROP_FRAME_WIDTH, cfg_.width);
                cap_.set(cv::CAP_PROP_FRAME_HEIGHT, cfg_.height);
                cap_.set(cv::CAP_PROP_FPS, cfg_.fps);
                if (cfg_.prefer_low_latency) cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
            }
            break;
        }
        case CaptureSource::VideoFile: {
            // Try GStreamer first (avoids FFmpeg crash); fall back to default.
            auto pipeline = build_video_pipeline(cfg_.video_path);
            ok = cap_.open(pipeline, cv::CAP_GSTREAMER);
            if (!ok) {
                ok = cap_.open(cfg_.video_path);
            }
            break;
        }
    }
    if (!ok) {
        std::cerr << "[capture] failed to open source\n";
    }
    return ok;
}

std::optional<cv::Mat> Capture::read_frame() {
    if (!cap_.isOpened()) return std::nullopt;
    cv::Mat frame;
    if (!cap_.read(frame) || frame.empty()) {
        if (cfg_.source == CaptureSource::VideoFile) {
            if (cfg_.loop_video) {
                cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
                if (cap_.read(frame) && !frame.empty()) return frame;
            }
            exhausted_ = true;
        }
        return std::nullopt;
    }
    return frame;
}

double Capture::source_fps() const {
    double f = const_cast<cv::VideoCapture&>(cap_).get(cv::CAP_PROP_FPS);
    if (f <= 0.0 || f > 240.0) return static_cast<double>(cfg_.fps);
    return f;
}

double Capture::source_total_frames() const {
    if (cfg_.source != CaptureSource::VideoFile) return -1.0;
    double n = const_cast<cv::VideoCapture&>(cap_).get(cv::CAP_PROP_FRAME_COUNT);
    return n > 0.0 ? n : -1.0;   // GStreamer often reports 0/-1 here
}

}  // namespace adas
