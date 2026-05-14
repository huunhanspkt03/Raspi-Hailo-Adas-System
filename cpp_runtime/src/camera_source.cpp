#include "adas/camera_source.hpp"

#include <iostream>

namespace adas {

CameraSource::CameraSource(const CameraOptions& opt) : opt_(opt) {}
CameraSource::~CameraSource() { release(); }

bool CameraSource::open() {
    if (opt_.use_camera) {
        // On Raspberry Pi you may prefer a GStreamer pipeline. We try v4l2
        // first and fall back to the default backend.
        if (!cap_.open(opt_.camera_index, cv::CAP_V4L2)) {
            if (!cap_.open(opt_.camera_index)) {
                std::cerr << "Failed to open camera " << opt_.camera_index << "\n";
                return false;
            }
        }
        cap_.set(cv::CAP_PROP_FRAME_WIDTH,  opt_.width);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, opt_.height);
        cap_.set(cv::CAP_PROP_FPS, opt_.fps);
    } else {
        if (opt_.video_path.empty()) {
            std::cerr << "CameraSource: no video path provided and camera disabled\n";
            return false;
        }
        if (!cap_.open(opt_.video_path)) {
            std::cerr << "Failed to open video: " << opt_.video_path << "\n";
            return false;
        }
    }
    frame_w_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    frame_h_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    if (frame_w_ <= 0) frame_w_ = opt_.width;
    if (frame_h_ <= 0) frame_h_ = opt_.height;
    return true;
}

bool CameraSource::read(cv::Mat& frame_bgr) {
    if (!cap_.isOpened()) return false;
    if (!cap_.read(frame_bgr) || frame_bgr.empty()) {
        if (!opt_.use_camera && opt_.loop_video) {
            cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
            if (!cap_.read(frame_bgr) || frame_bgr.empty()) return false;
        } else {
            return false;
        }
    }
    if (frame_w_ == 0) frame_w_ = frame_bgr.cols;
    if (frame_h_ == 0) frame_h_ = frame_bgr.rows;
    return true;
}

void CameraSource::release() {
    if (cap_.isOpened()) cap_.release();
}

double CameraSource::sourceFps() const {
    double f = cap_.get(cv::CAP_PROP_FPS);
    return (f > 0.0 && f < 1000.0) ? f : static_cast<double>(opt_.fps);
}

double CameraSource::sourcePosMs() const {
    double pos_ms = cap_.get(cv::CAP_PROP_POS_MSEC);
    return pos_ms >= 0.0 ? pos_ms : -1.0;
}

}  // namespace adas
