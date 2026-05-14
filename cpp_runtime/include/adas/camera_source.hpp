#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace adas {

struct CameraOptions {
    bool use_camera = false;
    std::string video_path;
    int width  = 1280;
    int height = 720;
    int fps    = 30;
    int camera_index = 0;
    bool loop_video = true;
};

class CameraSource {
public:
    explicit CameraSource(const CameraOptions& opt);
    ~CameraSource();

    bool open();
    bool read(cv::Mat& frame_bgr);
    void release();

    int width()  const { return frame_w_; }
    int height() const { return frame_h_; }
    double sourceFps() const;
    double sourcePosMs() const;

private:
    CameraOptions opt_;
    cv::VideoCapture cap_;
    int frame_w_ = 0;
    int frame_h_ = 0;
};

}  // namespace adas
