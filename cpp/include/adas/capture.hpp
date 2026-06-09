#pragma once

// Camera / video capture wrapper. Picks the right OpenCV backend based on
// the input source (CSI GStreamer, video file, USB webcam) and exposes a
// blocking read_frame() that returns the next BGR frame or std::nullopt on
// EOF / fatal error.

#include <cstdint>
#include <optional>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

namespace adas {

enum class CaptureSource { CsiCamera, UsbCamera, VideoFile };

struct CaptureConfig {
    CaptureSource source{CaptureSource::VideoFile};
    std::string video_path;        // used when source == VideoFile
    int width{1280};
    int height{720};
    int fps{30};
    int usb_device_index{0};
    bool prefer_low_latency{true};
    bool loop_video{false};
};

class Capture {
public:
    explicit Capture(const CaptureConfig& cfg);
    ~Capture();
    Capture(const Capture&) = delete;
    Capture& operator=(const Capture&) = delete;

    bool open();
    std::optional<cv::Mat> read_frame();  // BGR uint8
    bool is_exhausted() const { return exhausted_; }
    double source_fps() const;
    // Total frame count for VideoFile sources from OpenCV metadata, or < 0
    // when unknown (live cameras, or a GStreamer pipeline that doesn't
    // report CAP_PROP_FRAME_COUNT). Used by --eval-log only.
    double source_total_frames() const;

private:
    static std::string build_csi_pipeline(int w, int h, int fps);
    static std::string build_video_pipeline(const std::string& path);

    CaptureConfig cfg_;
    cv::VideoCapture cap_;
    bool exhausted_{false};
};

}  // namespace adas
