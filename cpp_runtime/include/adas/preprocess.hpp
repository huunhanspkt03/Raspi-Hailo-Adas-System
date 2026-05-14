#pragma once

#include <opencv2/opencv.hpp>

namespace adas {

// Fixed input sizes — do not generalise.
constexpr int DEEPLAB_INPUT = 512;
constexpr int YOLO26S_INPUT = 640;
constexpr uint8_t YOLO26S_PAD = 114;

struct LetterboxInfo {
    float scale = 1.0f;
    float pad_x = 0.0f;
    float pad_y = 0.0f;
};

// Resize BGR -> 512x512 RGB uint8 for DeepLab.
void preprocessDeeplab512(const cv::Mat& frame_bgr, cv::Mat& out_rgb_512);

// Letterbox BGR -> 640x640 RGB uint8 for YOLO26s. Returns the mapping so
// detections can be rescaled back to the original frame.
LetterboxInfo preprocessYolo26s640(const cv::Mat& frame_bgr, cv::Mat& out_rgb_640);

}  // namespace adas
