#include "adas/preprocess.hpp"

namespace adas {

void preprocessDeeplab512(const cv::Mat& frame_bgr, cv::Mat& out_rgb_512) {
    cv::Mat resized;
    cv::resize(frame_bgr, resized, cv::Size(DEEPLAB_INPUT, DEEPLAB_INPUT),
               0, 0, cv::INTER_LINEAR);
    cv::cvtColor(resized, out_rgb_512, cv::COLOR_BGR2RGB);
}

LetterboxInfo preprocessYolo26s640(const cv::Mat& frame_bgr, cv::Mat& out_rgb_640) {
    const int h = frame_bgr.rows;
    const int w = frame_bgr.cols;
    LetterboxInfo lb;
    lb.scale = std::min(static_cast<float>(YOLO26S_INPUT) / w,
                        static_cast<float>(YOLO26S_INPUT) / h);
    const int new_w = std::max(1, static_cast<int>(std::round(w * lb.scale)));
    const int new_h = std::max(1, static_cast<int>(std::round(h * lb.scale)));
    lb.pad_x = (YOLO26S_INPUT - new_w) / 2.0f;
    lb.pad_y = (YOLO26S_INPUT - new_h) / 2.0f;

    cv::Mat resized;
    cv::resize(frame_bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat canvas(YOLO26S_INPUT, YOLO26S_INPUT, CV_8UC3,
                   cv::Scalar(YOLO26S_PAD, YOLO26S_PAD, YOLO26S_PAD));
    const int off_x = static_cast<int>(lb.pad_x);
    const int off_y = static_cast<int>(lb.pad_y);
    resized.copyTo(canvas(cv::Rect(off_x, off_y, new_w, new_h)));
    cv::cvtColor(canvas, out_rgb_640, cv::COLOR_BGR2RGB);
    return lb;
}

}  // namespace adas
