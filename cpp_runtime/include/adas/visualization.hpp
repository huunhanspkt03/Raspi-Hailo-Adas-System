#pragma once

#include "adas/controller.hpp"
#include "adas/deeplab_postprocess.hpp"
#include "adas/object_localizer.hpp"
#include "adas/yolo26s_postprocess.hpp"

#include <opencv2/opencv.hpp>

namespace adas {

class Visualizer {
public:
    void drawLane(cv::Mat& frame_bgr, const LaneResult& lane);
    void drawDetections(cv::Mat& frame_bgr, const std::vector<Detection>& dets);
    void drawHud(cv::Mat& frame_bgr,
                 double fps,
                 double infer_ms,
                 const SteeringOutput& steer,
                 const char* mode,
                 const FrontObject& front);
};

}  // namespace adas
