#pragma once

#include <string>

#include "adas/runtime.hpp"

namespace adas {

// Parse argv into a RuntimeConfig. Recognised flags:
//   --video <path>              video replay
//   --camera csi|usb            live camera (default csi on Pi)
//   --lane-hef <path>           override lane HEF path
//   --yolo-hef <path>           override yolo HEF path
//   --no-display                disable cv::imshow
//   --save-video <path>         write annotated mp4 alongside display
//   --mode lane|detection|full  lane-only (default), object-only, or full
//   --operator-mode             full driving preset; still choose video/camera
//   --dry-run [N]               headless N-frame smoke; default N=30
//   --width / --height / --fps
//   --display-width <N>         resize for cv::imshow (default 960)
//   --display-every-n <N>       render/show every Nth output packet
//   --loop-video                loop a finite video source
int parse_cli(int argc, char** argv, RuntimeConfig* cfg, std::string* error);

void print_usage(const char* argv0);

}  // namespace adas
