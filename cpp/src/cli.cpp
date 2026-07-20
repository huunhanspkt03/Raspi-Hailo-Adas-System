#include "adas/cli.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace adas {

namespace {
bool eq(const char* a, const char* b) { return std::strcmp(a, b) == 0; }

void apply_operator_mode(RuntimeConfig* cfg) {
    // full / python_parity / alternate are the only pipeline configuration
    // this build supports — nothing to switch here anymore.
    cfg->capture.width = 1280;
    cfg->capture.height = 720;
    cfg->capture.fps = 30;
    cfg->overlay.display_width = 1280;
    cfg->gps.enabled = true;
    cfg->rc.enabled = true;
    cfg->quiet_logs = true;
}
}  // namespace

void print_usage(const char* argv0) {
    std::cerr <<
        "Usage: " << argv0 << " [options]\n"
        "  --video <path>          Replay video file (GStreamer decode path).\n"
        "  --camera csi|usb        Live camera; default: csi.\n"
        "  --lane-hef <path>       Lane HEF; default: src/models/stdc1_lane_640.hef\n"
        "  --yolo-hef <path>       YOLO HEF; default: src/models/yolo26n_object.hef\n"
        "  --mode full             Pipeline mode (full is the only supported\n"
        "                          value; lane/detection-only were removed).\n"
        "  --operator-mode         Preset for normal full operation:\n"
        "                          --width 1280 --height 720 --fps 30\n"
        "                          --display-width 1280 --gps --rc-enable --quiet.\n"
        "                          Does not choose --video/--camera.\n"
        "  --no-display            Do not open cv::imshow window.\n"
        "  --save-video <path>     Write annotated mp4 alongside the run.\n"
        "  --dry-run [N]           Headless N-frame smoke (default 30) then exit.\n"
        "  --width <N>             Capture width (default 1280).\n"
        "  --height <N>            Capture height (default 720).\n"
        "  --fps <N>               Capture fps (default 30).\n"
        "  --display-width <N>     Resize for display (default 960).\n"
        "  --display-every-n <N>   Render/show every Nth output packet (default 1).\n"
        "                          Perception still runs on every queued lane frame.\n"
        "                          Ignored when --save-video is set.\n"
        "  --loop-video            Restart finite video sources on EOF.\n"
        "  --object-every-n <N>    Run YOLO every Nth captured frame (default 2).\n"
        "                          Use 1 for FPS testing.\n"
        "  --object-conf <F>       YOLO confidence threshold (default 0.50).\n"
        "  --model-schedule parallel\n"
        "                          Hailo schedule (parallel is the only\n"
        "                          supported value; alternate was removed).\n"
        "                          Lane runs every frame; YOLO every Nth\n"
        "                          (--object-every-n). Lane drives the render.\n"
        "  --lane-mode python_parity\n"
        "                          Lane geometry path (python_parity is the only\n"
        "                          supported value; image was removed).\n"
        "  --profile               Enable per-stage EMA/max timing.\n"
        "  --profile-log-every <N> Print stage summary every N rendered frames\n"
        "                          (default 100; also gates render_fps line).\n"
        "  --quiet                 Suppress periodic / repeated stderr logs\n"
        "                          (render_fps, stage summaries, [lane_parity]\n"
        "                          diagnostics). Startup banner + fatal errors\n"
        "                          and source-exhausted still print. UI overlay\n"
        "                          text is unaffected.\n"
        "  --eval-log              Print a copy-paste [eval] statistics block to\n"
        "                          stdout at shutdown (frame counts, FPS, drops,\n"
        "                          GPS speed). Does not change pipeline behavior;\n"
        "                          prints even under --quiet.\n"
        "  --gps                   Enable GPS reader (U-Blox NEO-M10 over UART).\n"
        "                          Disabled by default. Runs in its own thread;\n"
        "                          never used for steering — speed/position only.\n"
        "  --gps-port <path>       Serial device (default: /dev/serial0).\n"
        "  --gps-stale-timeout <F> Mark GPS invalid after F seconds without a\n"
        "                          fresh fix (default: 2.0).\n"
        "  --rc-enable             Arm the RC actuator (servo + BTS7960 forward-\n"
        "                          only). Disabled by default; when armed, motor\n"
        "                          duty follows behavior/safety commands.\n"
        "  --rc-log-only           Run the RC layer in log-only mode (no GPIO/\n"
        "                          PWM writes). Useful for verifying behavior\n"
        "                          end-to-end before connecting the H-bridge.\n"
        "  --servo-gpio <N>        Servo signal GPIO (default 13, pin 33).\n"
        "  --motor-pwm-gpio <N>    BTS7960 RPWM (forward PWM) GPIO\n"
        "                          (default 18, pin 12). Drives motor speed.\n"
        "  --motor-reverse-gpio <N> BTS7960 LPWM GPIO (default 23, pin 16).\n"
        "                          Always driven LOW — no reverse.\n"
        "  --motor-ren-gpio <N>    BTS7960 R_EN GPIO (default 27, pin 13).\n"
        "                          Held HIGH while armed; LOW on stop.\n"
        "  --motor-len-gpio <N>    BTS7960 L_EN GPIO (default 22, pin 15).\n"
        "                          Held HIGH while armed; LOW on stop.\n"
        "  --servo-neutral-us <N>  Servo neutral pulse width µs (default 1528 ~= 95 deg).\n"
        "  --servo-left-us <N>     Servo full-left pulse width µs (default 1278 ~= 50 deg).\n"
        "  --servo-right-us <N>    Servo full-right pulse width µs (default 1722 ~= 130 deg).\n"
        "  --min-drive-throttle <F> Minimum applied duty once throttle>0\n"
        "                          (overcome stiction; default 0.10).\n";
}

int parse_cli(int argc, char** argv, RuntimeConfig* cfg, std::string* error) {
    if (cfg == nullptr) return 1;
    *cfg = RuntimeConfig{};
    cfg->capture.source = CaptureSource::CsiCamera;
    cfg->capture.video_path = "";
    cfg->show_window = true;
    cfg->overlay.display_width = 960;
    cfg->overlay.show_hud = true;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                if (error) *error = std::string("missing value for ") + name;
                return nullptr;
            }
            return argv[++i];
        };
        if (eq(a, "--video")) {
            const char* v = next("--video"); if (!v) return 1;
            cfg->capture.source = CaptureSource::VideoFile;
            cfg->capture.video_path = v;
        } else if (eq(a, "--camera")) {
            const char* v = next("--camera"); if (!v) return 1;
            if (eq(v, "csi")) cfg->capture.source = CaptureSource::CsiCamera;
            else if (eq(v, "usb")) cfg->capture.source = CaptureSource::UsbCamera;
            else { if (error) *error = std::string("unknown camera type: ") + v; return 1; }
        } else if (eq(a, "--lane-hef")) {
            const char* v = next("--lane-hef"); if (!v) return 1;
            cfg->lane_hef = v;
        } else if (eq(a, "--yolo-hef")) {
            const char* v = next("--yolo-hef"); if (!v) return 1;
            cfg->yolo_hef = v;
        } else if (eq(a, "--mode")) {
            const char* v = next("--mode"); if (!v) return 1;
            // Validation only — full is the sole supported pipeline mode.
            if (!eq(v, "full")) {
                if (error) *error = std::string("--mode ") + v +
                    " was removed; this build supports --mode full only";
                return 1;
            }
        } else if (eq(a, "--operator-mode")) {
            apply_operator_mode(cfg);
        } else if (eq(a, "--no-display")) {
            cfg->show_window = false;
        } else if (eq(a, "--save-video")) {
            const char* v = next("--save-video"); if (!v) return 1;
            cfg->save_video = true; cfg->save_video_path = v;
        } else if (eq(a, "--dry-run")) {
            cfg->dry_run = true;
            cfg->show_window = false;
            // Optional integer count follows.
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                cfg->dry_run_frames = std::atoi(argv[++i]);
                if (cfg->dry_run_frames <= 0) cfg->dry_run_frames = 30;
            }
        } else if (eq(a, "--width")) {
            const char* v = next("--width"); if (!v) return 1;
            cfg->capture.width = std::atoi(v);
        } else if (eq(a, "--height")) {
            const char* v = next("--height"); if (!v) return 1;
            cfg->capture.height = std::atoi(v);
        } else if (eq(a, "--fps")) {
            const char* v = next("--fps"); if (!v) return 1;
            cfg->capture.fps = std::atoi(v);
        } else if (eq(a, "--display-width")) {
            const char* v = next("--display-width"); if (!v) return 1;
            cfg->overlay.display_width = std::atoi(v);
        } else if (eq(a, "--display-every-n")) {
            const char* v = next("--display-every-n"); if (!v) return 1;
            int n = std::atoi(v);
            if (n <= 0) { if (error) *error = "--display-every-n requires N>=1"; return 1; }
            cfg->display_every_n = n;
        } else if (eq(a, "--loop-video")) {
            cfg->capture.loop_video = true;
        } else if (eq(a, "--object-every-n")) {
            const char* v = next("--object-every-n"); if (!v) return 1;
            int n = std::atoi(v);
            if (n <= 0) { if (error) *error = "--object-every-n requires N>=1"; return 1; }
            cfg->object_every_n = n;
        } else if (eq(a, "--object-conf")) {
            const char* v = next("--object-conf"); if (!v) return 1;
            cfg->object_decode.conf_thresh = static_cast<float>(std::atof(v));
        } else if (eq(a, "--model-schedule")) {
            const char* v = next("--model-schedule"); if (!v) return 1;
            // Validation only — parallel is the sole supported schedule.
            if (!eq(v, "parallel")) {
                if (error) *error = std::string("--model-schedule ") + v +
                    " was removed; this build runs parallel only "
                    "(lane every frame, YOLO every --object-every-n)";
                return 1;
            }
        } else if (eq(a, "--lane-mode")) {
            const char* v = next("--lane-mode"); if (!v) return 1;
            // Validation only — python_parity is the sole supported lane path.
            if (!eq(v, "python_parity")) {
                if (error) *error = std::string("--lane-mode ") + v +
                    " was removed; this build supports python_parity only";
                return 1;
            }
        } else if (eq(a, "--quiet")) {
            cfg->quiet_logs = true;
        } else if (eq(a, "--eval-log")) {
            cfg->eval_log = true;
        } else if (eq(a, "--profile")) {
            cfg->profile_enabled = true;
        } else if (eq(a, "--profile-log-every")) {
            const char* v = next("--profile-log-every"); if (!v) return 1;
            int n = std::atoi(v);
            if (n <= 0) { if (error) *error = "--profile-log-every requires N>0"; return 1; }
            cfg->profile_log_every = n;
        } else if (eq(a, "--gps")) {
            cfg->gps.enabled = true;
        } else if (eq(a, "--gps-port")) {
            const char* v = next("--gps-port"); if (!v) return 1;
            cfg->gps.port = v;
        } else if (eq(a, "--gps-stale-timeout")) {
            const char* v = next("--gps-stale-timeout"); if (!v) return 1;
            double t = std::atof(v);
            if (t <= 0.0) { if (error) *error = "--gps-stale-timeout requires F>0"; return 1; }
            cfg->gps.stale_timeout_s = t;
        } else if (eq(a, "--rc-enable")) {
            cfg->rc.enabled = true;
        } else if (eq(a, "--rc-log-only")) {
            cfg->rc.log_only = true;
        } else if (eq(a, "--servo-gpio")) {
            const char* v = next("--servo-gpio"); if (!v) return 1;
            cfg->rc.servo_gpio = std::atoi(v);
        } else if (eq(a, "--motor-pwm-gpio")) {
            const char* v = next("--motor-pwm-gpio"); if (!v) return 1;
            cfg->rc.motor_pwm_gpio = std::atoi(v);
        } else if (eq(a, "--motor-reverse-gpio")) {
            const char* v = next("--motor-reverse-gpio"); if (!v) return 1;
            cfg->rc.motor_reverse_gpio = std::atoi(v);
        } else if (eq(a, "--motor-ren-gpio")) {
            const char* v = next("--motor-ren-gpio"); if (!v) return 1;
            cfg->rc.motor_right_enable_gpio = std::atoi(v);
        } else if (eq(a, "--motor-len-gpio")) {
            const char* v = next("--motor-len-gpio"); if (!v) return 1;
            cfg->rc.motor_left_enable_gpio = std::atoi(v);
        } else if (eq(a, "--servo-neutral-us")) {
            const char* v = next("--servo-neutral-us"); if (!v) return 1;
            cfg->rc.servo_neutral_us = std::atoi(v);
        } else if (eq(a, "--servo-left-us")) {
            const char* v = next("--servo-left-us"); if (!v) return 1;
            cfg->rc.servo_left_us = std::atoi(v);
        } else if (eq(a, "--servo-right-us")) {
            const char* v = next("--servo-right-us"); if (!v) return 1;
            cfg->rc.servo_right_us = std::atoi(v);
        } else if (eq(a, "--min-drive-throttle")) {
            const char* v = next("--min-drive-throttle"); if (!v) return 1;
            float t = static_cast<float>(std::atof(v));
            if (t < 0.0f) { if (error) *error = "--min-drive-throttle must be >= 0"; return 1; }
            cfg->rc.min_drive_throttle = t;
        } else if (eq(a, "-h") || eq(a, "--help")) {
            print_usage(argv[0]);
            return 2;
        } else {
            if (error) *error = std::string("unknown flag: ") + a;
            return 1;
        }
    }
    if (cfg->save_video) {
        cfg->display_every_n = 1;
    }
    return 0;
}

}  // namespace adas
