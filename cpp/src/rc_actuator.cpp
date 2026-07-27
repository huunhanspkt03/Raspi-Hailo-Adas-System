#include "adas/rc_actuator.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#if defined(__linux__)
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#define ADAS_RC_HAS_POSIX 1
#else
#define ADAS_RC_HAS_POSIX 0
#endif

namespace adas {

namespace {

constexpr int kServoPeriodNs = 20'000'000;   // 50 Hz servo frame
constexpr int kMotorPeriodNs = 50'000;       // 20 kHz BTS7960-friendly

constexpr float kServoSteeringGain = 5.0f;

// Env-gated steering trace: set ADAS_DEBUG_STEERING=1 to log control steering,
// gains, effective (post-gain, pre-map) steering, and servo_us on every pulse
// change. Parsed once.
bool debug_steering() {
    static const bool enabled = std::getenv("ADAS_DEBUG_STEERING") != nullptr;
    return enabled;
}

#if ADAS_RC_HAS_POSIX
bool sysfs_write(const std::string& path, const std::string& value) {
    int fd = ::open(path.c_str(), O_WRONLY);
    if (fd < 0) return false;
    ssize_t n = ::write(fd, value.data(), value.size());
    ::close(fd);
    return n == static_cast<ssize_t>(value.size());
}

bool sysfs_exists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

// Poll for a freshly-exported sysfs node to appear (up to ~500 ms).
bool wait_for_sysfs(const std::string& path) {
    for (int i = 0; i < 50 && !sysfs_exists(path); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return sysfs_exists(path);
}

int read_sysfs_int(const std::string& path, int fallback) {
    std::ifstream in(path);
    int v = fallback;
    in >> v;
    return in ? v : fallback;
}

std::string read_sysfs_string(const std::string& path) {
    std::ifstream in(path);
    std::string v;
    std::getline(in, v);
    return v;
}

// Standard Pi PWM-capable header pins. On Pi 5/RP1 the exposed PWM
// controller reports four channels and GPIO18/19 are channels 2/3.
// Older Pi sysfs PWM controllers expose two channels, where GPIO18/19
// alias channels 0/1.
int resolve_pwm_channel(int gpio, int npwm) {
    if (npwm >= 4) {
        switch (gpio) {
            case 12: return 0;
            case 13: return 1;
            case 18: return 2;
            case 19: return 3;
            default: return -1;
        }
    }
    switch (gpio) {
        case 12: case 18: return 0;
        case 13: case 19: return 1;
        default: return -1;
    }
}

// Select the user PWM chip. This build is pinned to the verified Pi setup:
// /sys/class/pwm/pwmchip0 exposes four channels, where GPIO13 maps to pwm1
// and GPIO18 maps to pwm2. Do not scan/fallback to another chip, because
// choosing the wrong controller can silently drive the wrong pins.
std::string resolve_pwm_chip() {
    const std::string chosen = "/sys/class/pwm/pwmchip0";
    if (!sysfs_exists(chosen + "/npwm")) {
        std::cerr << "[rc] fixed pwm chip missing: " << chosen
                  << " (expected Pi PWM controller)\n";
        return {};
    }
    const int npwm = read_sysfs_int(chosen + "/npwm", 0);
    std::cerr << "[rc] pwm chip selected: " << chosen
              << " npwm=" << npwm << " reason=fixed\n";
    return chosen;
}

int resolve_sysfs_gpio_number(int bcm_gpio) {
    int best = -1;
    int best_base = -1;
    DIR* dir = ::opendir("/sys/class/gpio");
    if (!dir) return bcm_gpio;
    while (dirent* ent = ::readdir(dir)) {
        const std::string name = ent->d_name;
        if (name.rfind("gpiochip", 0) != 0) continue;
        const std::string chip = "/sys/class/gpio/" + name;
        const int base = read_sysfs_int(chip + "/base", -1);
        const int ngpio = read_sysfs_int(chip + "/ngpio", 0);
        if (base < 0 || bcm_gpio < 0 || bcm_gpio >= ngpio) continue;
        const std::string label = read_sysfs_string(chip + "/label");
        const bool likely_rp1 = label.find("rp1") != std::string::npos ||
                                label.find("RP1") != std::string::npos;
        if (likely_rp1) {
            best = base + bcm_gpio;
            best_base = base;
            break;
        }
        // On Pi 5 the 40-pin header GPIO controller commonly has the
        // highest dynamic sysfs base. Older Pis normally have base 0.
        if (base > best_base) {
            best = base + bcm_gpio;
            best_base = base;
        }
    }
    ::closedir(dir);
    return best >= 0 ? best : bcm_gpio;
}

int setup_pwm(const std::string& chip, int chan, int period_ns, const char* label) {
    const std::string pwm_dir = chip + "/pwm" + std::to_string(chan);
    if (!sysfs_exists(pwm_dir)) {
        if (!sysfs_write(chip + "/export", std::to_string(chan))) {
            std::cerr << "[rc] pwm " << label << " export failed: "
                      << chip << "/export chan=" << chan
                      << " errno=" << errno << " (" << std::strerror(errno) << ")\n";
            return -1;
        }
        if (!wait_for_sysfs(pwm_dir)) {
            std::cerr << "[rc] pwm " << label << " did not appear after export: "
                      << pwm_dir << "\n";
            return -1;
        }
    }
    // Disable + zero duty so a stale period < duty_cycle ordering doesn't
    // get rejected by the kernel on reconfigure.
    sysfs_write(pwm_dir + "/enable", "0");
    sysfs_write(pwm_dir + "/duty_cycle", "0");
    if (!sysfs_write(pwm_dir + "/period", std::to_string(period_ns))) {
        std::cerr << "[rc] pwm " << label << " period failed: "
                  << pwm_dir << "/period period_ns=" << period_ns
                  << " errno=" << errno << " (" << std::strerror(errno) << ")\n";
        return -1;
    }
    if (!sysfs_write(pwm_dir + "/enable", "1")) {
        std::cerr << "[rc] pwm " << label << " enable failed: "
                  << pwm_dir << "/enable errno=" << errno
                  << " (" << std::strerror(errno) << ")\n";
        return -1;
    }
    int fd = ::open((pwm_dir + "/duty_cycle").c_str(), O_WRONLY);
    if (fd < 0) {
        std::cerr << "[rc] pwm " << label << " duty open failed: "
                  << pwm_dir << "/duty_cycle errno=" << errno
                  << " (" << std::strerror(errno) << ")\n";
    }
    return fd;
}

int setup_gpio_output(int gpio) {
    const int sysfs_gpio = resolve_sysfs_gpio_number(gpio);
    const std::string base = "/sys/class/gpio/gpio" + std::to_string(sysfs_gpio);
    if (!sysfs_exists(base)) {
        if (!sysfs_write("/sys/class/gpio/export", std::to_string(sysfs_gpio))) {
            std::cerr << "[rc] gpio export failed: gpio=" << gpio
                      << " sysfs_gpio=" << sysfs_gpio
                      << " errno=" << errno << " (" << std::strerror(errno) << ")\n";
            return -1;
        }
        if (!wait_for_sysfs(base)) {
            std::cerr << "[rc] gpio did not appear after export: gpio=" << gpio
                      << " sysfs_gpio=" << sysfs_gpio << "\n";
            return -1;
        }
    }
    if (!sysfs_write(base + "/direction", "out")) {
        std::cerr << "[rc] gpio direction failed: gpio=" << gpio
                  << " sysfs_gpio=" << sysfs_gpio
                  << " errno=" << errno << " (" << std::strerror(errno) << ")\n";
        return -1;
    }
    if (!sysfs_write(base + "/value", "0")) {
        std::cerr << "[rc] gpio initial value failed: gpio=" << gpio
                  << " sysfs_gpio=" << sysfs_gpio
                  << " errno=" << errno << " (" << std::strerror(errno) << ")\n";
        return -1;
    }
    int fd = ::open((base + "/value").c_str(), O_WRONLY);
    if (fd < 0) {
        std::cerr << "[rc] gpio value open failed: gpio=" << gpio
                  << " sysfs_gpio=" << sysfs_gpio
                  << " errno=" << errno << " (" << std::strerror(errno) << ")\n";
    }
    return fd;
}

bool fd_write_int(int fd, long v) {
    if (fd < 0) return false;
    char buf[32];
    int n = std::snprintf(buf, sizeof(buf), "%ld", v);
    if (n <= 0) return false;
    if (::lseek(fd, 0, SEEK_SET) < 0) return false;
    return ::write(fd, buf, n) == n;
}
#endif  // ADAS_RC_HAS_POSIX

float effective_throttle(float t, float min_drive) {
    if (t <= 0.0f) return 0.0f;
    t = std::max(t, min_drive);
    return std::clamp(t, 0.0f, 1.0f);
}

// Apply the diagnostic + config steering gains, then clamp to [-1, 1]. The
// clamp guarantees the mapped pulse can never exceed the configured left/right
// endpoints no matter how strong the gains are.
float effective_steering(float steering, const RcConfig& cfg) {
    return std::clamp(steering * kServoSteeringGain * cfg.steering_gain,
                      -1.0f, 1.0f);
}

int steering_to_us(float steering, const RcConfig& cfg) {
    const float eff = effective_steering(steering, cfg);
    const int mid = cfg.servo_neutral_us;
    if (eff >= 0.0f) {
        return static_cast<int>(std::lround(mid + eff * (cfg.servo_right_us - mid)));
    }
    return static_cast<int>(std::lround(mid + eff * (mid - cfg.servo_left_us)));
}

}  // namespace

RcActuator::RcActuator() = default;
RcActuator::~RcActuator() { stop(); }

bool RcActuator::start(const RcConfig& cfg) {
    if (enabled_.load()) return true;
    cfg_ = cfg;
    if (!cfg_.enabled) return false;

    log_only_.store(cfg_.log_only);

#if ADAS_RC_HAS_POSIX
    motor_period_ns_ = kMotorPeriodNs;
    if (!cfg_.log_only) {
        const std::string chip = resolve_pwm_chip();
        const int npwm = chip.empty() ? 0 : read_sysfs_int(chip + "/npwm", 0);
        const int servo_chan = resolve_pwm_channel(cfg_.servo_gpio, npwm);
        const int motor_chan = resolve_pwm_channel(cfg_.motor_pwm_gpio, npwm);
        bool ok = !chip.empty() && servo_chan >= 0 && motor_chan >= 0 &&
                  servo_chan != motor_chan;
        if (ok) {
            servo_duty_fd_    = setup_pwm(chip, servo_chan, kServoPeriodNs, "servo");
            motor_duty_fd_    = setup_pwm(chip, motor_chan, kMotorPeriodNs, "motor (RPWM)");
            right_enable_fd_  = setup_gpio_output(cfg_.motor_right_enable_gpio);
            left_enable_fd_   = setup_gpio_output(cfg_.motor_left_enable_gpio);
            reverse_value_fd_ = setup_gpio_output(cfg_.motor_reverse_gpio);
            ok = servo_duty_fd_ >= 0 && motor_duty_fd_ >= 0 &&
                 right_enable_fd_ >= 0 && left_enable_fd_ >= 0 &&
                 reverse_value_fd_ >= 0;
        }
        if (!ok) {
            std::cerr << "[rc] sysfs setup failed (chip='" << chip
                      << "' servo_chan=" << servo_chan
                      << " motor_chan=" << motor_chan
                      << " npwm=" << npwm
                      << "). Falling back to --rc-log-only.\n";
            close_fds();
            log_only_.store(true);
        } else {
            std::cerr << "[rc] hardware backend armed (servo=GPIO" << cfg_.servo_gpio
                      << " RPWM=GPIO" << cfg_.motor_pwm_gpio
                      << " LPWM=GPIO" << cfg_.motor_reverse_gpio
                      << " R_EN=GPIO" << cfg_.motor_right_enable_gpio
                      << " L_EN=GPIO" << cfg_.motor_left_enable_gpio << ")\n";
        }
    }
#else
    log_only_.store(true);
    std::cerr << "[rc] non-Linux build — RC backend forced to log-only.\n";
#endif

    if (log_only_.load()) {
        std::cerr << "[rc] log-only mode: no GPIO/PWM writes will occur.\n";
    }
    std::cerr << "[rc] min_drive_throttle=" << cfg_.min_drive_throttle
              << " timeout=" << cfg_.command_timeout_s << "s\n";

    apply_safe_state();

    write_enables(true);
    stop_.store(false);
    enabled_.store(true);
    thread_ = std::thread(&RcActuator::run_watchdog, this);
    return true;
}

void RcActuator::stop() {
    if (!enabled_.exchange(false)) return;
    stop_.store(true);
    if (thread_.joinable()) thread_.join();
    apply_safe_state();

    write_enables(false);
#if ADAS_RC_HAS_POSIX
    close_fds();
#endif
}

void RcActuator::command(float steering, float throttle, bool safe) {
    if (!enabled_.load()) return;
    std::lock_guard<std::mutex> lock(mu_);
    desired_steering_ = std::clamp(steering, -1.0f, 1.0f);
    desired_throttle_ = throttle;
    desired_safe_     = safe;
    last_cmd_         = std::chrono::steady_clock::now();
    ever_commanded_   = true;
}

void RcActuator::run_watchdog() {
    using clock = std::chrono::steady_clock;
    int   last_us = -1;
    float last_duty = -1.0f;
    std::string last_reason;

    while (!stop_.load()) {

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (stop_.load()) break;

        float steering = 0.0f;
        float throttle_in = 0.0f;
        bool  safe = false;
        bool  stale = true;
        bool  never_commanded = false;
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (!ever_commanded_) {
                never_commanded = true;
            } else {
                const double age = std::chrono::duration<double>(
                    clock::now() - last_cmd_).count();
                stale = age > cfg_.command_timeout_s;
                if (!stale) {
                    steering    = desired_steering_;
                    throttle_in = desired_throttle_;
                    safe        = desired_safe_;
                }
            }
        }

        float duty = 0.0f;
        std::string reason;
        if (never_commanded) {
            reason = "startup";
        } else if (stale) {
            reason = "timeout";
        } else if (!safe) {
            reason = "unsafe";
        } else {
            const float eff = effective_throttle(
                throttle_in, cfg_.min_drive_throttle);
            if (eff > 0.0f) {
                duty = eff;
                reason = "drive";
            } else {
                reason = "idle";
            }
        }
        const int us = steering_to_us(steering, cfg_);

        // Diagnostic steering trace (ADAS_DEBUG_STEERING=1). Logged only on a
        // pulse change so steady-state holding doesn't spam at 50 Hz.
        if (debug_steering() && us != last_us) {
            const bool saturated =
                std::fabs(effective_steering(steering, cfg_)) >= 0.999f;
            std::cerr << "[steer] control=" << steering
                      << " gain=" << kServoSteeringGain
                      << " cfg_gain=" << cfg_.steering_gain
                      << " effective=" << effective_steering(steering, cfg_)
                      << " servo_us=" << us
                      << " delta_us=" << (us - cfg_.servo_neutral_us)
                      << " (neutral=" << cfg_.servo_neutral_us
                      << " left=" << cfg_.servo_left_us
                      << " right=" << cfg_.servo_right_us << ")"
                      << (saturated ? " [SATURATED]" : "")
                      << (log_only_.load() ? " [LOG-ONLY: servo will NOT move]" : "")
                      << "\n";
        }

        // BTS7960 forward-only: LPWM stays LOW; speed is entirely set
        // by RPWM duty. Enables remain HIGH from start() until stop().
        // Write LPWM first (defined state), then RPWM duty, then servo.
        write_reverse(false);
        write_motor_duty(duty);
        write_servo_us(us);

        const bool changed = us != last_us ||
            std::fabs(duty - last_duty) > 1e-3f ||
            reason != last_reason;
        if (log_only_.load() && changed) {
            std::cerr << "[rc] state servo_us=" << us
                      << " RPWM_duty=" << duty
                      << " LPWM=0 EN=1"
                      << " reason=" << reason << "\n";
        }
        if (changed) {
            last_us = us; last_duty = duty;
            last_reason = std::move(reason);
        }
    }
}

void RcActuator::apply_safe_state() { apply_safe_state_steering(0.0f); }

void RcActuator::apply_safe_state_steering(float steering) {
    // Stop motor BEFORE re-centering the servo so an immediate exit
    // cannot momentarily drive toward the prior steering target.
    // BTS7960 forward-only: RPWM=0, LPWM=0. Enables are managed by
    // start()/stop() — apply_safe_state() can run while still armed
    // (e.g. on watchdog timeout) and we must NOT drop enables here.
    write_motor_duty(0.0f);
    write_reverse(false);
    write_servo_us(steering_to_us(steering, cfg_));
}

void RcActuator::write_servo_us(int us) {
    if (log_only_.load()) return;
#if ADAS_RC_HAS_POSIX
    fd_write_int(servo_duty_fd_, static_cast<long>(us) * 1000L);
#else
    (void)us;
#endif
}

void RcActuator::write_motor_duty(float duty01) {
    duty01 = std::clamp(duty01, 0.0f, 1.0f);
    if (log_only_.load()) return;
#if ADAS_RC_HAS_POSIX
    const long ns = static_cast<long>(std::lround(duty01 * motor_period_ns_));
    fd_write_int(motor_duty_fd_, ns);
#else
    (void)duty01;
#endif
}

void RcActuator::write_enables(bool high) {
    if (log_only_.load()) {
        std::cerr << "[rc] enables -> " << (high ? "HIGH" : "LOW")
                  << " (R_EN=GPIO" << cfg_.motor_right_enable_gpio
                  << " L_EN=GPIO" << cfg_.motor_left_enable_gpio << ")\n";
        return;
    }
#if ADAS_RC_HAS_POSIX
    const int v = high ? 1 : 0;
    fd_write_int(right_enable_fd_, v);
    fd_write_int(left_enable_fd_,  v);
#else
    (void)high;
#endif
}

void RcActuator::write_reverse(bool high) {
    // LPWM is always LOW by design (forward-only). The arg is asserted,
    // not honored.
    (void)high;
    if (log_only_.load()) return;
#if ADAS_RC_HAS_POSIX
    fd_write_int(reverse_value_fd_, 0);
#endif
}

#if ADAS_RC_HAS_POSIX
void RcActuator::close_fds() {
    auto cleanup = [](int& fd){ if (fd >= 0) { ::close(fd); fd = -1; } };
    cleanup(servo_duty_fd_);
    cleanup(motor_duty_fd_);
    cleanup(right_enable_fd_);
    cleanup(left_enable_fd_);
    cleanup(reverse_value_fd_);
}
#endif

}  // namespace adas
