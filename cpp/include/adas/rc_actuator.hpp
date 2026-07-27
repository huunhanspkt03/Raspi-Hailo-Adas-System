#pragma once

// Forward-only RC actuator for the ADAS car (BTS7960 / STB7960 wiring).
//
// Hardware (Raspberry Pi 5):
//   * Servo signal              : GPIO13  (pin 33)  — hardware PWM, 50 Hz
//   * BTS7960 RPWM (forward PWM): GPIO18  (pin 12)  — hardware PWM, ~20 kHz
//                                 drives motor speed in the forward dir.
//   * BTS7960 LPWM (reverse PWM): GPIO23  (pin 16)  — digital, ALWAYS LOW
//   * BTS7960 R_EN  (enable)    : GPIO27  (pin 13)  — digital HIGH while armed
//   * BTS7960 L_EN  (enable)    : GPIO22  (pin 15)  — digital HIGH while armed
//


#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>

namespace adas {

constexpr int servo_angle_to_us(int angle_deg) {
    return 1000 + (angle_deg * 1000 + 90) / 180;
}

struct RcConfig {
    bool   enabled{false};
    bool   log_only{false};

    int    servo_gpio{13};
    // BTS7960 RPWM — hardware PWM, drives forward motor speed.
    int    motor_pwm_gpio{18};
    // BTS7960 LPWM — digital, always LOW (forward-only).
    int    motor_reverse_gpio{23};
    // BTS7960 enable pins — both held HIGH while armed (driver disabled
    // until both enables are HIGH). Dropped LOW on stop() so the bridge
    // is positively de-energised before fds close.
    int    motor_right_enable_gpio{27};
    int    motor_left_enable_gpio{22};

    // Default steering limits expressed as standard 1000..2000 us
    // servo pulses: left=50 deg, straight=95 deg, right=130 deg.
    int    servo_neutral_us{servo_angle_to_us(95)};
    int    servo_left_us{servo_angle_to_us(50)};
    int    servo_right_us{servo_angle_to_us(130)};
    // Amplify normalized steering before mapping to servo pulse. 1.30 makes
    // mid-range steering commands 30% stronger while the final command remains
    // clamped to the configured left/right endpoints.
    float  steering_gain{1.30f};

    // Throttle shaping applied by the watchdog (NOT by command() callers):
    // min_drive_throttle is applied only when commanded throttle is already
    // positive; it never spins the motor from a zero command. The final duty
    // is clamped to [0,1].
    float  min_drive_throttle{0.10f};

    // If no fresh command() lands within this window, the watchdog
    // forces motor=stop on the next tick.
    double command_timeout_s{0.5};
};

class RcActuator {
public:
    RcActuator();
    ~RcActuator();
    RcActuator(const RcActuator&) = delete;
    RcActuator& operator=(const RcActuator&) = delete;

    // Initialize backend and start the watchdog thread. Returns true
    // if the actuator is now armed (real-HW or log-only). Returns
    // false when cfg.enabled == false.
    bool start(const RcConfig& cfg);

    // Idempotent. Forces a final motor-stop / servo-neutral on the
    // wire before closing fds.
    void stop();

    // Latest desired command from the perception/behavior stack:
    //   steering : [-1, 1]; out-of-range is clamped.
    //   throttle : raw command; the watchdog clamps to [0,1] and applies
    //              min_drive_throttle when positive.
    //   safe     : false → motor stopped regardless of throttle.
    // Non-blocking: stores state under a mutex and returns.
    void command(float steering, float throttle, bool safe);

    bool enabled()  const { return enabled_.load(); }
    bool log_only() const { return log_only_.load(); }

private:
    void run_watchdog();
    void apply_safe_state();                 // motor stop + servo neutral
    void apply_safe_state_steering(float s); // motor stop, hold current steering

    void write_servo_us(int us);
    void write_motor_duty(float duty01);
    void write_enables(bool high);   // R_EN + L_EN together
    void write_reverse(bool high);   // asserted-LOW (forward-only)

    RcConfig cfg_;
    std::atomic<bool> enabled_{false};
    std::atomic<bool> log_only_{false};
    std::atomic<bool> stop_{false};
    std::thread thread_;

    mutable std::mutex mu_;
    float desired_steering_{0.0f};
    float desired_throttle_{0.0f};
    bool  desired_safe_{false};
    std::chrono::steady_clock::time_point last_cmd_{};
    bool  ever_commanded_{false};

#if defined(__linux__)
    void close_fds();
    int servo_duty_fd_{-1};
    int motor_duty_fd_{-1};
    int right_enable_fd_{-1};      // R_EN
    int left_enable_fd_{-1};       // L_EN
    int reverse_value_fd_{-1};     // LPWM
    int motor_period_ns_{50000};   // resolved at start() — 20 kHz default
#endif
};

}  // namespace adas
