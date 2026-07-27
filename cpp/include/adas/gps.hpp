#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

namespace adas {

struct GpsConfig {
    bool enabled{false};
    std::string port{"/dev/serial0"};
    int  baud{38400};

    
    double stale_timeout_s{2.0};
};

struct GpsSample {
    bool   valid{false};            // fresh fix + within stale_timeout
    double latitude{0.0};           // degrees, signed (N+, S-)
    double longitude{0.0};          // degrees, signed (E+, W-)
    double speed_mps{0.0};
    double speed_kmh{0.0};
    double heading_deg{0.0};        // course over ground, 0..360
    int    satellites{0};
    double hdop{0.0};
    int    fix_quality{0};          // GGA fix quality (0 = no fix)
    double altitude_m{0.0};         // GGA MSL altitude (0 if unknown)
    double age_s{0.0};              // seconds since last accepted sentence
    // "rmc" / "gga" / "stale" / "no_fix" / "port_open_failed" /
    // "unsupported_os" / "disabled" — useful for HUDs and logs.
    std::string source{"disabled"};
};

class GpsReader {
public:
    GpsReader();
    ~GpsReader();
    GpsReader(const GpsReader&) = delete;
    GpsReader& operator=(const GpsReader&) = delete;

    // Start the background reader. Returns true if the thread launched.
    // On non-Linux platforms this returns false and leaves latest() in
    // its "unsupported_os" state — callers should treat that as a soft
    // failure and continue without GPS.
    bool start(const GpsConfig& cfg);

    // Idempotent: safe to call multiple times / from destructor.
    void stop();

    // Returns a snapshot of the most recent parsed fix. `valid` reflects
    // both fix quality and freshness against stale_timeout_s.
    GpsSample latest() const;

    bool running() const { return running_.load(); }

private:
    void run_loop();

    GpsConfig cfg_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_{false};
    std::thread thread_;

    mutable std::mutex mu_;
    GpsSample sample_;             // last good fix (may be stale)
    std::chrono::steady_clock::time_point last_update_{};
    bool ever_updated_{false};
};

}  // namespace adas
