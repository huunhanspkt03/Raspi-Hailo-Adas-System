#pragma once

// GPS reader for U-Blox NEO-M10 (or compatible NMEA-0183 receivers) over
// UART. Background thread reads bytes from a serial port (POSIX termios),
// splits them into NMEA sentences, and parses the latest RMC + GGA fields
// into a single GpsSample snapshot guarded by a mutex.
//
// Wiring on Raspberry Pi 5 (default port /dev/serial0):
//   GPS TX  -> Pi GPIO15 RXD (pin 10)
//   GPS RX  -> Pi GPIO14 TXD (pin  8)
//   GND     -> Pi GND       (pin  6 / 9 / etc.)
//   VCC     -> 3V3 or 5V per module spec
// On Pi 5, enable the UART: add `enable_uart=1` to /boot/firmware/config.txt
// and disable the serial-console getty so /dev/serial0 is free for data.
//
// Scope:
//   * Disabled by default; only constructed/started when the runtime
//     config asks for it.
//   * Read-only: never used for steering. Surfaces speed/position for
//     logging and future ego-speed fusion.
//   * Non-blocking with respect to the perception pipeline: the reader
//     thread owns its serial fd and never touches camera/Hailo/display
//     state. Consumers read snapshots via latest().
//   * If no fresh sample arrives within stale_timeout_s, latest() flips
//     `valid` to false and sets `reason` accordingly.
//
// Build note: the implementation guards termios includes behind __linux__
// so the Windows dev build stays green. On non-Linux the reader becomes
// a no-op stub that always reports `valid=false, reason="unsupported_os"`.

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
    // Mark sample invalid if the parser hasn't accepted a fresh fix in
    // this many seconds. Kept generous because NEO-M10 emits ~1 Hz and
    // some sentences (e.g. GGA-only between RMCs) only partially refresh.
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
