#include "adas/gps.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#if defined(__linux__)
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <termios.h>
#include <unistd.h>
#define ADAS_GPS_HAS_POSIX 1
#else
#define ADAS_GPS_HAS_POSIX 0
#endif

namespace adas {

namespace {

#if ADAS_GPS_HAS_POSIX
speed_t baud_to_speed(int baud) {
    switch (baud) {
        case 4800:   return B4800;
        case 9600:   return B9600;
        case 19200:  return B19200;
        case 38400:  return B38400;
        case 57600:  return B57600;
        case 115200: return B115200;
        default:     return B9600;
    }
}

// Open a serial port in raw 8N1 mode at the requested baud. Returns the
// fd on success or -1 on failure (errno set).
int open_serial(const char* port, int baud) {
    int fd = ::open(port, O_RDONLY | O_NOCTTY | O_NONBLOCK);
    if (fd < 0) return -1;

    struct termios tio{};
    if (tcgetattr(fd, &tio) != 0) { int e = errno; ::close(fd); errno = e; return -1; }
    cfmakeraw(&tio);
    speed_t spd = baud_to_speed(baud);
    cfsetispeed(&tio, spd);
    cfsetospeed(&tio, spd);
    tio.c_cflag |= (CLOCAL | CREAD);
    tio.c_cflag &= ~CSIZE;
    tio.c_cflag |=  CS8;
    tio.c_cflag &= ~PARENB;          // no parity
    tio.c_cflag &= ~CSTOPB;          // 1 stop bit
    tio.c_cflag &= ~CRTSCTS;         // no hw flow control
    tio.c_iflag &= ~(IXON | IXOFF | IXANY);
    tio.c_cc[VMIN]  = 0;
    tio.c_cc[VTIME] = 0;
    if (tcsetattr(fd, TCSANOW, &tio) != 0) { int e = errno; ::close(fd); errno = e; return -1; }
    tcflush(fd, TCIFLUSH);
    return fd;
}
#endif  // ADAS_GPS_HAS_POSIX

// Verify and strip the NMEA "*HH" checksum. Returns true and rewrites
// `s` to the payload between '$' and '*' if the checksum matches.
bool nmea_checksum_strip(std::string& s) {
    if (s.size() < 4 || s.front() != '$') return false;
    auto star = s.rfind('*');
    if (star == std::string::npos || star + 3 > s.size()) return false;
    unsigned expected = 0;
    for (size_t i = star + 1; i < star + 3; ++i) {
        char c = s[i];
        expected <<= 4;
        if      (c >= '0' && c <= '9') expected |= (c - '0');
        else if (c >= 'A' && c <= 'F') expected |= (c - 'A' + 10);
        else if (c >= 'a' && c <= 'f') expected |= (c - 'a' + 10);
        else return false;
    }
    unsigned char actual = 0;
    for (size_t i = 1; i < star; ++i) actual ^= static_cast<unsigned char>(s[i]);
    if (actual != expected) return false;
    s = s.substr(1, star - 1);   // drop leading '$' and trailing '*HH'
    return true;
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    out.reserve(16);
    size_t start = 0;
    for (size_t i = 0; i <= s.size(); ++i) {
        if (i == s.size() || s[i] == ',') {
            out.emplace_back(s.substr(start, i - start));
            start = i + 1;
        }
    }
    return out;
}

// NMEA latitude/longitude are "ddmm.mmmm" / "dddmm.mmmm". Returns NaN
// on parse failure so callers can detect empty fields cleanly.
double parse_lat_lon(const std::string& v, const std::string& hemi, int deg_digits) {
    if (v.empty() || hemi.empty()) return std::nan("");
    if (static_cast<int>(v.size()) < deg_digits + 2) return std::nan("");
    char* end = nullptr;
    double deg = std::strtod(v.substr(0, deg_digits).c_str(), &end);
    if (end == nullptr || *end != '\0') return std::nan("");
    double minutes = std::strtod(v.substr(deg_digits).c_str(), &end);
    if (end == nullptr || *end != '\0') return std::nan("");
    double val = deg + minutes / 60.0;
    if (hemi == "S" || hemi == "W") val = -val;
    return val;
}

double parse_double_or(const std::string& s, double dflt) {
    if (s.empty()) return dflt;
    char* end = nullptr;
    double v = std::strtod(s.c_str(), &end);
    if (end == nullptr || *end != '\0') return dflt;
    return v;
}

int parse_int_or(const std::string& s, int dflt) {
    if (s.empty()) return dflt;
    char* end = nullptr;
    long v = std::strtol(s.c_str(), &end, 10);
    if (end == nullptr || *end != '\0') return dflt;
    return static_cast<int>(v);
}

// $--RMC,utc,status,lat,N/S,lon,E/W,sog_knots,cog_deg,date,...*hh
// Returns true if status=='A' and lat/lon parse cleanly (sets fields on
// `out`). Speed and heading remain at their previous values when the
// sentence omits them.
bool parse_rmc(const std::vector<std::string>& f, GpsSample& out) {
    if (f.size() < 10) return false;
    if (f[2] != "A") return false;  // V = void
    double lat = parse_lat_lon(f[3], f[4], 2);
    double lon = parse_lat_lon(f[5], f[6], 3);
    if (std::isnan(lat) || std::isnan(lon)) return false;
    out.latitude  = lat;
    out.longitude = lon;
    constexpr double kKnotsToMps = 0.514444;
    double sog_knots = parse_double_or(f[7], -1.0);
    if (sog_knots >= 0.0) {
        out.speed_mps = sog_knots * kKnotsToMps;
        out.speed_kmh = out.speed_mps * 3.6;
    }
    double cog = parse_double_or(f[8], -1.0);
    if (cog >= 0.0) out.heading_deg = cog;
    out.source = "rmc";
    return true;
}

// $--GGA,utc,lat,N/S,lon,E/W,fix,sat,hdop,alt,M,...*hh
bool parse_gga(const std::vector<std::string>& f, GpsSample& out) {
    if (f.size() < 10) return false;
    int fix = parse_int_or(f[6], 0);
    out.fix_quality = fix;
    out.satellites  = parse_int_or(f[7], 0);
    double hdop = parse_double_or(f[8], -1.0);
    if (hdop >= 0.0) out.hdop = hdop;
    out.altitude_m = parse_double_or(f[9], out.altitude_m);
    if (fix <= 0) return false;
    double lat = parse_lat_lon(f[2], f[3], 2);
    double lon = parse_lat_lon(f[4], f[5], 3);
    if (std::isnan(lat) || std::isnan(lon)) return false;
    out.latitude  = lat;
    out.longitude = lon;
    out.source = "gga";
    return true;
}

// Talker IDs vary (GP, GN, GA, ...). We only key off the 3-letter
// sentence type after the talker prefix.
bool sentence_type_is(const std::string& payload, const char* tag3) {
    if (payload.size() < 5) return false;
    return payload[2] == tag3[0] && payload[3] == tag3[1] && payload[4] == tag3[2];
}

}  // namespace

GpsReader::GpsReader() = default;

GpsReader::~GpsReader() { stop(); }

bool GpsReader::start(const GpsConfig& cfg) {
    if (running_.load()) return true;
    cfg_ = cfg;
    {
        std::lock_guard<std::mutex> lock(mu_);
        sample_ = GpsSample{};
        sample_.source = cfg_.enabled ? "starting" : "disabled";
        ever_updated_ = false;
    }
    if (!cfg_.enabled) return false;
#if !ADAS_GPS_HAS_POSIX
    std::lock_guard<std::mutex> lock(mu_);
    sample_.source = "unsupported_os";
    return false;
#else
    stop_.store(false);
    running_.store(true);
    thread_ = std::thread(&GpsReader::run_loop, this);
    return true;
#endif
}

void GpsReader::stop() {
    if (!running_.exchange(false)) {
        // Even if we never ran, joining is a no-op; nothing else to do.
        if (thread_.joinable()) thread_.join();
        return;
    }
    stop_.store(true);
    if (thread_.joinable()) thread_.join();
}

GpsSample GpsReader::latest() const {
    std::lock_guard<std::mutex> lock(mu_);
    GpsSample s = sample_;
    if (!cfg_.enabled) {
        s.valid = false;
        s.source = "disabled";
        s.age_s = 0.0;
        return s;
    }
    if (!ever_updated_) {
        s.valid = false;
        s.age_s = 0.0;
        // source already carries "starting" / "port_open_failed" / etc.
        return s;
    }
    using clock = std::chrono::steady_clock;
    double age = std::chrono::duration<double>(clock::now() - last_update_).count();
    s.age_s = age;
    if (age > cfg_.stale_timeout_s) {
        s.valid = false;
        s.source = "stale";
    }
    return s;
}

void GpsReader::run_loop() {
#if ADAS_GPS_HAS_POSIX
    // Reopen loop: if the device disappears (USB unplug, cable wiggle),
    // we back off briefly and retry instead of taking the whole runtime
    // with us.
    while (!stop_.load()) {
        int fd = open_serial(cfg_.port.c_str(), cfg_.baud);
        if (fd < 0) {
            {
                std::lock_guard<std::mutex> lock(mu_);
                sample_.source = "port_open_failed";
                sample_.valid = false;
            }
            std::cerr << "[gps] open(" << cfg_.port << ") failed: "
                      << std::strerror(errno) << " — retrying in 2s\n";
            for (int i = 0; i < 20 && !stop_.load(); ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            continue;
        }
        std::cerr << "[gps] reading from " << cfg_.port
                  << " @" << cfg_.baud << " baud\n";

        std::string line;
        line.reserve(96);
        char buf[256];
        while (!stop_.load()) {
            struct pollfd pfd{fd, POLLIN, 0};
            int pr = ::poll(&pfd, 1, 500);  // 500ms tick keeps stop snappy
            if (pr < 0) {
                if (errno == EINTR) continue;
                break;
            }
            if (pr == 0) continue;          // timeout — loop and re-check stop
            ssize_t n = ::read(fd, buf, sizeof(buf));
            if (n <= 0) {
                if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) continue;
                break;  // EOF / error — reopen
            }
            for (ssize_t i = 0; i < n; ++i) {
                char c = buf[i];
                if (c == '\r') continue;
                if (c == '\n') {
                    if (line.empty()) continue;
                    std::string s = line;
                    line.clear();
                    if (!nmea_checksum_strip(s)) continue;
                    auto fields = split_csv(s);
                    if (fields.empty()) continue;

                    std::lock_guard<std::mutex> lock(mu_);
                    GpsSample working = sample_;
                    bool updated = false;
                    if (sentence_type_is(fields[0], "RMC")) {
                        if (parse_rmc(fields, working)) updated = true;
                        else { working.valid = false; working.source = "no_fix"; }
                    } else if (sentence_type_is(fields[0], "GGA")) {
                        if (parse_gga(fields, working)) updated = true;
                        else { working.valid = false; working.source = "no_fix"; }
                    } else {
                        continue;  // ignore GSA/GSV/VTG/etc.
                    }
                    if (updated) {
                        working.valid = true;
                        last_update_ = std::chrono::steady_clock::now();
                        ever_updated_ = true;
                    }
                    sample_ = std::move(working);
                } else {
                    if (line.size() < 256) line.push_back(c);
                    else line.clear();   // runaway line guard
                }
            }
        }
        ::close(fd);
        if (!stop_.load()) {
            std::cerr << "[gps] serial read ended; reopening...\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
#endif  // ADAS_GPS_HAS_POSIX
}

}  // namespace adas
