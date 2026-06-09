#include "adas/overlay.hpp"
#include "adas/object_detection.hpp"
#include "adas/ui_smoothing.hpp"
#include "adas/warning_zone.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <sstream>

#include <opencv2/imgproc.hpp>

namespace adas {

namespace {

void blend_mask_color(cv::Mat& bgr, const cv::Mat& mask, const cv::Scalar& color, double alpha) {
    if (mask.empty() || bgr.empty()) return;
    if (mask.rows != bgr.rows || mask.cols != bgr.cols) return;
    // In-place full-frame blend with rounded integer alpha math. The inner
    // loop already skips zero-mask pixels, so a bounding box would only limit
    // which zero pixels get visited — output is identical either way. For the
    // lane_area / line_union masks (which span most of the frame) cv::boundingRect
    // was a wasted whole-mask scan that produced a bbox ~= the frame anyway, so
    // blending the full frame directly is both simpler and fewer passes.
    constexpr int kAlphaScale = 10000;
    const int a = std::clamp(
        static_cast<int>(std::lround(alpha * kAlphaScale)), 0, kAlphaScale);
    const int inv_a = kAlphaScale - a;
    const int cb = static_cast<int>(color[0]) * a + kAlphaScale / 2;
    const int cg = static_cast<int>(color[1]) * a + kAlphaScale / 2;
    const int cr = static_cast<int>(color[2]) * a + kAlphaScale / 2;
    const int h = bgr.rows;
    const int w = bgr.cols;
    for (int y = 0; y < h; ++y) {
        const std::uint8_t* mrow = mask.ptr<std::uint8_t>(y);
        cv::Vec3b* prow = bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < w; ++x) {
            if (!mrow[x]) continue;
            cv::Vec3b& px = prow[x];
            px[0] = static_cast<std::uint8_t>((px[0] * inv_a + cb) / kAlphaScale);
            px[1] = static_cast<std::uint8_t>((px[1] * inv_a + cg) / kAlphaScale);
            px[2] = static_cast<std::uint8_t>((px[2] * inv_a + cr) / kAlphaScale);
        }
    }
}

void draw_polyline(cv::Mat& bgr, const std::vector<cv::Point2f>& pts,
                   const cv::Scalar& color, int thickness) {
    if (pts.size() < 2) return;
    std::vector<cv::Point> ipts;
    ipts.reserve(pts.size());
    for (const auto& p : pts) {
        ipts.emplace_back(cv::Point(cvRound(p.x), cvRound(p.y)));
    }
    cv::polylines(bgr, ipts, /*isClosed=*/false, color, thickness, cv::LINE_AA);
}

// --- Operator-HUD primitives ---------------------------------------------
//
// The live UI is draw_adas_panels (LDWS/LKAS + mini LANE + FCWS side panels);
// draw_warning_icon (Green/Orange/Red glyph) is its shared collision icon.
// All icons are vector-drawn with OpenCV primitives — no PNG decode, no
// per-frame asset loading. FCWS severity is classified at draw time from the
// front object's distance vs the ego-speed-tiered caution distance
// (classify_fcws), mapped to the WarningZone glyph color.

// WarningZone is defined in adas/warning_zone.hpp (display-severity glyph level).
const cv::Scalar kZoneColorsBgr[3] = {
    cv::Scalar( 35, 190,  60),   // green  — MONITOR
    cv::Scalar(  0, 140, 255),   // orange — CAUTION
    cv::Scalar(  0,   0, 255),   // red    — STOP
};

// Dark translucent panel (ADAS side-panel look): a near-black fill blended in,
// with a thin light border. ROI-restricted.
void blend_dark_panel(cv::Mat& out, int x1, int y1, int x2, int y2,
                      double alpha) {
    cv::Rect r(x1, y1, x2 - x1, y2 - y1);
    r &= cv::Rect(0, 0, out.cols, out.rows);
    if (r.width <= 0 || r.height <= 0) return;
    cv::Mat roi = out(r);
    cv::Mat panel(r.size(), out.type(), cv::Scalar(28, 28, 28));
    cv::addWeighted(panel, alpha, roi, 1.0 - alpha, 0.0, roi);
    cv::rectangle(out, {r.x, r.y}, {r.x + r.width - 1, r.y + r.height - 1},
                  cv::Scalar(90, 90, 90), 1, cv::LINE_AA);
}

void clamp_icon_center(int w, int h, int size, int* cx, int* cy) {
    const int radius = std::max(4, size / 2 + 3);
    if (w <= radius * 2) *cx = w / 2;
    else                 *cx = std::clamp(*cx, radius, w - radius - 1);
    if (h <= radius * 2) *cy = h / 2;
    else                 *cy = std::clamp(*cy, radius, h - radius - 1);
}

// LaneKeepDir is moved up here so the icon-cache renderers below can see
// it without splitting the cache block across the file.
enum class LaneKeepDir { Straight, Left, Right };

// --- Cached icon sprites -------------------------------------------------
//
// Warning icons and lane-keep arrows are static images modulo class and
// size, so we render each unique (kind, size, variant) once into a small
// CV_8UC4 premultiplied-alpha canvas and reuse it for every subsequent
// frame. OpenCV's anti-aliased primitives, when written to a CV_8UC4
// canvas initialised to (0,0,0,0) with 4-element scalars, naturally
// produce premultiplied output, so the composite step is just:
//   dst[c] = src[c] + dst[c] * (255 - src_alpha) / 255
// (Porter-Duff "over" for premultiplied src).
//
// The cache lives in a thread_local because render_overlay is called
// from a single thread (Runtime::render_loop). Sprites are keyed on
// size so the legitimately-distinct HUD-size and per-detection-size
// requests coexist without re-rendering each frame.

// Outer-stroke padding around each cached icon so AA strokes don't clip
// at the canvas edge. Sized for the widest stroke that draw_warning_icon
// / draw_lane_keep_arrow can produce at the sizes the runtime uses.
constexpr int kIconCanvasPad = 8;

void paste_bgra(cv::Mat& out, const cv::Mat& src_bgra,
                int dst_x, int dst_y) {
    if (src_bgra.empty() || out.empty()) return;
    cv::Rect r(dst_x, dst_y, src_bgra.cols, src_bgra.rows);
    r &= cv::Rect(0, 0, out.cols, out.rows);
    if (r.width <= 0 || r.height <= 0) return;
    const int sx = r.x - dst_x;
    const int sy = r.y - dst_y;
    cv::Mat dst = out(r);
    for (int yy = 0; yy < r.height; ++yy) {
        const cv::Vec4b* srow = src_bgra.ptr<cv::Vec4b>(sy + yy);
        cv::Vec3b* drow = dst.ptr<cv::Vec3b>(yy);
        for (int xx = 0; xx < r.width; ++xx) {
            const cv::Vec4b& s = srow[sx + xx];
            const int a = s[3];
            if (a == 0) continue;
            const int inv = 255 - a;
            cv::Vec3b& d = drow[xx];
            d[0] = static_cast<std::uint8_t>(s[0] + (d[0] * inv) / 255);
            d[1] = static_cast<std::uint8_t>(s[1] + (d[1] * inv) / 255);
            d[2] = static_cast<std::uint8_t>(s[2] + (d[2] * inv) / 255);
        }
    }
}

cv::Mat render_warning_icon_bgra(WarningZone zone, int size) {
    size = std::max(18, size);
    const int W = size + 2 * kIconCanvasPad;
    cv::Mat canvas = cv::Mat::zeros(W, W, CV_8UC4);
    const int cx = W / 2;
    const int cy = W / 2;
    const cv::Scalar& base = kZoneColorsBgr[static_cast<int>(zone)];
    const cv::Scalar color (base[0], base[1], base[2], 255);
    const cv::Scalar white (245, 245, 245, 255);
    const cv::Scalar shadow(  0,   0,   0, 255);
    const int r = std::max(8, size / 2);
    const int thick = std::max(2, size / 12);

    if (zone == WarningZone::Green) {
        cv::circle(canvas, {cx, cy}, r, shadow, thick + 2, cv::LINE_AA);
        cv::circle(canvas, {cx, cy}, r, color,  thick,     cv::LINE_AA);
        cv::ellipse(canvas, {cx, cy},
                    {static_cast<int>(r * 0.70), static_cast<int>(r * 0.70)},
                    0, 215, 320, color, std::max(1, thick - 1), cv::LINE_AA);
        cv::ellipse(canvas, {cx, cy},
                    {static_cast<int>(r * 0.42), static_cast<int>(r * 0.42)},
                    0, 220, 315, color, std::max(1, thick - 1), cv::LINE_AA);
        cv::circle(canvas, {cx, cy}, std::max(3, size / 12), white, -1, cv::LINE_AA);
        const double ang = -38.0 * CV_PI / 180.0;
        const cv::Point end(static_cast<int>(cx + r * 0.82 * std::cos(ang)),
                            static_cast<int>(cy + r * 0.82 * std::sin(ang)));
        cv::line(canvas, {cx, cy}, end, white, std::max(2, thick), cv::LINE_AA);
        cv::circle(canvas, end, std::max(2, size / 14), white, -1, cv::LINE_AA);
        return canvas;
    }

    std::vector<cv::Point> pts;
    if (zone == WarningZone::Orange) {
        pts = {
            {cx, cy - r},
            {cx - static_cast<int>(r * 1.06), cy + static_cast<int>(r * 0.86)},
            {cx + static_cast<int>(r * 1.06), cy + static_cast<int>(r * 0.86)},
        };
    } else {
        pts.reserve(8);
        for (int k = 0; k < 8; ++k) {
            const double a = (22.5 + k * 45.0) * CV_PI / 180.0;
            pts.emplace_back(static_cast<int>(cx + r * std::cos(a)),
                             static_cast<int>(cy + r * std::sin(a)));
        }
    }
    cv::polylines(canvas, pts, true, shadow, thick + 3, cv::LINE_AA);
    cv::polylines(canvas, pts, true, color,  thick + 1, cv::LINE_AA);
    cv::line(canvas, {cx, cy - static_cast<int>(r * 0.45)},
                     {cx, cy + static_cast<int>(r * 0.24)},
                     white, std::max(3, thick + 2), cv::LINE_AA);
    cv::circle(canvas, {cx, cy + static_cast<int>(r * 0.58)},
               std::max(3, thick + 1), white, -1, cv::LINE_AA);
    return canvas;
}

// Smooth quadratic-bezier turn arrow, shared by the cached sprite and the
// severity-colored panel draw. Replaces the old straight-segment L-shape whose
// right-angle bend looked kinked ("gay/gap khuc"). The head is aligned to the
// curve tangent so it joins the shaft seamlessly. Works on a BGR panel or a
// premultiplied BGRA sprite canvas — pass scalars matching the channel count.
void paint_curved_arrow(cv::Mat& canvas, cv::Point ctr, int size,
                        LaneKeepDir dir, const cv::Scalar& fill,
                        const cv::Scalar& shadow, const cv::Scalar& highlight,
                        bool with_highlight) {
    if (size <= 0) return;
    const float s = static_cast<float>(size);
    const float cx = static_cast<float>(ctr.x);
    const float cy = static_cast<float>(ctr.y);
    cv::Point2f p0, p1, p2;
    if (dir == LaneKeepDir::Straight) {
        p0 = {cx, cy + 0.36f * s};
        p1 = {cx, cy + 0.02f * s};
        p2 = {cx, cy - 0.26f * s};
    } else {
        const float sign = (dir == LaneKeepDir::Left) ? -1.0f : 1.0f;
        p0 = {cx - sign * 0.10f * s, cy + 0.36f * s};
        p1 = {cx - sign * 0.10f * s, cy - 0.04f * s};
        p2 = {cx + sign * 0.30f * s, cy - 0.24f * s};
    }
    constexpr int N = 24;
    std::vector<cv::Point> shaft;
    shaft.reserve(N);
    cv::Point2f prev{}, endp{};
    for (int i = 0; i < N; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(N - 1);
        const float u = 1.0f - t;
        const cv::Point2f p = p0 * (u * u) + p1 * (2.0f * u * t) + p2 * (t * t);
        shaft.emplace_back(cvRound(p.x), cvRound(p.y));
        if (i == N - 2) prev = p;
        if (i == N - 1) endp = p;
    }
    cv::Point2f tang = endp - prev;
    float tn = std::hypot(tang.x, tang.y);
    if (tn < 1e-3f) tn = 1.0f;
    tang = tang * (1.0f / tn);
    const cv::Point2f perp(-tang.y, tang.x);
    const cv::Point2f tip = endp + tang * (0.26f * s);
    const cv::Point2f bl = endp + perp * (0.16f * s);
    const cv::Point2f br = endp - perp * (0.16f * s);
    const cv::Point2f cc = (tip + bl + br) * (1.0f / 3.0f);
    auto grow = [&](const cv::Point2f& p) {
        const cv::Point2f q = cc + (p - cc) * 1.30f;
        return cv::Point(cvRound(q.x), cvRound(q.y));
    };
    const std::vector<cv::Point> tri = {
        {cvRound(tip.x), cvRound(tip.y)},
        {cvRound(bl.x), cvRound(bl.y)},
        {cvRound(br.x), cvRound(br.y)},
    };
    const std::vector<cv::Point> tri_shadow = {grow(tip), grow(bl), grow(br)};
    const int th = std::max(3, size / 9);
    cv::polylines(canvas, shaft, false, shadow, th + 4, cv::LINE_AA);
    cv::fillConvexPoly(canvas, tri_shadow, shadow, cv::LINE_AA);
    cv::polylines(canvas, shaft, false, fill, th, cv::LINE_AA);
    cv::fillConvexPoly(canvas, tri, fill, cv::LINE_AA);
    cv::circle(canvas, shaft.front(), th / 2 + 1, fill, -1, cv::LINE_AA);
    if (with_highlight) {
        cv::polylines(canvas, shaft, false, highlight, std::max(1, th / 4),
                      cv::LINE_AA);
    }
}

// Direct severity-colored arrow for the LDWS/LKAS panel (uncached — one small
// draw per frame). Dark shadow underlay gives contrast on the dark panel.
void draw_curved_arrow(cv::Mat& out, int cx, int cy, int size,
                       LaneKeepDir dir, const cv::Scalar& color) {
    paint_curved_arrow(out, {cx, cy}, size, dir, color,
                       cv::Scalar(20, 20, 20), cv::Scalar(), false);
}

// Sprites are keyed on size because render_overlay legitimately asks
// for two warning-icon sizes per frame (the larger HUD glyph + the
// smaller per-detection chip). A flat std::map keeps the lookup
// allocation-free after warmup, and the runtime touches at most ~3
// distinct sizes total per video.
struct IconCache {
    std::map<int, std::array<cv::Mat, 3>> warning_by_size;

    const cv::Mat& warning(WarningZone z, int size) {
        size = std::max(18, size);
        auto it = warning_by_size.find(size);
        if (it == warning_by_size.end()) {
            std::array<cv::Mat, 3> sprites;
            for (int i = 0; i < 3; ++i) {
                sprites[i] = render_warning_icon_bgra(
                    static_cast<WarningZone>(i), size);
            }
            it = warning_by_size.emplace(size, std::move(sprites)).first;
        }
        return it->second[static_cast<int>(z)];
    }
};

thread_local IconCache g_icon_cache;

void draw_warning_icon(cv::Mat& out, WarningZone zone, int cx, int cy, int size) {
    size = std::max(18, size);
    clamp_icon_center(out.cols, out.rows, size, &cx, &cy);
    const cv::Mat& sprite = g_icon_cache.warning(zone, size);
    // Sprite is (size + 2*pad)² with the icon centered; paste so the
    // icon center lands at (cx, cy).
    paste_bgra(out, sprite,
               cx - sprite.cols / 2,
               cy - sprite.rows / 2);
}


// FCWS panel blink cadence (used by draw_adas_panels). The blink phase is
// derived FPS-independently from the packet timestamp.
constexpr long long kTtcBlinkHalfMs = 250;    // 250 ms on / 250 ms off -> ~2 Hz

void draw_hud(cv::Mat& bgr, const BehaviorPacket& pkt, float display_fps, bool hailo_active) {
    std::ostringstream l1, l2;
    l1.precision(1);
    l1.setf(std::ios::fixed);
    l1 << "FPS=" << display_fps
       << "  steer=" << std::showpos << pkt.control.steering << std::noshowpos
       << "  thr=" << pkt.control.throttle
       << "  " << to_string(pkt.behavior_action);
    l2 << "frame=" << pkt.frame_id
       << "  hailo=" << (hailo_active ? "on" : "off")
       << "  " << pkt.behavior_reason;
    const int x = 12;
    const int y0 = 26;
    const int y1 = 50;
    cv::rectangle(bgr, cv::Rect(0, 0, bgr.cols, 60), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(bgr, l1.str(), {x, y0}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {255, 255, 255}, 1, cv::LINE_AA);
    cv::putText(bgr, l2.str(), {x, y1}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {200, 200, 200}, 1, cv::LINE_AA);

    // GPS readout (display-only): top-right of the HUD bar so it never
    // covers the corridor or the lower warning / lane-keep icons. Present
    // only when --gps is enabled; speed shown only on a valid fix.
    if (pkt.gps.has_value()) {
        const GpsHud& gp = *pkt.gps;
        std::ostringstream gspeed, gstat;
        gspeed.precision(1);
        gspeed.setf(std::ios::fixed);
        if (gp.valid) {
            gspeed << "GPS " << gp.speed_kmh << " km/h";
        } else {
            gspeed << "GPS -- km/h";
        }
        gstat.precision(1);
        gstat.setf(std::ios::fixed);
        gstat << "sats=" << gp.satellites
              << "  age=" << gp.age_s << "s"
              << "  " << gp.source;
        const cv::Scalar gcolor = gp.valid ? cv::Scalar(120, 255, 120)
                                           : cv::Scalar(150, 150, 150);
        const int gmargin = 12;
        int base = 0;
        const cv::Size sz_speed = cv::getTextSize(
            gspeed.str(), cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &base);
        const cv::Size sz_stat = cv::getTextSize(
            gstat.str(), cv::FONT_HERSHEY_SIMPLEX, 0.40, 1, &base);
        const int gx_speed = std::max(x, bgr.cols - sz_speed.width - gmargin);
        const int gx_stat = std::max(x, bgr.cols - sz_stat.width - gmargin);
        cv::putText(bgr, gspeed.str(), {gx_speed, y0},
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, gcolor, 1, cv::LINE_AA);
        cv::putText(bgr, gstat.str(), {gx_stat, y1},
                    cv::FONT_HERSHEY_SIMPLEX, 0.40, {180, 180, 180}, 1, cv::LINE_AA);
    }
}

// --- FCWS risk from the CONFIGURED safe following distance ------------------
// FCWS keys off the front object's measured distance vs the motor's caution
// distance — NOT behavior_action and NOT the old corridor color band. The
// caution distance is ego-speed tiered, mirroring behavior.cpp's distance
// ladder so the warning matches where the motor begins to slow. DISPLAY ONLY —
// reads the packet, changes no control state.
enum class FcwsRisk { Clear, Normal, Prompt, Warning };

// Caution-distance constants mirror FollowConfig (adas/behavior.hpp). Kept in
// sync by hand because render_overlay does not receive the motor's FollowConfig.
constexpr float kFcwsEgoFastKmh    = 40.0f;  // ego > this -> far caution distance
constexpr float kFcwsCautionNearM  = 10.0f;  // ego <= 40 km/h, or GPS no fix
constexpr float kFcwsCautionFarM   = 15.0f;  // ego > 40 km/h

// Motor caution distance (m) for the front object: ego > 40 km/h -> far, else
// (incl. no GPS fix) -> near. Mirrors behavior.cpp::caution_distance_for. No
// stationary special-case — the motor ladder is distance-only.
float fcws_caution_distance(const BehaviorPacket& pkt) {
    const bool ego_valid = pkt.gps.has_value() && pkt.gps->valid;
    const float ego_kmh = ego_valid ? static_cast<float>(pkt.gps->speed_kmh) : 0.0f;
    return (ego_valid && ego_kmh > kFcwsEgoFastKmh) ? kFcwsCautionFarM
                                                    : kFcwsCautionNearM;
}

struct FcwsSubject {
    const Detection* det{nullptr};
    bool strict_front{false};
};

FcwsSubject fcws_subject(const BehaviorPacket& pkt) {
    if (!pkt.objects.has_value()) return {};
    if (pkt.objects->front_object) return {&(*pkt.objects->front_object), true};
    return {};
}

bool same_detection_box(const Detection& d, const Detection& selected) {
    if (d.class_id != selected.class_id) return false;
    const cv::Rect2f& a = d.bbox_img;
    const cv::Rect2f& b = selected.bbox_img;
    const float x1 = std::max(a.x, b.x);
    const float y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.width, b.x + b.width);
    const float y2 = std::min(a.y + a.height, b.y + b.height);
    const float inter = std::max(0.0f, x2 - x1) *
                        std::max(0.0f, y2 - y1);
    const float denom = std::min(a.width * a.height, b.width * b.height);
    return denom > 1.0f && (inter / denom) > 0.90f;
}

FcwsRisk classify_fcws(const BehaviorPacket& pkt) {
    const FcwsSubject subject = fcws_subject(pkt);
    const Detection* fo = subject.det;
    if (fo == nullptr)
        return FcwsRisk::Clear;                       // no selected lead object -> grey
    if (!(fo->distance_display_valid && fo->display_distance_m >= 0.0f))
        return FcwsRisk::Prompt;                       // present but un-ranged -> amber
    const float caution = fcws_caution_distance(pkt);
    if (fo->display_distance_m > caution) return FcwsRisk::Normal;  // beyond caution -> green
    return subject.strict_front ? FcwsRisk::Warning : FcwsRisk::Clear;
}

// Map FCWS risk to the shared Green/Orange/Red glyph so the selected box, icon,
// and the FCWS panel all match.
std::optional<WarningZone> zone_for_fcws(FcwsRisk r) {
    switch (r) {
        case FcwsRisk::Warning: return WarningZone::Red;
        case FcwsRisk::Prompt:  return WarningZone::Orange;
        case FcwsRisk::Normal:  return WarningZone::Green;
        default:                return std::nullopt;
    }
}

// UI-only anti-flicker smoothers (render thread; thread_local like g_icon_cache).
// FCWS uses sticky-to-danger over 5 frames — the FcwsRisk enum order
// (Clear=0 < Normal=1 < Prompt=2 < Warning=3) IS the danger order, so a raw
// Warning shows instantly while a drop releases only on a 5-frame majority.
// Neither smoother reads or writes any motor/control state.
thread_local ui::StickyLevel<5> g_fcws_smoother;
thread_local ui::SteerStatus    g_steer_status;

// Round speed-limit sign glyph (EU style): white disc, red ring, km/h number.
// Pure OpenCV primitives, no asset. The number font auto-fits the disc.
void draw_speed_sign_icon(cv::Mat& out, int cx, int cy, int size, int kmh) {
    const int r = std::max(10, size / 2);
    cv::circle(out, {cx, cy}, r, cv::Scalar(245, 245, 245), -1, cv::LINE_AA);  // white fill
    cv::circle(out, {cx, cy}, r, cv::Scalar(40, 40, 225),
               std::max(3, size / 8), cv::LINE_AA);                            // red ring
    const std::string num = std::to_string(kmh);
    double fs = size * 0.030; int b = 0;
    cv::Size ts = cv::getTextSize(num, cv::FONT_HERSHEY_SIMPLEX, fs, 2, &b);
    while (ts.width > static_cast<int>(r * 1.35) && fs > 0.3) {
        fs -= 0.05;
        ts = cv::getTextSize(num, cv::FONT_HERSHEY_SIMPLEX, fs, 2, &b);
    }
    cv::putText(out, num, {cx - ts.width / 2, cy + ts.height / 2},
                cv::FONT_HERSHEY_SIMPLEX, fs, cv::Scalar(20, 20, 20), 2, cv::LINE_AA);
}

// ---------------------------------------------------------------------------
// ADAS-style side panels (LDWS/LKAS, LANE, FCWS). DISPLAY ONLY — reads the
// packet + the resolved FcwsRisk; never mutates control/perception. Every glyph
// is an OpenCV primitive or self-drawn sprite (no 3rd-party assets). Layout
// (tuned for 1280x720); LDWS/LKAS and FCWS share one compact height:
//   bottom-left  : LDWS/LKAS status + smooth arrow
//   right top    : mini lane/corridor (clean, no text)
//   bottom-right : FCWS risk (Normal/Prompt/Warning/Clear) + distance/TTC/front
// ---------------------------------------------------------------------------
void draw_adas_panels(cv::Mat& out, const BehaviorPacket& pkt, FcwsRisk fcws) {
    const int W = out.cols, H = out.rows;
    const int dim = std::min(W, H);
    const int margin = std::max(10, dim / 64);
    const double F = std::clamp(dim / 720.0, 0.7, 1.6);   // text scale factor
    const cv::Scalar kInk(235, 235, 235), kDim(165, 165, 165);
    const cv::Scalar kAccent(255, 200, 0);                // cyan title (BGR)
    const cv::Scalar kGreen(70, 200, 90), kAmber(0, 175, 255), kRed(40, 40, 255);

    auto put = [&](const std::string& t, int x, int y, double s,
                   const cv::Scalar& c, int th = 1) {
        cv::putText(out, t, {x, y}, cv::FONT_HERSHEY_SIMPLEX, s * F, c, th,
                    cv::LINE_AA);
    };
    auto line_h = [&](double s) { return static_cast<int>(28 * s * F); };

    const bool  lane_valid = pkt.lane.has_value() && pkt.lane->geometry.valid;
    const float steer = pkt.control.steering;
    const int pw = std::clamp(static_cast<int>(W * 0.20), 190, 320);
    // LDWS/LKAS (bottom-left) and FCWS (bottom-right) share one compact height
    // so they read as a matched pair sitting low, near the arrow.
    const int panel_h = std::clamp(static_cast<int>(H * 0.27), 150, 240);

    // ===== TOP-LEFT: speed-limit sign + throttle ceiling (below the HUD bar) =====
    // Reads pkt.speed_limit (set by FollowController) — never recomputed here.
    {
        const int pwid = std::clamp(static_cast<int>(W * 0.16), 180, 220);
        const int phgt = std::clamp(static_cast<int>(H * 0.15), 90, 120);
        const int x = std::max(12, static_cast<int>(std::lround(W * 0.0156)));  // ~20 @1280
        const int y = std::max(64, static_cast<int>(std::lround(H * 0.097)));   // ~70 @720, below HUD bar
        blend_dark_panel(out, x, y, x + pwid, y + phgt, 0.55);

        const SpeedLimitHud& sl = pkt.speed_limit;
        const bool held = (sl.source == SpeedLimitHud::Source::Held);
        const int isz = std::min(phgt - 24, 58);
        draw_speed_sign_icon(out, x + 14 + isz / 2, y + phgt / 2, isz, sl.kmh);

        // No SIGN text: the round icon already shows the active limit (incl. the
        // default 60). Two lines: the ceiling, then the throttle actually sent.
        std::ostringstream lmax, lthr;
        lmax.setf(std::ios::fixed); lmax.precision(1); lmax << "MAX THR " << sl.cap;
        lthr.setf(std::ios::fixed); lthr.precision(1); lthr << "THR " << pkt.control.throttle;

        // Auto-fit each line to the text column (panel width is clamped while the
        // font scale F is not, so fit defensively at high resolutions).
        const int tx = x + 14 + isz + 10;
        const int tw = (x + pwid) - tx - 8;
        auto fit = [&](const std::string& s) {
            double fs = 0.5 * F; int b = 0;
            while (fs > 0.32 &&
                   cv::getTextSize(s, cv::FONT_HERSHEY_SIMPLEX, fs, 1, &b).width > tw)
                fs -= 0.02;
            return fs;
        };
        const cv::Scalar amber(0, 175, 255);
        // Held -> amber MAX THR line + thin amber border (no "HELD" text).
        cv::putText(out, lmax.str(), {tx, y + phgt / 2 - 4}, cv::FONT_HERSHEY_SIMPLEX,
                    fit(lmax.str()), held ? amber : kInk, 1, cv::LINE_AA);
        cv::putText(out, lthr.str(), {tx, y + phgt / 2 + line_h(0.8)},
                    cv::FONT_HERSHEY_SIMPLEX, fit(lthr.str()), kInk, 1, cv::LINE_AA);
        if (held)
            cv::rectangle(out, {x, y}, {x + pwid - 1, y + phgt - 1}, amber, 1, cv::LINE_AA);
    }

    // ===== BOTTOM-LEFT: LDWS / LKAS (compact, matched to FCWS) =====
    {
        const int x = margin, y = H - panel_h - margin;
        blend_dark_panel(out, x, y, x + pw, y + panel_h, 0.55);
        const int title_y  = y + line_h(0.95);
        const int status_y = title_y + line_h(0.95);
        put("LDWS / LKAS", x + 12, title_y, 0.5, kAccent, 1);

        // UI-only Schmitt hysteresis around the 0.10 / 0.22 |steering| thresholds
        // so the displayed status does not flip when steer hovers at a boundary.
        // Reads control.steering but NEVER writes it (servo command unchanged).
        // Reset on lane loss so a stale TURN is not held.
        //   level 0 = LANE KEEP (green), 1 = TURN (amber), 2 = departure (red).
        std::string status; cv::Scalar col;
        LaneKeepDir adir = LaneKeepDir::Straight;
        if (!lane_valid) {
            g_steer_status.reset();
            status = "LANE LOST"; col = kDim;
        } else {
            int lv = 0, d = 0;
            g_steer_status.update(steer, 0.10f, 0.22f, 0.03f, &lv, &d);
            const LaneKeepDir dir = (d < 0) ? LaneKeepDir::Left
                                  : (d > 0) ? LaneKeepDir::Right
                                            : LaneKeepDir::Straight;
            if (lv == 0) {
                status = "LANE KEEP"; col = kGreen;
            } else {
                status = (dir == LaneKeepDir::Left) ? "TURN LEFT" : "TURN RIGHT";
                col = (lv >= 2) ? kRed : kAmber;
                adir = dir;
            }
        }
        put(status, x + 12, status_y, 0.6, col, 2);
        // Smooth curved arrow, status-colored, filling the panel below the text.
        if (lane_valid) {
            const int top = status_y + line_h(0.35);
            const int avail = (y + panel_h - 10) - top;
            const int isz = std::clamp(std::min(avail, pw - 48), 40, 120);
            draw_curved_arrow(out, x + pw / 2, top + avail / 2, isz, adir, col);
        }
    }

    // ===== RIGHT-TOP: LANE / CORRIDOR / STEERING =====
    {
        const int ph = std::clamp(static_cast<int>(H * 0.25), 140, 230);
        const int x = W - pw - margin;
        const int y = std::max(margin, static_cast<int>(H * 0.10));  // below top bar
        blend_dark_panel(out, x, y, x + pw, y + ph, 0.55);
        int ty = y + line_h(1.05);
        put("PATH", x + 12, ty, 0.55, kAccent, 2);
        const int sx = x + 16, sw = pw - 32;
        // Mini lane uses the full panel below the title now that the VALID /
        // STEER text is gone — clean lane image only.
        const int sy = ty + 10, sh = (y + ph) - sy - 12;
        const cv::Scalar yel(0, 210, 210), red(0, 0, 255);

        // Mini forward-view uses ONLY the red centerline (geometry.fitted_pts_img).
        // The two yellow guide rails are UI-only PERSPECTIVE offsets from that red
        // path — never the corridor (left_corridor/right_corridor) or its polygon.
        // FCWS uses the strict lane gate elsewhere; this panel reads no corridor.
        auto map_red = [&](const std::vector<cv::Point2f>& src,
                           float ymin, float ymax) {
            std::vector<cv::Point2f> dst;
            dst.reserve(src.size());
            const float yspan = std::max(1.0f, ymax - ymin);
            for (const auto& p : src) {
                const float nx = (p.x - W * 0.5f) / (W * 0.5f);   // ~[-1, 1]
                const float ny = (p.y - ymin) / yspan;            // 0 far -> 1 near
                dst.emplace_back(
                    std::clamp(sx + sw * 0.5f + nx * sw * 0.62f,
                               static_cast<float>(sx), static_cast<float>(sx + sw)),
                    std::clamp(sy + ny * sh, static_cast<float>(sy),
                               static_cast<float>(sy + sh)));
            }
            return dst;
        };
        // Yellow rails = red center offset HORIZONTALLY by a depth-varying
        // half-width: wide near the ego (panel bottom), narrowing toward the far
        // end (panel top). Each rail point is center.x +/- half at the SAME row,
        // so the rails follow the red center's curve and converge with
        // perspective instead of standing as parallel verticals.
        const float near_half = std::clamp(sw * 0.30f, 20.0f, 90.0f);  // panel bottom (near)
        const float far_half  = std::clamp(sw * 0.09f,  8.0f, 32.0f);  // panel top (far)
        auto rail = [&](const std::vector<cv::Point2f>& center, float sign) {
            std::vector<cv::Point2f> pts;
            pts.reserve(center.size());
            const float span = std::max(1.0f, static_cast<float>(sh));
            for (const auto& p : center) {
                const float ny = std::clamp((p.y - sy) / span, 0.0f, 1.0f);  // 0 far -> 1 near
                const float half = far_half + (near_half - far_half) * ny;
                pts.emplace_back(
                    std::clamp(p.x + sign * half, static_cast<float>(sx),
                               static_cast<float>(sx + sw)),
                    p.y);
            }
            return pts;
        };
        auto draw_mini_path = [&](const std::vector<cv::Point2f>& center) {
            draw_polyline(out, rail(center, -1.0f), yel, 2);   // left  rail
            draw_polyline(out, rail(center, +1.0f), yel, 2);   // right rail
            draw_polyline(out, center, red, 2);                // red center on top
        };

        bool drew = false;
        if (lane_valid) {
            const auto& g = pkt.lane->geometry;
            if (g.fitted_pts_img.size() >= 2) {
                float ymin = 1e9f, ymax = -1e9f;
                for (const auto& p : g.fitted_pts_img) {
                    ymin = std::min(ymin, p.y);
                    ymax = std::max(ymax, p.y);
                }
                draw_mini_path(map_red(g.fitted_pts_img, ymin, ymax));
                drew = true;
            }
        }
        if (!drew) {
            // Fallback: synthesize a straight vertical red center at panel center
            // and run the SAME depth-varying rail builder, so even without lane
            // geometry the mini path is a converging trapezoid (never two
            // hard-parallel verticals).
            std::vector<cv::Point2f> center;
            const float cxp = static_cast<float>(sx) + sw * 0.5f;
            constexpr int kFallbackPts = 8;
            for (int i = 0; i < kFallbackPts; ++i) {
                const float t = static_cast<float>(i) /
                                static_cast<float>(kFallbackPts - 1);
                center.emplace_back(cxp, sy + t * static_cast<float>(sh));
            }
            draw_mini_path(center);
        }
        // Fixed ego reference marker at panel-bottom-center.
        const int egx = sx + sw / 2, egy = sy + sh;
        const std::vector<cv::Point> ego = {
            {egx, egy - 9}, {egx - 7, egy + 2}, {egx + 7, egy + 2}};
        cv::fillConvexPoly(out, ego, cv::Scalar(235, 235, 235), cv::LINE_AA);
    }

    // ===== BOTTOM-RIGHT: FCWS (driven by the final motor decision) =====
    {
        const int ph = panel_h, x = W - pw - margin, y = H - ph - margin;
        blend_dark_panel(out, x, y, x + pw, y + ph, 0.55);
        int ty = y + line_h(1.05);
        put("FCWS", x + 12, ty, 0.6, kAccent, 2); ty += line_h(0.5);
        std::string risk; cv::Scalar rc; bool warn = false;
        switch (fcws) {
            case FcwsRisk::Warning: risk = "Warning Risk"; rc = kRed;   warn = true; break;
            case FcwsRisk::Prompt:  risk = "Prompt Risk";  rc = kAmber;              break;
            case FcwsRisk::Normal:  risk = "Normal Risk";  rc = kGreen;              break;
            case FcwsRisk::Clear: default: risk = "Clear"; rc = kDim;                break;
        }
        const std::optional<WarningZone> icon_zone = zone_for_fcws(fcws);
        const int isz = std::clamp(static_cast<int>(ph * 0.34), 34, 76);
        if (icon_zone) draw_warning_icon(out, *icon_zone, x + 12 + isz / 2, ty + isz / 2, isz);
        put(risk, x + 20 + isz, ty + static_cast<int>(isz * 0.5), 0.56, rc, 2);
        ty += isz + line_h(0.9);
        float fc_d = -1; bool fc_tv = false; float fc_t = 0; bool fc_vv = false; float fc_v = 0;
        if (const Detection* fo = fcws_subject(pkt).det) {
            if (fo->distance_display_valid && fo->display_distance_m >= 0.0f) fc_d = fo->display_distance_m;
            if (fo->ttc_valid)         { fc_tv = true; fc_t = fo->ttc_s; }
            if (fo->front_speed_valid) { fc_vv = true; fc_v = fo->front_speed_kmh; }
        }
        std::ostringstream d, t, v;
        d.setf(std::ios::fixed); d.precision(1);
        if (fc_d >= 0) d << "DIST   " << fc_d << " m"; else d << "DIST   -- m";
        t.setf(std::ios::fixed); t.precision(1);
        // TTC is signed: positive = closing (time to collision), negative =
        // opening (gap growing). "(open)" tag makes the negative unambiguous.
        if (fc_tv) {
            t << "TTC    " << fc_t << " s";
            if (fc_t < 0.0f) t << " (open)";
        } else {
            t << "TTC    -- s";
        }
        v.setf(std::ios::fixed); v.precision(0);
        if (fc_vv) v << "FRONT  " << fc_v << " km/h"; else v << "FRONT  -- km/h";
        // Red alert ONLY when actually closing (positive TTC under threshold) —
        // an opening gap (negative TTC) must never flash danger.
        const bool ttc_alert = fc_tv && fc_t > 0.0f && fc_t < 2.0f;
        put(d.str(), x + 14, ty, 0.5, kInk); ty += line_h(0.85);
        put(t.str(), x + 14, ty, 0.5, ttc_alert ? kRed : kInk, ttc_alert ? 2 : 1); ty += line_h(0.85);
        put(v.str(), x + 14, ty, 0.5, kInk);
        if (warn) {
            const long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                pkt.timestamp.time_since_epoch()).count();
            if (((ms / kTtcBlinkHalfMs) % 2) == 0)
                cv::rectangle(out, {x, y}, {x + pw, y + ph}, kRed, 3, cv::LINE_AA);
        }
    }
}

}  // namespace

cv::Mat render_overlay(const BehaviorPacket& pkt, float display_fps, const OverlayConfig& cfg) {
    cv::Mat out = pkt.frame.clone();

    // Forward-collision risk = strict in-corridor front_object distance vs the
    // configured safe following distance. Lane-edge speed_object detections can
    // still display their own labels, but they do not enter the FCWS panel.
    // Drives BOTH the selected box/icon and the FCWS panel so they stay
    // consistent (no behavior_action, no corridor band).
    // UI-only sticky-to-danger smoothing: a raw Warning shows INSTANTLY; a drop
    // to Normal/Clear (e.g. front object lost for 1-2 frames) releases only after
    // a 5-frame majority, so the warning does not flicker. Does NOT affect motor.
    const FcwsRisk raw_fcws = classify_fcws(pkt);
    const FcwsRisk fcws_risk =
        static_cast<FcwsRisk>(g_fcws_smoother.update(static_cast<int>(raw_fcws)));

    if (pkt.lane.has_value()) {
        const auto& lp = *pkt.lane;
        if (!lp.lane_area.empty()) {
            // Translucent green lane area fill.
            blend_mask_color(out, lp.lane_area, cv::Scalar(110, 220, 90), 0.30);
        }
        if (!cfg.fast_overlay && !lp.line_union.empty()) {
            // Subtle line marking highlight.
            blend_mask_color(out, lp.line_union, cv::Scalar(255, 255, 0), 0.55);
        }
        // The large on-road drive corridor (yellow edges + red centerline +
        // collision-zone bands + anchor dot) is intentionally NOT drawn on the
        // road anymore: it was moved into the right-top LANE panel so the scene
        // stays clean and never occludes objects ahead. The corridor GEOMETRY
        // is still computed (lp.geometry) and still feeds the warning-zone /
        // front-object logic below — only its on-road rendering was removed.
        // The green lane_area fill + line markings above remain as the
        // perception layer. To restore the on-road corridor for debugging, draw
        // lp.geometry.left_corridor / right_corridor / fitted_pts_img here.
    }

    // Per-detection boxes. In-corridor objects get a pixel-fraction tint; the
    // selected FCWS object additionally gets a centered, metric-distance
    // warning icon + matching box color so it agrees with the warning HUD.
    // Motor hard-stop still uses the stricter front_object path. Cf.
    // src/pipeline/postprocessor.py::draw_detections — note the
    // intentional divergence: C++ draws the metric warning icon for the front
    // object only (centered), where Python draws a pixel-fraction icon per box.
    const int dim_for_icon = std::min(out.cols, out.rows);
    const int det_icon_size = std::clamp(
        static_cast<int>(std::lround(dim_for_icon * 0.055)), 30, 52);
    // Traffic signs use a distinct color and skip the collision-zone icon —
    // they're informational and never trigger STOP/CAUTION, so coloring them
    // by drive-corridor band would mislead the operator.
    const cv::Scalar kSignColorBgr(255, 0, 255);  // magenta
    if (pkt.objects.has_value()) {
        const Detection* fcws_obj = fcws_subject(pkt).det;
        const Detection* front_speed = fcws_obj;
        auto is_front_speed_box = [&](const Detection& d) {
            if (front_speed == nullptr ||
                (!front_speed->front_speed_valid &&
                 !front_speed->relative_speed_valid)) {
                return false;
            }
            return same_detection_box(d, *front_speed);
        };
        // Identity match of a raw detection against the selected FCWS object,
        // without the speed-valid gate.
        auto is_fcws_object_box = [&](const Detection& d) {
            return fcws_obj != nullptr && same_detection_box(d, *fcws_obj);
        };
        for (const auto& d : pkt.objects->detections) {
            const bool is_sign = yolo26_is_traffic_sign(d.class_id);
            const float foot_x = d.bbox_img.x + d.bbox_img.width * 0.5f;
            const float foot_y = d.bbox_img.y + d.bbox_img.height;
            // The selected FCWS object's box + icon follow the FCWS risk
            // (distance vs configured safe distance), NOT the old corridor color
            // band, so the box, the centered icon, and the FCWS panel all agree:
            // within safe distance -> red + warning icon; beyond -> green.
            const bool is_front = !is_sign && is_fcws_object_box(d);
            std::optional<WarningZone> front_overlay_zone;
            if (is_front) front_overlay_zone = zone_for_fcws(fcws_risk);
            cv::Scalar color;
            if (is_sign) {
                color = kSignColorBgr;
            } else if (front_overlay_zone) {
                color = kZoneColorsBgr[static_cast<int>(*front_overlay_zone)];
            } else {
                color = cv::Scalar(0, 255, 255);
            }
            cv::rectangle(out, d.bbox_img, color, 2);
            cv::circle(out, {cvRound(foot_x), cvRound(foot_y)}, 3, color, -1, cv::LINE_AA);
            // Warning icon: selected FCWS object only, centered inside the bbox and
            // sized to fit so it stays within the box and clear of the
            // top-left label/distance text. Skipped when the box is too small
            // to contain even the 18px minimum glyph — the zone-colored box
            // still conveys the warning without the icon spilling outside (R4).
            const int icon_fit = static_cast<int>(
                std::min(d.bbox_img.width, d.bbox_img.height));
            if (front_overlay_zone && icon_fit >= 22) {
                const int icon_cx =
                    cvRound(d.bbox_img.x + d.bbox_img.width * 0.5f);
                const int icon_cy =
                    cvRound(d.bbox_img.y + d.bbox_img.height * 0.5f);
                const int icon_sz = std::clamp(det_icon_size, 18, icon_fit - 4);
                draw_warning_icon(out, *front_overlay_zone,
                                  icon_cx, icon_cy, icon_sz);
            }
            std::ostringstream tag;
            if (is_sign) {
                // Prefix so the operator can spot signs at a glance even
                // when the magenta box overlaps a vehicle box.
                tag << "sign:" << d.label << " "
                    << static_cast<int>(d.score * 100) << "%";
            } else {
                tag << d.label << " " << static_cast<int>(d.score * 100) << "%";
            }
            if (!is_sign && d.pose_valid && d.distance_display_valid &&
                d.display_distance_m >= 0.0f) {
                std::ostringstream dist;
                dist.setf(std::ios::fixed);
                dist.precision(1);
                dist << " " << d.display_distance_m << "m";
                tag << dist.str();
            }
            if (!is_sign && is_front_speed_box(d)) {
                if (front_speed->front_speed_valid) {
                    const int kmh = static_cast<int>(
                        std::lround(front_speed->front_speed_kmh));
                    tag << " v=" << kmh << "km/h";
                } else {
                    tag << " v=--km/h";
                }
            }
            cv::putText(out, tag.str(),
                        {static_cast<int>(d.bbox_img.x),
                         std::max(18, static_cast<int>(d.bbox_img.y) - 8)},
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
        }
    }

    if (cfg.show_hud) {
        draw_hud(out, pkt, display_fps, cfg.hailo_active);
    }
    // ADAS-style side panels (LDWS/LKAS bottom-left, mini LANE top-right, FCWS
    // bottom-right). Display only; FCWS + selected object box/icon are driven
    // by the resolved fcws_risk (= distance vs configured safe following distance).
    if (cfg.show_warning_hud) {
        draw_adas_panels(out, pkt, fcws_risk);
    }
    return out;
}

cv::Mat resize_for_display(const cv::Mat& src, int target_w) {
    if (target_w <= 0 || src.empty()) return src;
    if (src.cols == target_w) return src;
    int target_h = src.rows * target_w / std::max(src.cols, 1);
    cv::Mat dst;
    cv::resize(src, dst, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);
    return dst;
}

}  // namespace adas
