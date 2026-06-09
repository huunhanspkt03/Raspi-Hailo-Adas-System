#pragma once

// Display-severity glyph level for the operator HUD. The overlay maps its
// FCWS risk to this 3-level enum to pick the warning-icon color (green /
// orange / red). Motor control is distance-only (see FollowController) and
// does NOT use this enum — it is purely a rendering classification now.

namespace adas {

enum class WarningZone { Green = 0, Orange = 1, Red = 2 };

}  // namespace adas
