#pragma once
#include <optional>

#include "adas/types.hpp"

namespace adas {

// Defined in object_detection.hpp; held by the runtime/fusion path so the
// layered front-object gate can reuse the last valid NEAR-LANE WIDTH across
// frames. Passed by reference (incomplete type is fine for a reference param).
class FrontObjectNearLaneMemory;

constexpr int kStaleFrameTolerance = 8;
BehaviorPacket fuse(const LanePacket& lane, const ObjectPacket* objects,
                    FrontObjectNearLaneMemory& near_mem);

}  // namespace adas
