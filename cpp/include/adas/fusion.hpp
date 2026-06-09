#pragma once

// Partial fusion / control mirror target:
//   src/pipeline/fusion_worker.py  (drain_state_queues, emit_fused,
//                                   safety_supervisor)
//
// fuse() pairs the LanePacket with the freshest ObjectPacket where
// object.frame_id <= lane.frame_id within the stale-tolerance window. A
// nullptr object packet (pipeline warmup / stale cache) builds the lane-only
// BehaviorPacket.

#include <optional>

#include "adas/types.hpp"

namespace adas {

constexpr int kStaleFrameTolerance = 8;

// `objects` is a borrowed pointer into the lane worker's shared object-packet
// cache (nullptr when no fresh object is in tolerance). fuse() makes exactly
// one mutable copy of it to stamp front_object/speed_object into bp.objects.
// TODO(adas-cpp/full): port full safety_supervisor/control behavior.
BehaviorPacket fuse(const LanePacket& lane, const ObjectPacket* objects);

// (fuse_object_driven was removed with the alternate scheduler — the lane
// worker drives every render packet in the parallel schedule.)

}  // namespace adas
