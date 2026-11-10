#ifndef ROBOTAC_VOXEL_CONFIDENCE_HPP_
#define ROBOTAC_VOXEL_CONFIDENCE_HPP_
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
namespace robotac_mapping_baseline {
// Fixed, origin-zero evidence grid; this never changes stored point positions.
inline constexpr double kTemporalPatchSize = 0.10;
// Evidence counts sensor frames, never the number of rays in one frame. A
// repeated timestamp cannot become independent evidence under another frame ID.
struct ObservationSupport {
  uint32_t distinct_frames{0};
  std::size_t last_frame{std::numeric_limits<std::size_t>::max()};
  int64_t first_stamp_ns{0};
  int64_t last_stamp_ns{0};
  void observe(std::size_t frame, int64_t stamp_ns) {
    if (stamp_ns <= 0 || frame == last_frame ||
      (distinct_frames > 0 && stamp_ns <= last_stamp_ns)) {return;}
    if (distinct_frames == 0) {first_stamp_ns = stamp_ns;}
    ++distinct_frames;
    last_frame = frame;
    last_stamp_ns = stamp_ns;
  }
  double span() const {
    return distinct_frames > 0 ? static_cast<double>(last_stamp_ns - first_stamp_ns) * 1e-9 : 0.0;
  }
  bool trusted(uint32_t minimum_frames, double minimum_span) const {
    // Subtract integer ROS timestamps before conversion: subtracting epoch-sized
    // doubles can turn exactly 0.3 s into 0.299999952 s and reject valid support.
    return distinct_frames >= minimum_frames &&
      static_cast<double>(last_stamp_ns - first_stamp_ns) >= minimum_span * 1e9;
  }
};
enum class ConfidenceReason : std::size_t {DISABLED=0, GROUND=1, SELF=2, NEIGHBOR=3, REJECTED=4, PATCH=5};
}
#endif
