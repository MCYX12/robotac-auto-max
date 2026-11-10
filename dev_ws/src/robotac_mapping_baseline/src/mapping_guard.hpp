#ifndef ROBOTAC_MAPPING_GUARD_HPP_
#define ROBOTAC_MAPPING_GUARD_HPP_
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
namespace robotac_mapping_baseline {
struct CloudFieldLayout {std::string name; uint32_t offset; uint8_t datatype; uint32_t count;};
inline std::string cloud_layout_problem(
  uint32_t width, uint32_t height, uint32_t point_step, uint32_t row_step,
  std::size_t bytes, bool big_endian, const std::vector<CloudFieldLayout> & fields)
{
  const uint16_t endian_probe = 1;
  const bool host_big_endian = *reinterpret_cast<const uint8_t *>(&endian_probe) == 0;
  if (big_endian != host_big_endian) {return "unsupported_point_byte_order";}
  if (width == 0 || height == 0 || point_step < 12) {return "empty_or_invalid_point_stride";}
  if (uint64_t(width) * point_step > row_step || uint64_t(row_step) * height != bytes) {
    return "point_cloud_byte_length_mismatch";
  }
  std::vector<uint32_t> offsets;
  for (const auto * axis : {"x", "y", "z"}) {
    const CloudFieldLayout * found = nullptr;
    for (const auto & field : fields) {
      if (field.name == axis) {
        if (found != nullptr) {return "duplicate_xyz_field";}
        found = &field;
      }
    }
    if (found == nullptr || found->datatype != 7 || found->count != 1 ||
      uint64_t(found->offset) + sizeof(float) > point_step) {return "invalid_xyz_float32_layout";}
    for (const auto other : offsets) {
      if (found->offset < uint64_t(other)+4 && other < uint64_t(found->offset)+4) {
        return "overlapping_xyz_fields";
      }
    }
    offsets.push_back(found->offset);
  }
  return "";
}
inline std::string pose_continuity_problem(
  double dt, double distance, double angle, double distance_allowance,
  double maximum_speed, double angle_allowance, double maximum_angular_speed)
{
  if (!std::isfinite(dt) || !std::isfinite(distance) || !std::isfinite(angle)) {
    return "nonfinite_pose_difference";
  }
  // Long gaps are not themselves faults: permit physically plausible travel.
  // Interpolation remains independently limited by max_odom_time_offset.
  if (dt < -0.5) {return "odometry_clock_reset";}
  if (dt <= 0.0) {return "";}  // Small packet reordering is discarded by caller.
  if (distance > distance_allowance + maximum_speed * dt) {return "odometry_position_jump";}
  if (angle > angle_allowance + maximum_angular_speed * dt) {return "odometry_orientation_jump";}
  return "";
}
}
#endif
