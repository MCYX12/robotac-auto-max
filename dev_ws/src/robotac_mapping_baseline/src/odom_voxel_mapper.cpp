#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include "mapping_guard.hpp"
#include "voxel_confidence.hpp"
#include "scan_surface_filter.hpp"
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <nav_msgs/msg/odometry.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace robotac_mapping_baseline
{

double stamp_seconds(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
}

struct VoxelKey
{
  int64_t x;
  int64_t y;
  int64_t z;

  bool operator==(const VoxelKey & other) const
  {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelKeyHash
{
  std::size_t operator()(const VoxelKey & key) const
  {
    std::size_t seed = std::hash<int64_t>{}(key.x);
    seed ^= std::hash<int64_t>{}(key.y) + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
    seed ^= std::hash<int64_t>{}(key.z) + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
    return seed;
  }
};

struct VoxelAccumulator
{
  Eigen::Vector3d sum{Eigen::Vector3d::Zero()};
  uint32_t count{0};
  ObservationSupport observation;
  uint32_t lidar_measurements{0};
  uint32_t depth_measurements{0};

  void add(const Eigen::Vector3d & point, const std::size_t scan, const int64_t stamp_ns,
    const bool depth_source)
  {
    sum += point;
    ++count;
    if (depth_source) {++depth_measurements;} else {++lidar_measurements;}
    observation.observe(scan, stamp_ns);
  }

  Eigen::Vector3d centroid() const
  {
    return sum / static_cast<double>(count);
  }
};

struct OdomSample
{
  double stamp{0.0};
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
  double yaw_rate{0.0};
  double linear_speed{0.0};
  double angular_speed{0.0};
};

class OdomVoxelMapper : public rclcpp::Node
{
public:
  OdomVoxelMapper()
  : Node("odom_voxel_mapper")
  {
    lidar_topic_ = declare_parameter<std::string>("lidar_topic", "/front_lidar");
    const auto depth_ground_topic = declare_parameter<std::string>("depth_ground_topic", "");
    odom_topic_ = declare_parameter<std::string>("odom_topic", "/odom/mujoco_odom");
    map_topic_ = declare_parameter<std::string>("map_topic", "/robotac/odom_map");
    preview_topic_ = declare_parameter<std::string>(
      "preview_topic", "/robotac/odom_map_preview");
    save_service_name_ = declare_parameter<std::string>("save_service", "/robotac/save_odom_map");
    save_path_ = declare_parameter<std::string>(
      "save_path", "/workspace/dev_ws/maps/odom_baseline.pcd");
    map_frame_ = declare_parameter<std::string>("map_frame", "world");
    voxel_size_ = declare_parameter<double>("voxel_size", 0.035);
    min_range_ = declare_parameter<double>("min_range", 0.20);
    max_range_ = declare_parameter<double>("max_range", 40.0);
    max_odom_time_offset_ = declare_parameter<double>("max_odom_time_offset", 0.02);
    lidar_pose_time_offset_ = declare_parameter<double>("lidar_pose_time_offset", 0.0);
    odom_jump_distance_allowance_ = declare_parameter<double>("odom_jump_distance_allowance", 0.25);
    odom_maximum_speed_ = declare_parameter<double>("odom_maximum_speed", 3.0);
    odom_jump_angle_allowance_ = declare_parameter<double>("odom_jump_angle_allowance", 0.35);
    odom_maximum_angular_speed_ = declare_parameter<double>("odom_maximum_angular_speed", 4.0);
    adaptive_ground_timing_ = declare_parameter<bool>("adaptive_ground_timing", false);
    stationary_mapping_ = declare_parameter<bool>("stationary_mapping", false);
    stationary_linear_limit_ = declare_parameter<double>("stationary_linear_limit", 0.04);
    stationary_angular_limit_ = declare_parameter<double>("stationary_angular_limit", 0.08);
    stationary_settle_seconds_ = declare_parameter<double>("stationary_settle_seconds", 1.0);
    process_every_n_scans_ = declare_parameter<int>("process_every_n_scans", 1);
    world_z_min_ = declare_parameter<double>("world_z_min", -1.0);
    world_z_max_ = declare_parameter<double>("world_z_max", 5.0);
    publish_every_n_scans_ = declare_parameter<int>("publish_every_n_scans", 10);
    preview_every_n_scans_ = declare_parameter<int>("preview_every_n_scans", 10);
    preview_max_points_ = declare_parameter<int>("preview_max_points", 120000);
    preview_z_min_ = declare_parameter<double>("preview_z_min", -1.0e6);
    preview_z_max_ = declare_parameter<double>("preview_z_max", 1.0e6);
    preview_minimum_voxel_observations_ = declare_parameter<int>(
      "preview_minimum_voxel_observations", 1);
    checkpoint_every_n_scans_ = declare_parameter<int>("checkpoint_every_n_scans", 300);
    self_filter_enabled_ = declare_parameter<bool>("self_filter_enabled", true);
    ground_snap_enabled_ = declare_parameter<bool>("ground_snap_enabled", false);
    ground_plane_z_ = declare_parameter<double>("ground_plane_z", 0.0);
    ground_snap_tolerance_ = declare_parameter<double>("ground_snap_tolerance", 0.10);
    maximum_mapping_yaw_rate_ = declare_parameter<double>(
      "maximum_mapping_yaw_rate", 0.30);
    native_confidence_filter_enabled_ = declare_parameter<bool>(
      "native_confidence_filter_enabled", false);
    native_temporal_support_enabled_ = declare_parameter<bool>("native_temporal_support_enabled", false);
    native_temporal_patch_enabled_ = declare_parameter<bool>("native_temporal_patch_enabled", false);
    scan_surface_filter_enabled_ = declare_parameter<bool>("scan_surface_filter_enabled", false);
    const auto filtered_topic = declare_parameter<std::string>("filtered_lidar_topic", "");
    if (!filtered_topic.empty()) {
      filtered_lidar_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(filtered_topic, rclcpp::SensorDataQoS());
    }
    native_ground_temporal_filter_enabled_ = declare_parameter<bool>(
      "native_ground_temporal_filter_enabled", false);
    native_lidar_ground_patch_minimum_frames_ = declare_parameter<int>("native_lidar_ground_patch_minimum_frames", 3);
    native_lidar_ground_patch_minimum_span_ = declare_parameter<double>("native_lidar_ground_patch_minimum_span", 0.30);
    native_temporal_minimum_frames_ = declare_parameter<int>("native_temporal_minimum_frames", 3);
    native_temporal_minimum_span_ = declare_parameter<double>("native_temporal_minimum_span", 0.30);
    native_rejected_evidence_enabled_ = declare_parameter<bool>("native_rejected_evidence_enabled", false);
    native_minimum_distinct_scans_ = declare_parameter<int>(
      "native_minimum_distinct_scans", 2);
    native_neighbor_radius_voxels_ = declare_parameter<int>(
      "native_neighbor_radius_voxels", 1);
    native_minimum_neighbor_voxels_ = declare_parameter<int>(
      "native_minimum_neighbor_voxels", 1);
    native_preserve_ground_ = declare_parameter<bool>(
      "native_preserve_ground", true);
    native_ground_tolerance_ = declare_parameter<double>(
      "native_ground_tolerance", 0.10);

    lidar_translation_ = vector3_parameter("lidar_translation", {0.0, 0.0, 0.30});
    const Eigen::Vector3d lidar_rpy = vector3_parameter("lidar_rpy", {0.0, 0.0, 0.0});
    self_min_ = vector3_parameter("self_min", {-0.45, -0.30, 0.00});
    self_max_ = vector3_parameter("self_max", {0.45, 0.30, 0.55});
    lidar_rotation_ =
      Eigen::AngleAxisd(lidar_rpy.z(), Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(lidar_rpy.y(), Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(lidar_rpy.x(), Eigen::Vector3d::UnitX());

    // Fine voxels must nest inside the fixed evidence grid. Otherwise a fine
    // centroid can fall in a third patch that none of its measurements hit.
    const double patch_subdivisions = voxel_size_ > 0.0 ? kTemporalPatchSize / voxel_size_ : 0.0;
    if (
      native_lidar_ground_patch_minimum_frames_ < 3 || native_lidar_ground_patch_minimum_frames_ > 100 ||
      !std::isfinite(native_lidar_ground_patch_minimum_span_) || native_lidar_ground_patch_minimum_span_ < .30 || native_lidar_ground_patch_minimum_span_ > 5.0 ||
      native_temporal_minimum_frames_ < 2 || native_temporal_minimum_frames_ > 100 ||
      !std::isfinite(native_temporal_minimum_span_) || native_temporal_minimum_span_ <= 0.0 ||
      native_temporal_minimum_span_ > 5.0 ||
      (native_ground_temporal_filter_enabled_ &&
      (!native_preserve_ground_ || !native_temporal_patch_enabled_)) ||
      (native_temporal_support_enabled_ && !native_confidence_filter_enabled_) ||
      (native_temporal_patch_enabled_ && (!native_temporal_support_enabled_ ||
      native_temporal_minimum_frames_ < 3 || native_temporal_minimum_span_ < 0.30 ||
      voxel_size_ > kTemporalPatchSize || !std::isfinite(patch_subdivisions) ||
      std::abs(patch_subdivisions - std::round(patch_subdivisions)) > 1e-9)) ||
      !std::isfinite(odom_jump_distance_allowance_) || odom_jump_distance_allowance_ <= 0.0 ||
      !std::isfinite(odom_maximum_speed_) || odom_maximum_speed_ <= 0.0 ||
      !std::isfinite(odom_jump_angle_allowance_) || odom_jump_angle_allowance_ <= 0.0 ||
      !std::isfinite(odom_maximum_angular_speed_) || odom_maximum_angular_speed_ <= 0.0 ||
      !std::isfinite(voxel_size_) || !std::isfinite(min_range_) || !std::isfinite(max_range_) ||
      !std::isfinite(max_odom_time_offset_) || max_odom_time_offset_ <= 0.0 ||
      !std::isfinite(world_z_min_) || !std::isfinite(world_z_max_) ||
      !std::isfinite(lidar_pose_time_offset_) || std::abs(lidar_pose_time_offset_) > 0.20 ||
      !std::isfinite(stationary_linear_limit_) || stationary_linear_limit_ <= 0.0 ||
      !std::isfinite(stationary_angular_limit_) || stationary_angular_limit_ <= 0.0 ||
      !std::isfinite(stationary_settle_seconds_) || stationary_settle_seconds_ < 0.0 ||
      voxel_size_ <= 0.0 || min_range_ < 0.0 || max_range_ <= min_range_ ||
      process_every_n_scans_ <= 0 || world_z_max_ <= world_z_min_ ||
      publish_every_n_scans_ < 0 || preview_every_n_scans_ < 0 ||
      preview_max_points_ < 0 || !std::isfinite(preview_z_min_) ||
      !std::isfinite(preview_z_max_) || preview_z_max_ <= preview_z_min_ ||
      preview_minimum_voxel_observations_ < 1 ||
      preview_minimum_voxel_observations_ > 100 || !std::isfinite(ground_plane_z_) ||
      !std::isfinite(ground_snap_tolerance_) || ground_snap_tolerance_ <= 0.0 ||
      ground_snap_tolerance_ > 1.0 || !std::isfinite(maximum_mapping_yaw_rate_) ||
      maximum_mapping_yaw_rate_ < 0.0 || native_minimum_distinct_scans_ < 1 ||
      native_minimum_distinct_scans_ > 100 || native_neighbor_radius_voxels_ < 0 ||
      native_neighbor_radius_voxels_ > 3 || native_minimum_neighbor_voxels_ < 0 ||
      native_minimum_neighbor_voxels_ > 100 || !std::isfinite(native_ground_tolerance_) ||
      native_ground_tolerance_ < 0.0 || native_ground_tolerance_ > 1.0)
    {
      throw std::invalid_argument("Invalid mapper parameters");
    }

    // Point-cloud accumulation can take longer than one high-rate odometry
    // period. Separate callback groups let odometry continue filling the pose
    // buffer while a LiDAR frame is transformed, instead of creating periodic
    // false "no odom within tolerance" drops in a single-threaded executor.
    odom_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    lidar_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    service_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions odom_options;
    odom_options.callback_group = odom_callback_group_;
    rclcpp::SubscriptionOptions lidar_options;
    lidar_options.callback_group = lidar_callback_group_;
    auto sensor_qos = rclcpp::SensorDataQoS();
    odom_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, sensor_qos,
      std::bind(&OdomVoxelMapper::odom_callback, this, std::placeholders::_1), odom_options);
    lidar_subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      lidar_topic_, sensor_qos,
      [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) {lidar_callback(msg, false);},
      lidar_options);
    if (!depth_ground_topic.empty()) {
      depth_subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        depth_ground_topic, sensor_qos,
        [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) {lidar_callback(msg, true);},
        lidar_options);
    }
    map_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      map_topic_, rclcpp::QoS(1).transient_local().reliable());
    if (!preview_topic_.empty() && preview_max_points_ > 0) {
      preview_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        preview_topic_, rclcpp::QoS(1).transient_local().reliable());
    }
    save_service_ = create_service<std_srvs::srv::Trigger>(
      save_service_name_,
      std::bind(
        &OdomVoxelMapper::save_service_callback, this,
        std::placeholders::_1, std::placeholders::_2),
      rmw_qos_profile_services_default, service_callback_group_);

    RCLCPP_INFO(
      get_logger(),
      "Mapping %s with %s, voxel=%.3f m, ground_snap=%s "
      "plane_z=%.3f tolerance=%.3f m, confidence_filter=%s "
      "min_scans=%d neighbor_radius=%d min_neighbors=%d, "
      "temporal_support=%s min_frames=%d min_span=%.3f s patch_10cm=%s, "
      "lidar translation=[%.3f %.3f %.3f]",
      lidar_topic_.c_str(), odom_topic_.c_str(), voxel_size_,
      ground_snap_enabled_ ? "on" : "off", ground_plane_z_, ground_snap_tolerance_,
      native_confidence_filter_enabled_ ? "on" : "off",
      native_minimum_distinct_scans_, native_neighbor_radius_voxels_,
      native_minimum_neighbor_voxels_,
      native_temporal_support_enabled_ ? "on" : "off",
      native_temporal_minimum_frames_, native_temporal_minimum_span_,
      native_temporal_patch_enabled_ ? "on" : "off",
      lidar_translation_.x(), lidar_translation_.y(), lidar_translation_.z());
  }

private:
  Eigen::Vector3d vector3_parameter(
    const std::string & name, const std::array<double, 3> & default_value)
  {
    const std::vector<double> fallback(default_value.begin(), default_value.end());
    const auto values = declare_parameter<std::vector<double>>(name, fallback);
    if (values.size() != 3U || !std::all_of(values.begin(), values.end(), [](double v) {return std::isfinite(v);})) {
      throw std::invalid_argument(name + " must contain exactly three finite values");
    }
    return {values[0], values[1], values[2]};
  }

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr message)
  {
    OdomSample sample;
    sample.stamp = stamp_seconds(message->header.stamp);
    sample.position = {
      message->pose.pose.position.x,
      message->pose.pose.position.y,
      message->pose.pose.position.z};
    sample.orientation = Eigen::Quaterniond(
      message->pose.pose.orientation.w,
      message->pose.pose.orientation.x,
      message->pose.pose.orientation.y,
      message->pose.pose.orientation.z);
    if (!std::isfinite(sample.stamp) || sample.stamp <= 0.0 ||
      !sample.orientation.coeffs().allFinite() || sample.orientation.squaredNorm() < 0.5 ||
      sample.orientation.squaredNorm() > 1.5) {return;}
    sample.orientation.normalize();
    sample.yaw_rate = message->twist.twist.angular.z;
    const auto & v = message->twist.twist.linear;
    const auto & w = message->twist.twist.angular;
    sample.linear_speed = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    sample.angular_speed = std::sqrt(w.x*w.x + w.y*w.y + w.z*w.z);
    if (!sample.position.allFinite() || !sample.orientation.coeffs().allFinite() ||
      !std::isfinite(sample.linear_speed) || !std::isfinite(sample.angular_speed)) {return;}

    std::string fault;
    {
      std::lock_guard<std::mutex> lock(odom_mutex_);
      if (mapping_fault_.load()) {return;}
      if (!odom_samples_.empty()) {
        const auto & previous = odom_samples_.back();
        fault = pose_continuity_problem(
          sample.stamp - previous.stamp, (sample.position - previous.position).norm(),
          previous.orientation.angularDistance(sample.orientation),
          odom_jump_distance_allowance_, odom_maximum_speed_,
          odom_jump_angle_allowance_, odom_maximum_angular_speed_);
        if (!fault.empty()) {mapping_fault_.store(true);}
        else if (sample.stamp <= previous.stamp) {return;}
      }
      if (fault.empty()) {
        odom_samples_.push_back(sample);
        while (!odom_samples_.empty() && sample.stamp - odom_samples_.front().stamp > 5.0) {
          odom_samples_.pop_front();
        }
      }
    }
    if (!fault.empty()) {
      // Release the odometry lock before waiting for an acquisition/save: the
      // acquisition callback takes the locks in the opposite order.
      {
        std::lock_guard<std::recursive_mutex> lock(acquisition_mutex_);
        mapping_fault_reason_ = fault;
      }
      RCLCPP_FATAL(get_logger(), "Mapping aborted: %s; preserving the pre-fault cloud", fault.c_str());
      try {save_map(save_path_ + ".fault_checkpoint.pcd");}
      catch (const std::exception & error) {
        RCLCPP_ERROR(get_logger(), "Fault checkpoint failed: %s", error.what());
      }
      // This is a required node. Exiting propagates failure to the mission
      // supervisor; subsequent poses must never be fused into the old frame.
      throw std::runtime_error("mapping continuity fault: " + fault);
    }
  }

  bool odom_at_time(double stamp, OdomSample & output, double & time_offset)
  {
    std::lock_guard<std::mutex> lock(odom_mutex_);
    if (odom_samples_.empty()) {
      return false;
    }
    const auto iterator = std::lower_bound(
      odom_samples_.begin(), odom_samples_.end(), stamp,
      [](const OdomSample & sample, double value) {return sample.stamp < value;});

    const OdomSample * next = nullptr;
    if (iterator != odom_samples_.end()) {
      next = &(*iterator);
    }
    const OdomSample * previous = nullptr;
    if (iterator != odom_samples_.begin()) {
      previous = &(*std::prev(iterator));
    }

    // MuJoCo odometry is published much faster than the LiDAR. Interpolating
    // between the two bracketing samples removes the remaining millisecond
    // timestamp quantisation without introducing scan-matching drift.
    if (previous != nullptr && next != nullptr && next->stamp > previous->stamp) {
      const double before_offset = stamp - previous->stamp;
      const double after_offset = next->stamp - stamp;
      if (
        before_offset >= 0.0 && after_offset >= 0.0 &&
        before_offset <= max_odom_time_offset_ && after_offset <= max_odom_time_offset_)
      {
        const double alpha = before_offset / (next->stamp - previous->stamp);
        output.stamp = stamp;
        output.position =
          (1.0 - alpha) * previous->position + alpha * next->position;
        output.orientation = previous->orientation.slerp(alpha, next->orientation).normalized();
        output.yaw_rate =
          (1.0 - alpha) * previous->yaw_rate + alpha * next->yaw_rate;
        output.linear_speed = (1.0-alpha)*previous->linear_speed + alpha*next->linear_speed;
        output.angular_speed = (1.0-alpha)*previous->angular_speed + alpha*next->angular_speed;
        time_offset = std::min(before_offset, after_offset);
        return true;
      }
    }

    const OdomSample * best = next;
    if (
      previous != nullptr &&
      (best == nullptr || std::abs(previous->stamp - stamp) < std::abs(best->stamp - stamp)))
    {
      best = previous;
    }
    if (best == nullptr) {
      return false;
    }
    time_offset = best->stamp - stamp;
    if (std::abs(time_offset) > max_odom_time_offset_) {
      return false;
    }
    output = *best;
    return true;
  }

  bool inside_self_filter(const Eigen::Vector3d & point_base) const
  {
    return self_filter_enabled_ &&
           (point_base.array() >= self_min_.array()).all() &&
           (point_base.array() <= self_max_.array()).all();
  }

  void refine_ground_timing(
    const sensor_msgs::msg::PointCloud2 & message, double stamp, OdomSample & odom)
  {
    // Flat-ground timing calibration only. Never changes the odometry stream
    // or estimates a free global pose. Insufficient ground => fixed-offset fallback.
    std::vector<Eigen::Vector3d> samples;
    std::array<uint32_t, 3> offsets{};
    for (const auto & field : message.fields) {
      if (field.name == "x") {offsets[0] = field.offset;}
      else if (field.name == "y") {offsets[1] = field.offset;}
      else if (field.name == "z") {offsets[2] = field.offset;}
    }
    Eigen::Vector3d previous = Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN());
    for (uint32_t row = 0; row < message.height; ++row) {
    for (uint32_t column = 0; column < message.width; ++column) {
      std::array<float, 3> xyz{};
      const auto byte_offset = std::size_t(row) * message.row_step + std::size_t(column) * message.point_step;
      for (std::size_t axis = 0; axis < xyz.size(); ++axis) {
        std::memcpy(&xyz[axis], message.data.data() + byte_offset + offsets[axis], sizeof(float));
      }
      const Eigen::Vector3d raw(xyz[0], xyz[1], xyz[2]);
      if (!raw.allFinite() || raw == previous) {continue;}
      previous = raw;
      if (raw.norm() < std::max(0.6, min_range_) || raw.head<2>().norm() > 10.0) {continue;}
      const Eigen::Vector3d base = lidar_rotation_ * raw + lidar_translation_;
      if (inside_self_filter(base)) {continue;}
      if (std::abs((odom.orientation * base + odom.position).z() - ground_plane_z_) < 0.035) {
        samples.push_back(base);
      }
    }
    }
    if (samples.size() < 100) {return;}
    const auto score = [&](const OdomSample & pose, std::size_t parity) {
        std::vector<double> residuals;
        for (std::size_t i = parity; i < samples.size(); i += 2) {
          residuals.push_back(std::abs(
            (pose.orientation * samples[i] + pose.position).z() - ground_plane_z_));
        }
        auto mid = residuals.begin() + residuals.size() / 2;
        std::nth_element(residuals.begin(), mid, residuals.end());
        return *mid;
      };
    const double baseline = score(odom, 0);
    double best_score = baseline;
    OdomSample best = odom;
    for (int i = -8; i <= 8; ++i) {
      OdomSample candidate;
      double dt;
      if (!odom_at_time(stamp + lidar_pose_time_offset_ + i * 0.0025, candidate, dt)) {continue;}
      const double value = score(candidate, 0);
      if (value < best_score) {best_score = value; best = candidate;}
    }
    // Held-out points must also improve; ambiguous or tiny corrections are rejected.
    const double validation_before = score(odom, 1);
    const double validation_after = score(best, 1);
    if (best_score < baseline * 0.75 && baseline - best_score > 0.0005 &&
      validation_after < validation_before * 0.9 &&
      validation_before - validation_after > 0.00025)
    {
      odom = best;
      ++timing_refined_scans_;
    }
  }

  void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr message, bool ground_only)
  {
    std::lock_guard<std::recursive_mutex> acquisition_lock(acquisition_mutex_);
    if (mapping_fault_.load()) {return;}
    ++received_scans_;
    std::vector<CloudFieldLayout> fields;
    for (const auto & field : message->fields) {
      fields.push_back({field.name, field.offset, field.datatype, field.count});
    }
    const auto layout_error = cloud_layout_problem(
      message->width, message->height, message->point_step, message->row_step,
      message->data.size(), message->is_bigendian, fields);
    if (!layout_error.empty()) {
      ++invalid_cloud_frames_;
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
        "Rejected malformed PointCloud2: %s", layout_error.c_str());
      return;
    }
    if ((received_scans_ - 1U) % static_cast<std::size_t>(process_every_n_scans_) != 0U) {
      ++skipped_scans_;
      return;
    }
    OdomSample odom;
    double time_offset = 0.0;
    const double lidar_stamp = stamp_seconds(message->header.stamp);
    const int64_t lidar_stamp_ns = static_cast<int64_t>(message->header.stamp.sec) * 1000000000LL +
      static_cast<int64_t>(message->header.stamp.nanosec);
    double & last_stamp = ground_only ? last_depth_stamp_ : last_lidar_stamp_;
    if (!std::isfinite(lidar_stamp) || lidar_stamp <= 0.0 || lidar_stamp <= last_stamp) {
      ++nonadvancing_cloud_frames_;
      return;
    }
    last_stamp = lidar_stamp;
    // Sensor publication time can lag the actual render pose. This offset is
    // calibrated from raw scans, not inferred from message arrival latency.
    if (!odom_at_time(lidar_stamp + lidar_pose_time_offset_, odom, time_offset)) {
      ++dropped_scans_;
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Dropping lidar scan: no odom within %.3f s (dropped=%zu)",
        max_odom_time_offset_, dropped_scans_);
      return;
    }
    bool stable_structure = true;
    if (!ground_only && (stationary_mapping_ || native_ground_temporal_filter_enabled_)) {
      double & settled_since = stationary_since_;
      const bool stable = odom.linear_speed <= stationary_linear_limit_ &&
        odom.angular_speed <= stationary_angular_limit_;
      if (!stable) {settled_since = -1.0;}
      else if (settled_since < 0.0) {settled_since = lidar_stamp;}
      if (!stable || lidar_stamp - settled_since < stationary_settle_seconds_) {
        ++stationary_rejected_scans_;
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 10000,
          "Stationary structural mapping waiting: v=%.3f w=%.3f rejected=%zu structural=%zu",
          odom.linear_speed, odom.angular_speed, stationary_rejected_scans_, structural_scans_);
        stable_structure = false;
      }
    }
    if (adaptive_ground_timing_) {
      try {
        refine_ground_timing(*message, lidar_stamp, odom);
      } catch (const std::runtime_error & error) {
        ++dropped_scans_;
        RCLCPP_ERROR(get_logger(), "Timing PointCloud2 field error: %s", error.what());
        return;
      }
    }
    if (
      maximum_mapping_yaw_rate_ > 0.0 &&
      std::abs(odom.yaw_rate) > maximum_mapping_yaw_rate_)
    {
      ++motion_rejected_scans_;
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Rejecting LiDAR during |yaw_rate|=%.3f > %.3f rad/s "
        "(motion_rejected=%zu)",
        std::abs(odom.yaw_rate), maximum_mapping_yaw_rate_, motion_rejected_scans_);
      return;
    }

    if (!ground_only && stable_structure) {++structural_scans_;}
    try {
      std::array<uint32_t, 3> offsets{};
      for (const auto & field : fields) {
        if (field.name == "x") {offsets[0] = field.offset;}
        else if (field.name == "y") {offsets[1] = field.offset;}
        else if (field.name == "z") {offsets[2] = field.offset;}
      }
      const auto surface_started = std::chrono::steady_clock::now();
      std::unique_ptr<ScanSurfaceFilter> surface;
      if (scan_surface_filter_enabled_ && !ground_only && stable_structure) {
        std::vector<Eigen::Vector3d> support;
        support.reserve(std::size_t(message->width) * message->height);
        for (uint32_t row=0; row<message->height; ++row) {
          for (uint32_t column=0; column<message->width; ++column) {
            std::array<float,3> xyz{};
            const auto base=std::size_t(row)*message->row_step+std::size_t(column)*message->point_step;
            for (std::size_t axis=0;axis<3;++axis) {std::memcpy(&xyz[axis],message->data.data()+base+offsets[axis],sizeof(float));}
            Eigen::Vector3d p(xyz[0],xyz[1],xyz[2]);
            if (p.allFinite() && p.norm()>=min_range_ && p.norm()<=max_range_ && !inside_self_filter(lidar_rotation_*p+lidar_translation_)) {support.push_back(p);}
          }
        }
        surface=std::make_unique<ScanSurfaceFilter>(support);
      }
      auto filtered_local=std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
      Eigen::Vector3d previous_point = Eigen::Vector3d::Constant(
        std::numeric_limits<double>::quiet_NaN());
      std::unique_lock<std::mutex> map_lock(map_mutex_);
      for (uint32_t row = 0; row < message->height; ++row) {
      for (uint32_t column = 0; column < message->width; ++column) {
        std::array<float, 3> xyz{};
        const auto base_offset = std::size_t(row) * message->row_step + std::size_t(column) * message->point_step;
        for (std::size_t axis = 0; axis < xyz.size(); ++axis) {
          std::memcpy(&xyz[axis], message->data.data() + base_offset + offsets[axis], sizeof(float));
        }
        const Eigen::Vector3d point_lidar(xyz[0], xyz[1], xyz[2]);
        if (!point_lidar.allFinite()) {
          continue;
        }
        // Matrix can repeat one identical ray thousands of times in a frame.
        // Count that measurement once so it cannot dominate a voxel centroid.
        if (point_lidar == previous_point) {
          continue;
        }
        previous_point = point_lidar;
        const double range = point_lidar.norm();
        if (range < min_range_ || range > max_range_) {
          continue;
        }
        const Eigen::Vector3d point_base = lidar_rotation_ * point_lidar + lidar_translation_;
        if (inside_self_filter(point_base)) {
          continue;
        }
        Eigen::Vector3d point_world = odom.orientation * point_base + odom.position;
        // Depth is permitted only to add measured near-floor returns. It must
        // not introduce a second, less well calibrated wall surface.
        if ((ground_only && std::abs(point_world.z() - ground_plane_z_) > 0.04) ||
          (!ground_only && !stable_structure &&
          std::abs(point_world.z() - ground_plane_z_) > ground_snap_tolerance_)) {
          continue;
        }
        // Temporal repetition alone cannot reject a moving-floor smear: a
        // biased patch can recur across frames. Apply the same stable-pose
        // requirement to LiDAR floor returns before normalization. Measured
        // depth floor candidates retain their existing 4 cm world-plane gate
        // and the new cross-frame gate: excluding all moving depth removed
        // valid floor coverage in the complete candidate-2 validation.
        if (native_ground_temporal_filter_enabled_ && !ground_only && !stable_structure &&
          std::abs(point_world.z() - ground_plane_z_) <= ground_snap_tolerance_)
        {
          ++ground_motion_rejected_measurements_;
          continue;
        }
        if (point_world.z() < world_z_min_ || point_world.z() > world_z_max_) {
          continue;
        }
        if (surface && std::abs(point_world.z()-ground_plane_z_)>native_ground_tolerance_) {
          ++surface_checked_measurements_;
          if (!surface->accepts(point_lidar)) {++surface_rejected_measurements_;continue;}
        }
        if (!ground_only && filtered_lidar_publisher_ != nullptr) {
          filtered_local->emplace_back(point_lidar.x(),point_lidar.y(),point_lidar.z());
        }
        // A quadruped's small roll/pitch motion can spread one physical floor
        // across several vertical voxels. Normalize only the configured
        // near-floor band before voxel accumulation; all other geometry keeps
        // its measured height and every floor XY cell remains represented.
        if (
          ground_snap_enabled_ &&
          std::abs(point_world.z() - ground_plane_z_) <= ground_snap_tolerance_)
        {
          point_world.z() = ground_plane_z_;
        }
        const double smallest_key_size = native_temporal_patch_enabled_ ?
          std::min(voxel_size_, kTemporalPatchSize) : voxel_size_;
        const Eigen::Vector3d voxel_coordinate = point_world / smallest_key_size;
        // Reject overflow before converting to integer keys. A finite but
        // absurd odometry value must not trigger undefined integer conversion.
        if (!voxel_coordinate.allFinite() ||
          (voxel_coordinate.array().abs() >= static_cast<double>(std::numeric_limits<int64_t>::max())/2.0).any()) {
          ++invalid_world_points_; continue;
        }
        const VoxelKey key{
          static_cast<int64_t>(std::floor(point_world.x() / voxel_size_)),
          static_cast<int64_t>(std::floor(point_world.y() / voxel_size_)),
          static_cast<int64_t>(std::floor(point_world.z() / voxel_size_))};
        voxels_[key].add(point_world, received_scans_, lidar_stamp_ns, ground_only);
        if (native_temporal_patch_enabled_) {
          // Sparse scanning can hit different fine voxels of the same small
          // surface across frames. Only raw accepted observations count here;
          // eligibility, neighbors and ground exemptions never create evidence.
          patch_observations_[temporal_patch_key(point_world)].observe(received_scans_, lidar_stamp_ns);
        }
        if (ground_only) {++depth_ground_points_;}
      }
      }
      map_lock.unlock();
      if (surface) {
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-surface_started).count();
        surface_filter_total_seconds_+=seconds;surface_filter_maximum_seconds_=std::max(surface_filter_maximum_seconds_,seconds);
      }
      if (!ground_only && filtered_lidar_publisher_ != nullptr) {
        sensor_msgs::msg::PointCloud2 filtered;
        filtered_local->width=static_cast<uint32_t>(filtered_local->size());filtered_local->height=1;filtered_local->is_dense=true;
        pcl::toROSMsg(*filtered_local,filtered);filtered.header=message->header;filtered_lidar_publisher_->publish(filtered);
      }
    } catch (const std::runtime_error & error) {
      ++dropped_scans_;
      RCLCPP_ERROR(get_logger(), "PointCloud2 field error: %s", error.what());
      return;
    }

    ++processed_scans_;
    absolute_time_offset_sum_ += std::abs(time_offset);
    max_absolute_time_offset_ = std::max(max_absolute_time_offset_, std::abs(time_offset));

    if (publish_every_n_scans_ > 0 && processed_scans_ % publish_every_n_scans_ == 0U) {
      publish_map(message->header.stamp);
    }
    if (
      preview_publisher_ != nullptr && preview_every_n_scans_ > 0 &&
      processed_scans_ % preview_every_n_scans_ == 0U)
    {
      publish_preview(message->header.stamp);
    }
    if (
      checkpoint_every_n_scans_ > 0 &&
      processed_scans_ % checkpoint_every_n_scans_ == 0U)
    {
      save_map(save_path_ + ".checkpoint.pcd");
    }
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "scans=%zu skipped=%zu dropped=%zu motion_rejected=%zu voxels=%zu "
      "odom_offset_mean=%.6f s max=%.6f s timing_refined=%zu depth_ground_points=%zu",
      processed_scans_, skipped_scans_, dropped_scans_, motion_rejected_scans_, voxel_count(),
      absolute_time_offset_sum_ / static_cast<double>(processed_scans_),
      max_absolute_time_offset_, timing_refined_scans_, depth_ground_points_);
  }

  bool has_spatial_support(const VoxelKey & key) const
  {
    if (native_minimum_neighbor_voxels_ == 0) {
      return !native_temporal_support_enabled_;
    }
    int neighbors = 0;
    const int radius = native_neighbor_radius_voxels_;
    for (int dx = -radius; dx <= radius; ++dx) {
      for (int dy = -radius; dy <= radius; ++dy) {
        for (int dz = -radius; dz <= radius; ++dz) {
          if (dx == 0 && dy == 0 && dz == 0) {
            continue;
          }
          const VoxelKey candidate{key.x + dx, key.y + dy, key.z + dz};
          const auto found = voxels_.find(candidate);
          if (found != voxels_.end() &&
            (!native_temporal_support_enabled_ || found->second.observation.trusted(
              static_cast<uint32_t>(native_temporal_minimum_frames_), native_temporal_minimum_span_)))
          {
            // Only direct temporal evidence creates an anchor. An adjacent
            // unconfirmed voxel cannot bootstrap another unconfirmed voxel.
            ++neighbors;
            if (neighbors >= native_minimum_neighbor_voxels_) {
              return true;
            }
          }
        }
      }
    }
    return false;
  }

  bool ground_has_temporal_support(const VoxelAccumulator & accumulator) const
  {
    // Ground must never acquire trust merely from one dense frame or from
    // neighboring one-hit floor voxels. A fixed 10 cm measured patch allows
    // sparse fine-voxel hits only after genuinely time-separated observations.
    if (accumulator.observation.trusted(
        static_cast<uint32_t>(native_temporal_minimum_frames_), native_temporal_minimum_span_))
    {return true;}
    const auto found = patch_observations_.find(temporal_patch_key(accumulator.centroid()));
    // A short burst can scatter one LiDAR return into several fine cells of
    // one coarse floor patch. Require longer patch evidence for LiDAR-only
    // single hits; direct repeat support and measured depth keep their rules.
    const bool lidar_only = accumulator.depth_measurements == 0;
    const int minimum_frames = lidar_only ? std::max(native_temporal_minimum_frames_, native_lidar_ground_patch_minimum_frames_) : native_temporal_minimum_frames_;
    const double minimum_span = lidar_only ? std::max(native_temporal_minimum_span_, native_lidar_ground_patch_minimum_span_) : native_temporal_minimum_span_;
    return found != patch_observations_.end() && found->second.trusted(
      static_cast<uint32_t>(minimum_frames), minimum_span);
  }

  ConfidenceReason native_voxel_reason(
    const VoxelKey & key, const VoxelAccumulator & accumulator) const
  {
    if (!native_confidence_filter_enabled_) {return ConfidenceReason::DISABLED;}
    const double z = accumulator.centroid().z();
    if (native_preserve_ground_ && std::abs(z - ground_plane_z_) <= native_ground_tolerance_) {
      return !native_ground_temporal_filter_enabled_ || ground_has_temporal_support(accumulator) ?
        ConfidenceReason::GROUND : ConfidenceReason::REJECTED;
    }
    const bool self_supported = native_temporal_support_enabled_ ?
      accumulator.observation.trusted(
        static_cast<uint32_t>(native_temporal_minimum_frames_), native_temporal_minimum_span_) :
      accumulator.observation.distinct_frames >= static_cast<uint32_t>(native_minimum_distinct_scans_);
    if (self_supported) {return ConfidenceReason::SELF;}
    if (has_spatial_support(key)) {return ConfidenceReason::NEIGHBOR;}
    if (native_temporal_patch_enabled_) {
      const auto found = patch_observations_.find(temporal_patch_key(accumulator.centroid()));
      if (found != patch_observations_.end() && found->second.trusted(
          static_cast<uint32_t>(native_temporal_minimum_frames_), native_temporal_minimum_span_)) {
        return ConfidenceReason::PATCH;
      }
    }
    return ConfidenceReason::REJECTED;
  }

  static VoxelKey temporal_patch_key(const Eigen::Vector3d & point)
  {
    return {static_cast<int64_t>(std::floor(point.x() / kTemporalPatchSize)),
      static_cast<int64_t>(std::floor(point.y() / kTemporalPatchSize)),
      static_cast<int64_t>(std::floor(point.z() / kTemporalPatchSize))};
  }

  bool native_voxel_eligible(
    const VoxelKey & key, const VoxelAccumulator & accumulator) const
  {
    // Eligibility is the same union as native_voxel_reason(). Try the cheap
    // independent patch lookup before up to 26 fine-neighbor lookups. Keep the
    // original reason ordering in saved diagnostics so their meaning is stable.
    if (!native_confidence_filter_enabled_) {return true;}
    const auto point = accumulator.centroid();
    if (native_preserve_ground_ && std::abs(point.z() - ground_plane_z_) <= native_ground_tolerance_) {
      return !native_ground_temporal_filter_enabled_ || ground_has_temporal_support(accumulator);
    }
    if (native_temporal_support_enabled_ ? accumulator.observation.trusted(
        static_cast<uint32_t>(native_temporal_minimum_frames_), native_temporal_minimum_span_) :
      accumulator.observation.distinct_frames >= static_cast<uint32_t>(native_minimum_distinct_scans_)) {
      return true;
    }
    if (native_temporal_patch_enabled_) {
      const auto found = patch_observations_.find(temporal_patch_key(point));
      if (found != patch_observations_.end() && found->second.trusted(
          static_cast<uint32_t>(native_temporal_minimum_frames_), native_temporal_minimum_span_)) {
        return true;
      }
    }
    return has_spatial_support(key);
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr snapshot_cloud(
    std::size_t * rejected_voxels = nullptr, std::array<std::size_t, 6> * reasons = nullptr)
  {
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    std::lock_guard<std::mutex> lock(map_mutex_);
    cloud->reserve(voxels_.size());
    for (const auto & item : voxels_) {
      const auto reason = native_voxel_reason(item.first, item.second);
      if (reasons != nullptr) {++(*reasons)[static_cast<std::size_t>(reason)];}
      if (reason == ConfidenceReason::REJECTED) {continue;}
      const Eigen::Vector3d point = item.second.centroid();
      cloud->emplace_back(
        static_cast<float>(point.x()),
        static_cast<float>(point.y()),
        static_cast<float>(point.z()));
    }
    cloud->width = static_cast<uint32_t>(cloud->size());
    cloud->height = 1U;
    cloud->is_dense = true;
    if (rejected_voxels != nullptr) {
      *rejected_voxels = voxels_.size() - cloud->size();
    }
    return cloud;
  }

  std::size_t voxel_count()
  {
    std::lock_guard<std::mutex> lock(map_mutex_);
    return voxels_.size();
  }

  void publish_map(const builtin_interfaces::msg::Time & stamp)
  {
    const auto cloud = snapshot_cloud();
    sensor_msgs::msg::PointCloud2 message;
    pcl::toROSMsg(*cloud, message);
    message.header.stamp = stamp;
    message.header.frame_id = map_frame_;
    map_publisher_->publish(message);
  }

  void publish_preview(const builtin_interfaces::msg::Time & stamp)
  {
    const auto started = std::chrono::steady_clock::now();
    auto preview = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    const std::size_t maximum = static_cast<std::size_t>(preview_max_points_);
    {
      std::lock_guard<std::mutex> lock(map_mutex_);
      const auto eligible = [this](const auto & item) {
          const double z = item.second.centroid().z();
          return
            // With confidence enabled, the native decision supersedes the old
            // ray-count heuristic; otherwise patch-supported single hits would
            // be present in the PCD but hidden from the preview.
            (native_confidence_filter_enabled_ ||
            item.second.count >= static_cast<uint32_t>(preview_minimum_voxel_observations_)) &&
            z >= preview_z_min_ && z <= preview_z_max_ &&
            native_voxel_eligible(item.first, item.second);
        };
      // Evaluate confidence once per voxel. Pointers remain valid under the
      // existing map lock, and preserve the former iteration order and stride.
      std::vector<const VoxelAccumulator *> accepted;
      accepted.reserve(voxels_.size());
      for (const auto & item : voxels_) {
        if (eligible(item)) {accepted.push_back(&item.second);}
      }
      const std::size_t eligible_count = accepted.size();
      const std::size_t stride = std::max<std::size_t>(
        1U, (eligible_count + maximum - 1U) / maximum);
      preview->reserve(std::min(maximum, eligible_count));
      for (std::size_t index = 0U; index < accepted.size() && preview->size() < maximum; index += stride) {
        const Eigen::Vector3d point = accepted[index]->centroid();
        preview->emplace_back(
          static_cast<float>(point.x()),
          static_cast<float>(point.y()),
          static_cast<float>(point.z()));
      }
    }
    if (preview->empty()) {
      return;
    }
    preview->width = static_cast<uint32_t>(preview->size());
    preview->height = 1U;
    preview->is_dense = true;
    sensor_msgs::msg::PointCloud2 message;
    pcl::toROSMsg(*preview, message);
    message.header.stamp = stamp;
    message.header.frame_id = map_frame_;
    preview_publisher_->publish(message);
    const double elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
    ++preview_publications_;
    preview_total_seconds_ += elapsed;
    preview_maximum_seconds_ = std::max(preview_maximum_seconds_, elapsed);
  }

  bool save_map(const std::string & path)
  {
    std::lock_guard<std::recursive_mutex> acquisition_lock(acquisition_mutex_);
    std::size_t rejected_voxels = 0U;
    std::array<std::size_t, 6> confidence_counts{};
    const auto cloud = snapshot_cloud(&rejected_voxels, &confidence_counts);
    if (cloud->empty()) {
      RCLCPP_ERROR(get_logger(), "Refusing to save an empty map to %s", path.c_str());
      return false;
    }
    const std::filesystem::path output_path(path);
    std::filesystem::create_directories(output_path.parent_path());
    const std::filesystem::path temporary_path = output_path.string() + ".tmp";
    if (pcl::io::savePCDFileBinary(temporary_path.string(), *cloud) != 0) {
      RCLCPP_ERROR(get_logger(), "Failed to write %s", temporary_path.c_str());
      return false;
    }
    std::filesystem::rename(temporary_path, output_path);
    // Diagnostic evidence is written only for the requested final native map,
    // never for periodic/fault checkpoints or live publications.
    const bool write_rejected_evidence = native_rejected_evidence_enabled_ && path == save_path_;
    if (write_rejected_evidence) {
      std::ofstream rejected(path + ".rejected_voxels.csv.tmp");
      rejected << "x,y,z,measurements,distinct_frames,span_seconds,first_stamp,last_stamp,first_stamp_ns,last_stamp_ns\n";
      rejected << std::setprecision(17);
      std::lock_guard<std::mutex> map_lock(map_mutex_);
      for (const auto & item : voxels_) {
        if (native_voxel_reason(item.first, item.second) != ConfidenceReason::REJECTED) {continue;}
        const auto point = item.second.centroid();
        const auto & evidence = item.second.observation;
        rejected << point.x() << ',' << point.y() << ',' << point.z() << ',' << item.second.count
                 << ',' << evidence.distinct_frames << ',' << evidence.span()
                 << ',' << static_cast<double>(evidence.first_stamp_ns) * 1e-9
                 << ',' << static_cast<double>(evidence.last_stamp_ns) * 1e-9
                 << ',' << evidence.first_stamp_ns << ',' << evidence.last_stamp_ns << '\n';
      }
      rejected.close();
      if (!rejected) {throw std::runtime_error("Failed to write rejected-voxel evidence");}
      std::filesystem::rename(path + ".rejected_voxels.csv.tmp", path + ".rejected_voxels.csv");
      // Keep accepted AND rejected measured floor evidence. This makes the
      // legacy ground exemption comparable on exactly the same acquisition;
      // diagnostic rows never create, move or crop native map points.
      std::ofstream ground(path + ".ground_voxels.csv.tmp");
      ground << "x,y,z,measurements,lidar_measurements,depth_measurements,distinct_frames,span_seconds,first_stamp_ns,last_stamp_ns,patch_frames,patch_span_seconds,accepted\n";
      ground << std::setprecision(17);
      for (const auto & item : voxels_) {
        const auto point = item.second.centroid();
        if (std::abs(point.z() - ground_plane_z_) > native_ground_tolerance_) {continue;}
        const auto & evidence = item.second.observation;
        const auto patch = patch_observations_.find(temporal_patch_key(point));
        ground << point.x() << ',' << point.y() << ',' << point.z() << ',' << item.second.count
               << ',' << item.second.lidar_measurements << ',' << item.second.depth_measurements
               << ',' << evidence.distinct_frames << ',' << evidence.span()
               << ',' << evidence.first_stamp_ns << ',' << evidence.last_stamp_ns
               << ',' << (patch == patch_observations_.end() ? 0U : patch->second.distinct_frames)
               << ',' << (patch == patch_observations_.end() ? 0.0 : patch->second.span())
               << ',' << (native_voxel_eligible(item.first, item.second) ? 1 : 0) << '\n';
      }
      ground.close();
      if (!ground) {throw std::runtime_error("Failed to write ground-voxel evidence");}
      std::filesystem::rename(path + ".ground_voxels.csv.tmp", path + ".ground_voxels.csv");
    }
    std::ofstream stats(path + ".acquisition.json.tmp");
    stats << "{\"stationary_structure_enabled\":" << (stationary_mapping_ ? "true" : "false")
          << ",\"received_frames\":" << received_scans_
          << ",\"processed_frames\":" << processed_scans_
          << ",\"structural_frames\":" << structural_scans_
          << ",\"moving_structure_rejected_frames\":" << stationary_rejected_scans_
          << ",\"preview_publications\":" << preview_publications_
          << ",\"preview_total_seconds\":" << preview_total_seconds_
          << ",\"preview_maximum_seconds\":" << preview_maximum_seconds_
          << ",\"missing_odom_frames\":" << dropped_scans_
          << ",\"depth_ground_measurements\":" << depth_ground_points_
          << ",\"invalid_cloud_frames\":" << invalid_cloud_frames_
          << ",\"nonadvancing_cloud_frames\":" << nonadvancing_cloud_frames_
          << ",\"invalid_world_points\":" << invalid_world_points_
          << ",\"mapping_fault\":\"" << mapping_fault_reason_ << "\""
          << ",\"odom_jump_distance_allowance\":" << odom_jump_distance_allowance_
          << ",\"odom_maximum_speed\":" << odom_maximum_speed_
          << ",\"odom_jump_angle_allowance\":" << odom_jump_angle_allowance_
          << ",\"odom_maximum_angular_speed\":" << odom_maximum_angular_speed_
          << ",\"native_confidence_filter_enabled\":" << (native_confidence_filter_enabled_ ? "true" : "false")
          << ",\"native_temporal_support_enabled\":" << (native_temporal_support_enabled_ ? "true" : "false")
          << ",\"scan_surface_filter_enabled\":" << (scan_surface_filter_enabled_ ? "true" : "false")
          << ",\"native_lidar_ground_patch_minimum_frames\":" << native_lidar_ground_patch_minimum_frames_
          << ",\"native_lidar_ground_patch_minimum_span\":" << native_lidar_ground_patch_minimum_span_
          << ",\"surface_checked_measurements\":" << surface_checked_measurements_
          << ",\"surface_rejected_measurements\":" << surface_rejected_measurements_
          << ",\"surface_filter_total_seconds\":" << surface_filter_total_seconds_
          << ",\"surface_filter_maximum_seconds\":" << surface_filter_maximum_seconds_
          << ",\"native_ground_temporal_filter_enabled\":" << (native_ground_temporal_filter_enabled_ ? "true" : "false")
          << ",\"ground_motion_rejected_measurements\":" << ground_motion_rejected_measurements_
          << ",\"ground_motion_gate_scope\":\"lidar_only_depth_geometry_and_temporal\""
          << ",\"ground_evidence_written\":" << (write_rejected_evidence ? "true" : "false")
          << ",\"native_temporal_patch_enabled\":" << (native_temporal_patch_enabled_ ? "true" : "false")
          << ",\"native_temporal_patch_size_m\":" << kTemporalPatchSize
          << ",\"native_temporal_patch_origin_m\":[0,0,0]"
          << ",\"native_temporal_patch_cells\":" << patch_observations_.size()
          << ",\"native_temporal_minimum_frames\":" << native_temporal_minimum_frames_
          << ",\"native_temporal_minimum_span_seconds\":" << native_temporal_minimum_span_
          << ",\"native_neighbor_radius_voxels\":" << native_neighbor_radius_voxels_
          << ",\"native_minimum_neighbor_voxels\":" << native_minimum_neighbor_voxels_
          << ",\"native_minimum_distinct_scans\":" << native_minimum_distinct_scans_
          << ",\"native_preserve_ground\":" << (native_preserve_ground_ ? "true" : "false")
          << ",\"native_ground_tolerance\":" << native_ground_tolerance_
          << ",\"native_ground_plane_z\":" << ground_plane_z_
          << ",\"voxel_size\":" << voxel_size_
          << ",\"native_rejected_evidence_enabled\":" << (native_rejected_evidence_enabled_ ? "true" : "false")
          << ",\"saved_points\":" << cloud->size()
          << ",\"confidence_reason_counts\":{\"disabled\":" << confidence_counts[0]
          << ",\"ground\":" << confidence_counts[1] << ",\"self\":" << confidence_counts[2]
          << ",\"neighbor\":" << confidence_counts[3] << ",\"patch\":" << confidence_counts[5]
          << ",\"rejected\":" << confidence_counts[4] << "}"
          << ",\"rejected_evidence_written\":" << (write_rejected_evidence ? "true" : "false") << "}\n";
    stats.close();
    if (!stats) {throw std::runtime_error("Failed to write mapper acquisition evidence");}
    std::filesystem::rename(path + ".acquisition.json.tmp", path + ".acquisition.json");
    RCLCPP_INFO(
      get_logger(), "Saved %zu points from %zu scans to %s",
      cloud->size(), processed_scans_, output_path.c_str());
    if (native_confidence_filter_enabled_) {
      RCLCPP_INFO(
        get_logger(), "Online confidence filter excluded %zu unsupported voxels",
        rejected_voxels);
    }
    return true;
  }

  void save_service_callback(
    const std_srvs::srv::Trigger::Request::SharedPtr,
    std_srvs::srv::Trigger::Response::SharedPtr response)
  {
    response->success = save_map(save_path_);
    response->message = response->success ? save_path_ : "Map save failed; inspect mapper logs";
  }

  std::string lidar_topic_;
  std::string odom_topic_;
  std::string map_topic_;
  std::string preview_topic_;
  std::string save_service_name_;
  std::string save_path_;
  std::string map_frame_;
  double voxel_size_{0.035};
  double min_range_{0.20};
  double max_range_{40.0};
  double max_odom_time_offset_{0.02};
  double lidar_pose_time_offset_{0.0};
  std::atomic<bool> mapping_fault_{false};
  std::string mapping_fault_reason_;
  double odom_jump_distance_allowance_{0.25};
  double odom_maximum_speed_{3.0};
  double odom_jump_angle_allowance_{0.35};
  double odom_maximum_angular_speed_{4.0};
  std::size_t invalid_cloud_frames_{0};
  std::size_t nonadvancing_cloud_frames_{0};
  std::size_t invalid_world_points_{0};
  double last_lidar_stamp_{-1.0};
  double last_depth_stamp_{-1.0};
  bool adaptive_ground_timing_{false};
  bool stationary_mapping_{false};
  double stationary_linear_limit_{0.04};
  double stationary_angular_limit_{0.08};
  double stationary_settle_seconds_{1.0};
  double stationary_since_{-1.0};
  std::size_t ground_motion_rejected_measurements_{0};
  std::size_t stationary_rejected_scans_{0};
  std::size_t structural_scans_{0};
  std::size_t timing_refined_scans_{0};
  std::size_t depth_ground_points_{0};
  int process_every_n_scans_{1};
  double world_z_min_{-1.0};
  double world_z_max_{5.0};
  int publish_every_n_scans_{10};
  int preview_every_n_scans_{10};
  std::size_t preview_publications_{0};
  double preview_total_seconds_{0.0};
  double preview_maximum_seconds_{0.0};
  int preview_max_points_{120000};
  double preview_z_min_{-1.0e6};
  double preview_z_max_{1.0e6};
  int preview_minimum_voxel_observations_{1};
  int checkpoint_every_n_scans_{300};
  bool self_filter_enabled_{true};
  bool ground_snap_enabled_{false};
  double maximum_mapping_yaw_rate_{0.30};
  bool native_confidence_filter_enabled_{false};
  bool native_temporal_support_enabled_{false};
  bool native_temporal_patch_enabled_{false};
  int native_lidar_ground_patch_minimum_frames_{3};
  double native_lidar_ground_patch_minimum_span_{0.30};
  bool scan_surface_filter_enabled_{false};
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_lidar_publisher_;
  std::size_t surface_checked_measurements_{0},surface_rejected_measurements_{0};
  double surface_filter_total_seconds_{0.0},surface_filter_maximum_seconds_{0.0};
  bool native_ground_temporal_filter_enabled_{false};
  int native_temporal_minimum_frames_{3};
  double native_temporal_minimum_span_{0.30};
  bool native_rejected_evidence_enabled_{false};
  int native_minimum_distinct_scans_{2};
  int native_neighbor_radius_voxels_{1};
  int native_minimum_neighbor_voxels_{1};
  bool native_preserve_ground_{true};
  double native_ground_tolerance_{0.10};
  double ground_plane_z_{0.0};
  double ground_snap_tolerance_{0.10};
  Eigen::Vector3d lidar_translation_{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond lidar_rotation_{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d self_min_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d self_max_{Eigen::Vector3d::Zero()};

  std::mutex odom_mutex_;
  std::recursive_mutex acquisition_mutex_;
  std::deque<OdomSample> odom_samples_;
  std::mutex map_mutex_;
  std::unordered_map<VoxelKey, VoxelAccumulator, VoxelKeyHash> voxels_;
  std::unordered_map<VoxelKey, ObservationSupport, VoxelKeyHash> patch_observations_;
  std::size_t processed_scans_{0};
  std::size_t received_scans_{0};
  std::size_t skipped_scans_{0};
  std::size_t dropped_scans_{0};
  std::size_t motion_rejected_scans_{0};
  double absolute_time_offset_sum_{0.0};
  double max_absolute_time_offset_{0.0};

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr depth_subscription_;
  rclcpp::CallbackGroup::SharedPtr odom_callback_group_;
  rclcpp::CallbackGroup::SharedPtr lidar_callback_group_;
  rclcpp::CallbackGroup::SharedPtr service_callback_group_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr preview_publisher_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_service_;
};

}  // namespace robotac_mapping_baseline

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<robotac_mapping_baseline::OdomVoxelMapper>();
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 3U);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
