#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <nav_msgs/msg/odometry.hpp>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace robotac_mapping_baseline
{

double gicp_stamp_seconds(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
}

struct GicpOdomSample
{
  double stamp{0.0};
  Eigen::Matrix4f pose{Eigen::Matrix4f::Identity()};
};

struct VoxelKeyG
{
  int64_t x;
  int64_t y;
  int64_t z;
  bool operator==(const VoxelKeyG & other) const
  {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelKeyGHash
{
  std::size_t operator()(const VoxelKeyG & key) const
  {
    std::size_t seed = std::hash<int64_t>{}(key.x);
    seed ^= std::hash<int64_t>{}(key.y) + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
    seed ^= std::hash<int64_t>{}(key.z) + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
    return seed;
  }
};

struct VoxelAccumulatorG
{
  Eigen::Vector3d sum{Eigen::Vector3d::Zero()};
  uint32_t count{0};
  void add(const Eigen::Vector3d & point) {sum += point; ++count;}
};

class GicpScanMapper : public rclcpp::Node
{
public:
  GicpScanMapper()
  : Node("gicp_scan_mapper")
  {
    lidar_topic_ = declare_parameter<std::string>("lidar_topic", "/front_lidar");
    odom_topic_ = declare_parameter<std::string>("odom_topic", "/odom/mujoco_odom");
    use_odom_prior_ = declare_parameter<bool>("use_odom_prior", false);
    max_odom_time_offset_ = declare_parameter<double>("max_odom_time_offset", 0.02);
    save_service_name_ = declare_parameter<std::string>(
      "save_service", "/robotac/save_gicp_map");
    save_path_ = declare_parameter<std::string>(
      "save_path", "/workspace/dev_ws/maps/gicp_baseline.pcd");
    map_voxel_size_ = declare_parameter<double>("map_voxel_size", 0.035);
    registration_voxel_size_ = declare_parameter<double>("registration_voxel_size", 0.10);
    min_range_ = declare_parameter<double>("min_range", 0.20);
    max_range_ = declare_parameter<double>("max_range", 40.0);
    max_correspondence_distance_ = declare_parameter<double>(
      "max_correspondence_distance", 0.75);
    max_iterations_ = declare_parameter<int>("max_iterations", 50);
    max_fitness_score_ = declare_parameter<double>("max_fitness_score", 0.12);
    process_every_n_scans_ = declare_parameter<int>("process_every_n_scans", 1);
    self_filter_enabled_ = declare_parameter<bool>("self_filter_enabled", true);
    lidar_translation_ = vector3_parameter("lidar_translation", {0.0, 0.0, 0.30});
    const auto rpy = vector3_parameter("lidar_rpy", {0.0, 0.0, 0.0});
    self_min_ = vector3_parameter("self_min", {-0.45, -0.30, 0.00});
    self_max_ = vector3_parameter("self_max", {0.45, 0.30, 0.55});
    lidar_rotation_ =
      Eigen::AngleAxisd(rpy.z(), Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(rpy.y(), Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(rpy.x(), Eigen::Vector3d::UnitX());

    if (
      map_voxel_size_ <= 0.0 || registration_voxel_size_ <= 0.0 ||
      process_every_n_scans_ <= 0)
    {
      throw std::invalid_argument("voxel sizes and process_every_n_scans must be positive");
    }

    auto offline_qos = rclcpp::QoS(rclcpp::KeepAll()).best_effort();
    if (use_odom_prior_) {
      odom_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_, rclcpp::SensorDataQoS(),
        std::bind(&GicpScanMapper::odom_callback, this, std::placeholders::_1));
    }
    lidar_subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      lidar_topic_, offline_qos,
      std::bind(&GicpScanMapper::lidar_callback, this, std::placeholders::_1));
    save_service_ = create_service<std_srvs::srv::Trigger>(
      save_service_name_,
      std::bind(
        &GicpScanMapper::save_callback, this,
        std::placeholders::_1, std::placeholders::_2));
    RCLCPP_INFO(
      get_logger(), "Pairwise GICP mapping %s, registration voxel=%.3f m, odom_prior=%d",
      lidar_topic_.c_str(), registration_voxel_size_, use_odom_prior_);
  }

private:
  Eigen::Vector3d vector3_parameter(
    const std::string & name, const std::array<double, 3> & fallback)
  {
    const auto values = declare_parameter<std::vector<double>>(
      name, std::vector<double>(fallback.begin(), fallback.end()));
    if (values.size() != 3U) {
      throw std::invalid_argument(name + " must have exactly three values");
    }
    return {values[0], values[1], values[2]};
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_scan(
    const sensor_msgs::msg::PointCloud2 & message)
  {
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    cloud->reserve(message.width * message.height);
    sensor_msgs::PointCloud2ConstIterator<float> x(message, "x");
    sensor_msgs::PointCloud2ConstIterator<float> y(message, "y");
    sensor_msgs::PointCloud2ConstIterator<float> z(message, "z");
    for (; x != x.end(); ++x, ++y, ++z) {
      const Eigen::Vector3d lidar(*x, *y, *z);
      if (!lidar.allFinite() || lidar.norm() < min_range_ || lidar.norm() > max_range_) {
        continue;
      }
      const Eigen::Vector3d base = lidar_rotation_ * lidar + lidar_translation_;
      if (
        self_filter_enabled_ && (base.array() >= self_min_.array()).all() &&
        (base.array() <= self_max_.array()).all())
      {
        continue;
      }
      cloud->push_back(pcl::PointXYZ(base.x(), base.y(), base.z()));
    }
    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
  }

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr message)
  {
    GicpOdomSample sample;
    sample.stamp = gicp_stamp_seconds(message->header.stamp);
    const Eigen::Quaternionf orientation(
      message->pose.pose.orientation.w,
      message->pose.pose.orientation.x,
      message->pose.pose.orientation.y,
      message->pose.pose.orientation.z);
    sample.pose.block<3, 3>(0, 0) = orientation.normalized().toRotationMatrix();
    sample.pose(0, 3) = message->pose.pose.position.x;
    sample.pose(1, 3) = message->pose.pose.position.y;
    sample.pose(2, 3) = message->pose.pose.position.z;
    std::lock_guard<std::mutex> lock(odom_mutex_);
    odom_samples_.push_back(sample);
    while (!odom_samples_.empty() && sample.stamp - odom_samples_.front().stamp > 5.0) {
      odom_samples_.pop_front();
    }
  }

  bool nearest_odom(double stamp, GicpOdomSample & output)
  {
    std::lock_guard<std::mutex> lock(odom_mutex_);
    if (odom_samples_.empty()) {
      return false;
    }
    const auto iterator = std::lower_bound(
      odom_samples_.begin(), odom_samples_.end(), stamp,
      [](const GicpOdomSample & sample, double value) {return sample.stamp < value;});
    const GicpOdomSample * best = iterator == odom_samples_.end() ? nullptr : &(*iterator);
    if (iterator != odom_samples_.begin()) {
      const auto & previous = *std::prev(iterator);
      if (best == nullptr || std::abs(previous.stamp - stamp) < std::abs(best->stamp - stamp)) {
        best = &previous;
      }
    }
    if (best == nullptr || std::abs(best->stamp - stamp) > max_odom_time_offset_) {
      return false;
    }
    output = *best;
    return true;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr downsample(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & input)
  {
    auto output = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::VoxelGrid<pcl::PointXYZ> filter;
    filter.setLeafSize(
      registration_voxel_size_, registration_voxel_size_, registration_voxel_size_);
    filter.setInputCloud(input);
    filter.filter(*output);
    return output;
  }

  void accumulate(
    const pcl::PointCloud<pcl::PointXYZ> & scan, const Eigen::Matrix4f & pose)
  {
    pcl::PointCloud<pcl::PointXYZ> world;
    pcl::transformPointCloud(scan, world, pose);
    for (const auto & point : world) {
      const VoxelKeyG key{
        static_cast<int64_t>(std::floor(point.x / map_voxel_size_)),
        static_cast<int64_t>(std::floor(point.y / map_voxel_size_)),
        static_cast<int64_t>(std::floor(point.z / map_voxel_size_))};
      voxels_[key].add(Eigen::Vector3d(point.x, point.y, point.z));
    }
  }

  void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr message)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++received_scans_;
    if ((received_scans_ - 1U) % static_cast<std::size_t>(process_every_n_scans_) != 0U) {
      ++skipped_scans_;
      return;
    }
    GicpOdomSample current_odom;
    if (use_odom_prior_ && !nearest_odom(gicp_stamp_seconds(message->header.stamp), current_odom)) {
      ++prior_misses_;
      return;
    }
    auto full = filtered_scan(*message);
    auto current = downsample(full);
    if (current->size() < 100U) {
      ++rejected_scans_;
      return;
    }

    if (!previous_scan_) {
      pose_.setIdentity();
      previous_scan_ = current;
      if (use_odom_prior_) {
        previous_odom_ = current_odom;
      }
      accumulate(*full, pose_);
      ++accepted_scans_;
      return;
    }

    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
    gicp.setMaximumIterations(max_iterations_);
    gicp.setMaxCorrespondenceDistance(max_correspondence_distance_);
    gicp.setTransformationEpsilon(1e-4);
    gicp.setEuclideanFitnessEpsilon(1e-4);
    gicp.setInputSource(current);
    gicp.setInputTarget(previous_scan_);
    pcl::PointCloud<pcl::PointXYZ> aligned;
    Eigen::Matrix4f registration_guess = last_relative_;
    if (use_odom_prior_) {
      registration_guess = previous_odom_.pose.inverse() * current_odom.pose;
    }
    gicp.align(aligned, registration_guess);
    const double fitness = gicp.getFitnessScore(max_correspondence_distance_);
    if (!gicp.hasConverged() || !std::isfinite(fitness) || fitness > max_fitness_score_) {
      ++rejected_scans_;
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "GICP rejected scan: converged=%d fitness=%.5f accepted=%zu rejected=%zu",
        gicp.hasConverged(), fitness, accepted_scans_, rejected_scans_);
      return;
    }

    last_relative_ = gicp.getFinalTransformation();
    last_prior_correction_translation_ =
      (registration_guess.inverse() * last_relative_).block<3, 1>(0, 3).norm();
    pose_ = pose_ * last_relative_;
    previous_scan_ = current;
    if (use_odom_prior_) {
      previous_odom_ = current_odom;
    }
    accumulate(*full, pose_);
    last_fitness_ = fitness;
    ++accepted_scans_;
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "accepted=%zu rejected=%zu fitness=%.5f correction=%.3f m voxels=%zu pose_xyz=[%.3f %.3f %.3f]",
      accepted_scans_, rejected_scans_, last_fitness_,
      last_prior_correction_translation_, voxels_.size(),
      pose_(0, 3), pose_(1, 3), pose_(2, 3));
  }

  bool save_map(const std::string & path)
  {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.reserve(voxels_.size());
    for (const auto & item : voxels_) {
      const Eigen::Vector3d point = item.second.sum / static_cast<double>(item.second.count);
      cloud.push_back(pcl::PointXYZ(point.x(), point.y(), point.z()));
    }
    cloud.width = cloud.size();
    cloud.height = 1;
    cloud.is_dense = true;
    const std::filesystem::path output(path);
    std::filesystem::create_directories(output.parent_path());
    const auto temporary = output.string() + ".tmp.pcd";
    if (pcl::io::savePCDFileBinary(temporary, cloud) != 0) {
      return false;
    }
    std::filesystem::rename(temporary, output);
    return true;
  }

  void save_callback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    response->success = save_map(save_path_);
    response->message = save_path_ + " accepted=" + std::to_string(accepted_scans_) +
      " rejected=" + std::to_string(rejected_scans_) +
      " skipped=" + std::to_string(skipped_scans_) +
      " prior_misses=" + std::to_string(prior_misses_) +
      " fitness=" + std::to_string(last_fitness_);
    RCLCPP_INFO(get_logger(), "%s", response->message.c_str());
  }

  std::string lidar_topic_;
  std::string odom_topic_;
  std::string save_service_name_;
  std::string save_path_;
  double map_voxel_size_{0.035};
  double registration_voxel_size_{0.10};
  double min_range_{0.20};
  double max_range_{40.0};
  double max_correspondence_distance_{0.75};
  int max_iterations_{50};
  double max_fitness_score_{0.12};
  bool use_odom_prior_{false};
  double max_odom_time_offset_{0.02};
  int process_every_n_scans_{1};
  bool self_filter_enabled_{true};
  Eigen::Vector3d lidar_translation_{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond lidar_rotation_{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d self_min_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d self_max_{Eigen::Vector3d::Zero()};
  Eigen::Matrix4f pose_{Eigen::Matrix4f::Identity()};
  Eigen::Matrix4f last_relative_{Eigen::Matrix4f::Identity()};
  GicpOdomSample previous_odom_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr previous_scan_;
  std::unordered_map<VoxelKeyG, VoxelAccumulatorG, VoxelKeyGHash> voxels_;
  std::mutex mutex_;
  std::mutex odom_mutex_;
  std::deque<GicpOdomSample> odom_samples_;
  std::size_t accepted_scans_{0};
  std::size_t rejected_scans_{0};
  std::size_t received_scans_{0};
  std::size_t skipped_scans_{0};
  std::size_t prior_misses_{0};
  double last_fitness_{0.0};
  double last_prior_correction_translation_{0.0};
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_service_;
};

}  // namespace robotac_mapping_baseline

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<robotac_mapping_baseline::GicpScanMapper>());
  rclcpp::shutdown();
  return 0;
}
