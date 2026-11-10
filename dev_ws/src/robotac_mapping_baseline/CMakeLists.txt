cmake_minimum_required(VERSION 3.8)
project(robotac_mapping_baseline)

# Full-map preview and confidence queries run alongside live sensor callbacks.
# Keep an explicit caller-selected Debug build, but never default to unoptimized
# code when this package is built outside the provided deployment script.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters io registration)

add_executable(odom_voxel_mapper src/odom_voxel_mapper.cpp)
target_compile_features(odom_voxel_mapper PUBLIC cxx_std_17)
target_include_directories(odom_voxel_mapper PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(odom_voxel_mapper ${PCL_LIBRARIES})
ament_target_dependencies(
  odom_voxel_mapper
  rclcpp
  sensor_msgs
  nav_msgs
  std_srvs
  pcl_conversions
)

add_executable(gicp_scan_mapper src/gicp_scan_mapper.cpp)
target_compile_features(gicp_scan_mapper PUBLIC cxx_std_17)
target_include_directories(gicp_scan_mapper PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(gicp_scan_mapper ${PCL_LIBRARIES})
ament_target_dependencies(
  gicp_scan_mapper
  rclcpp
  sensor_msgs
  nav_msgs
  std_srvs
  pcl_conversions
)

install(TARGETS odom_voxel_mapper gicp_scan_mapper DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_package()
