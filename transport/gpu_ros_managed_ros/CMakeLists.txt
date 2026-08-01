cmake_minimum_required(VERSION 3.22)
project(gpu_ros_managed_ros LANGUAGES CXX)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
add_library(gpu_ros_managed_ros INTERFACE)
add_library(gpu_ros_managed::ros ALIAS gpu_ros_managed_ros)
set_target_properties(gpu_ros_managed_ros PROPERTIES EXPORT_NAME ros)
target_compile_features(gpu_ros_managed_ros INTERFACE cxx_std_17)
target_include_directories(gpu_ros_managed_ros INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
install(TARGETS gpu_ros_managed_ros EXPORT gpu_ros_managed_rosTargets)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT gpu_ros_managed_rosTargets
  FILE gpu_ros_managed_rosTargets.cmake
  NAMESPACE gpu_ros_managed::
  DESTINATION share/gpu_ros_managed_ros/cmake)
ament_export_targets(gpu_ros_managed_rosTargets NAMESPACE gpu_ros_managed::)
ament_export_dependencies(rclcpp)
ament_package()
