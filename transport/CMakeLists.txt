cmake_minimum_required(VERSION 3.22)
project(gpu_ros_managed_workspace LANGUAGES CXX)

option(GPU_ROS_MANAGED_BUILD_TESTING "Build standalone core tests" ON)
option(GPU_ROS_MANAGED_BUILD_CUDA "Build the CUDA backend when CUDA is available" ON)
option(GPU_ROS_MANAGED_BUILD_HIP "Build the HIP backend when HIP is available" ON)

if(GPU_ROS_MANAGED_BUILD_TESTING)
  include(CTest)
else()
  # Keep package-local CTest blocks disabled for standalone minimal builds.
  set(BUILD_TESTING OFF)
endif()

add_subdirectory(gpu_ros_managed_core)
if(GPU_ROS_MANAGED_BUILD_CUDA)
  add_subdirectory(gpu_ros_managed_cuda)
endif()
if(GPU_ROS_MANAGED_BUILD_HIP)
  add_subdirectory(gpu_ros_managed_hip)
endif()
