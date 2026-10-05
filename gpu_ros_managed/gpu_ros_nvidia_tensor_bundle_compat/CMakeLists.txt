cmake_minimum_required(VERSION 3.22.1)
project(gpu_ros_nvidia_tensor_bundle_compat LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(isaac_ros_tensor_msgs REQUIRED)
find_package(tensor_msgs REQUIRED)
find_package(gpu_ros_tensor_bundle_msgs REQUIRED)
find_package(rosidl_buffer REQUIRED)
find_package(cuda_buffer REQUIRED)
find_package(gpu_ros_managed_core REQUIRED)
find_package(gpu_ros_managed_cuda REQUIRED)
find_package(gpu_ros_managed_ros REQUIRED)
find_package(gpu_ros_managed_tensor_bundle REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(gpu_ros_nvidia_tensor_bundle_compat SHARED
  src/tensor_bundle_conversion.cpp
  src/tensor_list_buffer_adapter.cpp
  src/tensor_bundle_compat_nodes.cpp)
add_library(gpu_ros::nvidia_tensor_bundle_compat ALIAS
  gpu_ros_nvidia_tensor_bundle_compat)
target_include_directories(gpu_ros_nvidia_tensor_bundle_compat PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(gpu_ros_nvidia_tensor_bundle_compat PUBLIC cxx_std_20)
ament_target_dependencies(gpu_ros_nvidia_tensor_bundle_compat
  rclcpp rclcpp_components isaac_ros_tensor_msgs tensor_msgs
  gpu_ros_tensor_bundle_msgs rosidl_buffer cuda_buffer
  gpu_ros_managed_core gpu_ros_managed_ros gpu_ros_managed_tensor_bundle)
target_link_libraries(gpu_ros_nvidia_tensor_bundle_compat gpu_ros_managed::cuda CUDA::cudart)

rclcpp_components_register_nodes(gpu_ros_nvidia_tensor_bundle_compat
  "gpu_ros::nvidia_tensor_bundle_compat::NvidiaTensorListToTensorBundleNode"
  "gpu_ros::nvidia_tensor_bundle_compat::TensorBundleToNvidiaTensorListNode")

install(TARGETS gpu_ros_nvidia_tensor_bundle_compat
  EXPORT gpu_ros_nvidia_tensor_bundle_compatTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(EXPORT gpu_ros_nvidia_tensor_bundle_compatTargets
  FILE gpu_ros_nvidia_tensor_bundle_compatTargets.cmake
  NAMESPACE gpu_ros::
  DESTINATION share/${PROJECT_NAME}/cmake)
ament_export_targets(gpu_ros_nvidia_tensor_bundle_compatTargets HAS_LIBRARY_TARGET
  NAMESPACE gpu_ros::)
ament_export_dependencies(
  CUDAToolkit rclcpp rclcpp_components isaac_ros_tensor_msgs tensor_msgs
  gpu_ros_tensor_bundle_msgs rosidl_buffer cuda_buffer
  gpu_ros_managed_core gpu_ros_managed_cuda gpu_ros_managed_ros gpu_ros_managed_tensor_bundle)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_tensor_bundle_conversion test/test_tensor_bundle_conversion.cpp)
  if(TARGET test_tensor_bundle_conversion)
    target_link_libraries(test_tensor_bundle_conversion
      gpu_ros_nvidia_tensor_bundle_compat)
    ament_target_dependencies(test_tensor_bundle_conversion
      isaac_ros_tensor_msgs tensor_msgs gpu_ros_tensor_bundle_msgs)
  endif()
  ament_add_gtest(test_tensor_list_buffer_adapter test/test_tensor_list_buffer_adapter.cpp)
  target_link_libraries(test_tensor_list_buffer_adapter
    gpu_ros_nvidia_tensor_bundle_compat gpu_ros_managed::cuda CUDA::cudart)
endif()

ament_package()
