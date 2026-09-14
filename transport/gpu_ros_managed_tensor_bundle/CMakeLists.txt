cmake_minimum_required(VERSION 3.22)
project(gpu_ros_managed_tensor_bundle LANGUAGES CXX)
find_package(ament_cmake REQUIRED)
find_package(gpu_ros_managed_core REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(gpu_ros_tensor_bundle_msgs REQUIRED)
add_library(gpu_ros_managed_tensor_bundle SHARED src/tensor_bundle.cpp)
add_library(gpu_ros_managed::tensor_bundle ALIAS gpu_ros_managed_tensor_bundle)
set_target_properties(gpu_ros_managed_tensor_bundle PROPERTIES
  EXPORT_NAME tensor_bundle
  POSITION_INDEPENDENT_CODE ON)
target_compile_features(gpu_ros_managed_tensor_bundle PUBLIC cxx_std_17)
target_include_directories(gpu_ros_managed_tensor_bundle PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
# ament_target_dependencies() uses CMake's plain signature on Jazzy, so this
# target must use that signature consistently.  The plain form is transitive.
target_link_libraries(gpu_ros_managed_tensor_bundle gpu_ros_managed::core)
ament_target_dependencies(gpu_ros_managed_tensor_bundle
  rclcpp std_msgs gpu_ros_tensor_bundle_msgs)
install(TARGETS gpu_ros_managed_tensor_bundle EXPORT gpu_ros_managed_tensor_bundleTargets
  LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT gpu_ros_managed_tensor_bundleTargets
  FILE gpu_ros_managed_tensor_bundleTargets.cmake
  NAMESPACE gpu_ros_managed::
  DESTINATION share/gpu_ros_managed_tensor_bundle/cmake)
ament_export_targets(gpu_ros_managed_tensor_bundleTargets HAS_LIBRARY_TARGET
  NAMESPACE gpu_ros_managed::)
ament_export_dependencies(gpu_ros_managed_core rclcpp std_msgs gpu_ros_tensor_bundle_msgs)
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(gpu_ros_managed_ros REQUIRED)
  ament_add_gtest(type_adapter_probe test/type_adapter_probe.cpp)
  target_link_libraries(type_adapter_probe gpu_ros_managed_tensor_bundle)
  ament_target_dependencies(type_adapter_probe
    rclcpp gpu_ros_managed_ros gpu_ros_tensor_bundle_msgs)
endif()
ament_package()
