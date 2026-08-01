cmake_minimum_required(VERSION 3.22)
project(gpu_ros_managed_tensor_list LANGUAGES CXX)
find_package(ament_cmake REQUIRED)
find_package(gpu_ros_managed_core REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(isaac_ros_tensor_list_interfaces REQUIRED)
add_library(gpu_ros_managed_tensor_list src/tensor_list.cpp)
add_library(gpu_ros_managed::tensor_list ALIAS gpu_ros_managed_tensor_list)
set_target_properties(gpu_ros_managed_tensor_list PROPERTIES EXPORT_NAME tensor_list)
target_compile_features(gpu_ros_managed_tensor_list PUBLIC cxx_std_17)
target_include_directories(gpu_ros_managed_tensor_list PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(gpu_ros_managed_tensor_list PUBLIC gpu_ros_managed::core)
ament_target_dependencies(gpu_ros_managed_tensor_list
  rclcpp std_msgs isaac_ros_tensor_list_interfaces)
install(TARGETS gpu_ros_managed_tensor_list EXPORT gpu_ros_managed_tensor_listTargets
  LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT gpu_ros_managed_tensor_listTargets
  FILE gpu_ros_managed_tensor_listTargets.cmake
  NAMESPACE gpu_ros_managed::
  DESTINATION share/gpu_ros_managed_tensor_list/cmake)
ament_export_targets(gpu_ros_managed_tensor_listTargets HAS_LIBRARY_TARGET
  NAMESPACE gpu_ros_managed::)
ament_export_dependencies(gpu_ros_managed_core rclcpp std_msgs isaac_ros_tensor_list_interfaces)
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(gpu_ros_managed_ros REQUIRED)
  ament_add_gtest(type_adapter_probe test/type_adapter_probe.cpp)
  target_link_libraries(type_adapter_probe PRIVATE gpu_ros_managed_tensor_list)
  ament_target_dependencies(type_adapter_probe
    rclcpp gpu_ros_managed_ros isaac_ros_tensor_list_interfaces)
endif()
ament_package()
