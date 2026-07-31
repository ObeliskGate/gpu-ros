cmake_minimum_required(VERSION 3.22)
project(gpu_ros_managed_core LANGUAGES CXX)

add_library(gpu_ros_managed_core
  src/buffer.cpp
  src/fixed_device_memory_pool.cpp)
add_library(gpu_ros_managed::core ALIAS gpu_ros_managed_core)
set_target_properties(gpu_ros_managed_core PROPERTIES EXPORT_NAME core)
target_compile_features(gpu_ros_managed_core PUBLIC cxx_std_17)
target_include_directories(gpu_ros_managed_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_options(gpu_ros_managed_core PRIVATE -Wall -Wextra -Wpedantic)

install(TARGETS gpu_ros_managed_core EXPORT gpu_ros_managed_coreTargets
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT gpu_ros_managed_coreTargets
  FILE gpu_ros_managed_coreTargets.cmake
  NAMESPACE gpu_ros_managed::
  DESTINATION share/gpu_ros_managed_core/cmake)

find_package(ament_cmake QUIET)
include(CTest)
if(BUILD_TESTING)
  add_executable(gpu_ros_managed_core_test test/core_test.cpp)
  target_link_libraries(gpu_ros_managed_core_test PRIVATE gpu_ros_managed_core)
  target_compile_features(gpu_ros_managed_core_test PRIVATE cxx_std_17)
  target_compile_options(gpu_ros_managed_core_test PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME gpu_ros_managed_core_test COMMAND gpu_ros_managed_core_test)
endif()

if(ament_cmake_FOUND)
  ament_export_targets(gpu_ros_managed_coreTargets HAS_LIBRARY_TARGET)
  ament_package()
endif()
