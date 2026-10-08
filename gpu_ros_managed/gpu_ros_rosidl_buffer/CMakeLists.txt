cmake_minimum_required(VERSION 3.22.1)
project(gpu_ros_rosidl_buffer LANGUAGES CXX)

option(GPU_ROS_ROSIDL_BUFFER_BUILD_CUDA "Build official CUDA Buffer read access" OFF)

find_package(ament_cmake REQUIRED)
find_package(rosidl_buffer REQUIRED)

add_library(gpu_ros_rosidl_buffer SHARED src/cpu_buffer_access.cpp)
add_library(gpu_ros::rosidl_buffer ALIAS gpu_ros_rosidl_buffer)
set_target_properties(gpu_ros_rosidl_buffer PROPERTIES EXPORT_NAME rosidl_buffer)
target_compile_features(gpu_ros_rosidl_buffer PUBLIC cxx_std_17)
target_include_directories(gpu_ros_rosidl_buffer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(gpu_ros_rosidl_buffer PUBLIC rosidl_buffer)

set(buffer_targets gpu_ros_rosidl_buffer)
if(GPU_ROS_ROSIDL_BUFFER_BUILD_CUDA)
  # An explicitly enabled backend must never silently fall back to CPU.
  find_package(CUDAToolkit REQUIRED)
  find_package(cuda_buffer REQUIRED)
  add_library(gpu_ros_rosidl_buffer_cuda SHARED src/cuda_buffer_access.cpp)
  add_library(gpu_ros::rosidl_buffer_cuda ALIAS gpu_ros_rosidl_buffer_cuda)
  set_target_properties(gpu_ros_rosidl_buffer_cuda PROPERTIES EXPORT_NAME rosidl_buffer_cuda)
  target_link_libraries(gpu_ros_rosidl_buffer_cuda PUBLIC gpu_ros_rosidl_buffer
    PRIVATE CUDA::cudart)
  ament_target_dependencies(gpu_ros_rosidl_buffer_cuda PRIVATE cuda_buffer)
  list(APPEND buffer_targets gpu_ros_rosidl_buffer_cuda)
  ament_export_dependencies(CUDAToolkit cuda_buffer)
endif()

install(TARGETS ${buffer_targets} EXPORT gpu_ros_rosidl_bufferTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT gpu_ros_rosidl_bufferTargets
  FILE gpu_ros_rosidl_bufferTargets.cmake
  NAMESPACE gpu_ros::
  DESTINATION share/${PROJECT_NAME}/cmake)
ament_export_targets(gpu_ros_rosidl_bufferTargets HAS_LIBRARY_TARGET NAMESPACE gpu_ros::)
ament_export_dependencies(rosidl_buffer)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_rosidl_buffer_cpu test/test_rosidl_buffer_cpu.cpp)
  if(TARGET test_rosidl_buffer_cpu)
    target_link_libraries(test_rosidl_buffer_cpu gpu_ros_rosidl_buffer)
  endif()
  if(GPU_ROS_ROSIDL_BUFFER_BUILD_CUDA)
    find_package(Threads REQUIRED)
    ament_add_gtest(test_rosidl_buffer_cuda test/test_rosidl_buffer_cuda.cpp)
    if(TARGET test_rosidl_buffer_cuda)
      target_link_libraries(test_rosidl_buffer_cuda
        gpu_ros_rosidl_buffer_cuda CUDA::cudart Threads::Threads)
      ament_target_dependencies(test_rosidl_buffer_cuda cuda_buffer)
      set_tests_properties(test_rosidl_buffer_cuda PROPERTIES
        SKIP_REGULAR_EXPRESSION "Buffer CUDA acceptance requires a real CUDA device")
    endif()
  endif()
endif()

ament_package()
