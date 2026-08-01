cmake_minimum_required(VERSION 3.22)
project(gpu_ros_managed_cuda LANGUAGES CXX)

find_package(CUDAToolkit QUIET)
find_package(ament_cmake QUIET)
if(CUDAToolkit_FOUND)
  if(NOT TARGET gpu_ros_managed_core)
    find_package(gpu_ros_managed_core REQUIRED)
  endif()
  add_library(gpu_ros_managed_cuda src/cuda_backend.cpp)
  add_library(gpu_ros_managed::cuda ALIAS gpu_ros_managed_cuda)
  set_target_properties(gpu_ros_managed_cuda PROPERTIES EXPORT_NAME cuda)
  target_compile_features(gpu_ros_managed_cuda PUBLIC cxx_std_17)
  target_include_directories(gpu_ros_managed_cuda PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
  if(TARGET gpu_ros_managed::core)
    set(GPU_ROS_MANAGED_CORE_TARGET gpu_ros_managed::core)
  else()
    set(GPU_ROS_MANAGED_CORE_TARGET gpu_ros_managed_core)
  endif()
  target_link_libraries(gpu_ros_managed_cuda PUBLIC
    ${GPU_ROS_MANAGED_CORE_TARGET} CUDA::cudart)
  install(TARGETS gpu_ros_managed_cuda EXPORT gpu_ros_managed_cudaTargets
    ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
  install(DIRECTORY include/ DESTINATION include)
  install(EXPORT gpu_ros_managed_cudaTargets
    FILE gpu_ros_managed_cudaTargets.cmake
    NAMESPACE gpu_ros_managed::
    DESTINATION share/gpu_ros_managed_cuda/cmake)
  if(ament_cmake_FOUND)
    # Consumers link gpu_ros_managed::cuda, not the package-specific namespace.
    ament_export_targets(gpu_ros_managed_cudaTargets HAS_LIBRARY_TARGET
      NAMESPACE gpu_ros_managed::)
    ament_export_dependencies(gpu_ros_managed_core)
  endif()
  include(CTest)
  if(BUILD_TESTING)
  add_executable(gpu_ros_managed_cuda_stream_test test/cuda_stream_test.cpp)
  target_link_libraries(gpu_ros_managed_cuda_stream_test PRIVATE gpu_ros_managed_cuda)
  target_compile_features(gpu_ros_managed_cuda_stream_test PRIVATE cxx_std_17)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(gpu_ros_managed_cuda_stream_test PRIVATE -UNDEBUG)
  endif()
  add_test(NAME gpu_ros_managed_cuda_stream_test COMMAND gpu_ros_managed_cuda_stream_test)
    set_tests_properties(gpu_ros_managed_cuda_stream_test PROPERTIES SKIP_RETURN_CODE 77)
  endif()
else()
  message(STATUS "CUDA Toolkit not found; gpu_ros_managed_cuda target is unavailable")
endif()
if(ament_cmake_FOUND)
  ament_package()
endif()
