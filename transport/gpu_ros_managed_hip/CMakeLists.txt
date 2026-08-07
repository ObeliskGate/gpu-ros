cmake_minimum_required(VERSION 3.22)
project(gpu_ros_managed_hip LANGUAGES CXX)
find_package(ament_cmake QUIET)
find_package(hip QUIET)
if(hip_FOUND)
  if(NOT TARGET gpu_ros_managed_core)
    find_package(gpu_ros_managed_core REQUIRED)
  endif()
  add_library(gpu_ros_managed_hip src/hip_backend.cpp)
  add_library(gpu_ros_managed::hip ALIAS gpu_ros_managed_hip)
  set_target_properties(gpu_ros_managed_hip PROPERTIES
    EXPORT_NAME hip
    POSITION_INDEPENDENT_CODE ON)
  target_compile_features(gpu_ros_managed_hip PUBLIC cxx_std_17)
  target_include_directories(gpu_ros_managed_hip PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
  target_link_libraries(gpu_ros_managed_hip PUBLIC gpu_ros_managed::core hip::host)
  install(TARGETS gpu_ros_managed_hip EXPORT gpu_ros_managed_hipTargets
    LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
  install(DIRECTORY include/ DESTINATION include)
  install(EXPORT gpu_ros_managed_hipTargets
    FILE gpu_ros_managed_hipTargets.cmake
    NAMESPACE gpu_ros_managed::
    DESTINATION share/gpu_ros_managed_hip/cmake)
  if(ament_cmake_FOUND)
    ament_export_targets(gpu_ros_managed_hipTargets HAS_LIBRARY_TARGET
      NAMESPACE gpu_ros_managed::)
    ament_export_dependencies(gpu_ros_managed_core)
  endif()
  include(CTest)
  if(BUILD_TESTING)
    add_executable(gpu_ros_managed_hip_copy_test test/hip_copy_test.cpp)
    target_link_libraries(gpu_ros_managed_hip_copy_test PRIVATE gpu_ros_managed_hip)
    target_compile_features(gpu_ros_managed_hip_copy_test PRIVATE cxx_std_17)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
      target_compile_options(gpu_ros_managed_hip_copy_test PRIVATE -UNDEBUG)
    endif()
    add_test(NAME gpu_ros_managed_hip_copy_test COMMAND gpu_ros_managed_hip_copy_test)
    set_tests_properties(gpu_ros_managed_hip_copy_test PROPERTIES SKIP_RETURN_CODE 77)
  endif()
else()
  message(STATUS "HIP SDK not found; gpu_ros_managed_hip target is unavailable")
endif()
if(ament_cmake_FOUND)
  ament_package()
endif()
