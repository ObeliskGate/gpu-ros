cmake_minimum_required(VERSION 3.22.1)
project(gpu_ros_detection_common LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
find_package(hip CONFIG QUIET)

add_library(gpu_ros_detection_common SHARED src/image_preprocess.cpp)
add_library(gpu_ros_detection_common::core ALIAS gpu_ros_detection_common)
set_target_properties(gpu_ros_detection_common PROPERTIES EXPORT_NAME core)
target_include_directories(gpu_ros_detection_common PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(gpu_ros_detection_common sensor_msgs)
target_link_libraries(gpu_ros_detection_common ${OpenCV_LIBRARIES})

install(TARGETS gpu_ros_detection_common
  EXPORT gpu_ros_detection_commonTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

if(hip_FOUND)
  enable_language(HIP)
  add_library(gpu_ros_detection_common_hip SHARED src/hip_preprocess.hip)
  add_library(gpu_ros_detection_common::hip ALIAS gpu_ros_detection_common_hip)
  set_target_properties(gpu_ros_detection_common_hip PROPERTIES EXPORT_NAME hip)
  target_include_directories(gpu_ros_detection_common_hip PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
  target_link_libraries(gpu_ros_detection_common_hip PUBLIC
    gpu_ros_detection_common hip::host)
  target_compile_features(gpu_ros_detection_common_hip PUBLIC cxx_std_17)
  install(TARGETS gpu_ros_detection_common_hip
    EXPORT gpu_ros_detection_commonTargets
    ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
  ament_export_dependencies(hip)
endif()

ament_export_targets(gpu_ros_detection_commonTargets HAS_LIBRARY_TARGET
  NAMESPACE gpu_ros_detection_common::)
ament_export_dependencies(sensor_msgs)
ament_export_dependencies(OpenCV)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_image_preprocess test/test_image_preprocess.cpp)
  if(TARGET test_image_preprocess)
    target_link_libraries(test_image_preprocess gpu_ros_detection_common)
    ament_target_dependencies(test_image_preprocess sensor_msgs)
  endif()
  if(hip_FOUND AND TARGET gpu_ros_detection_common_hip)
    ament_add_gtest(test_hip_preprocess test/test_hip_preprocess.cpp)
    if(TARGET test_hip_preprocess)
      target_link_libraries(test_hip_preprocess gpu_ros_detection_common_hip)
      ament_target_dependencies(test_hip_preprocess sensor_msgs)
    endif()
  endif()
endif()

ament_package()
