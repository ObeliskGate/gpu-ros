cmake_minimum_required(VERSION 3.22.1)
project(gpu_ros_rtdetr LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
find_package(gpu_ros_detection_common REQUIRED)
find_package(gpu_ros_managed_core QUIET)
find_package(gpu_ros_managed_hip QUIET)
find_package(gpu_ros_managed_ros QUIET)
find_package(gpu_ros_managed_tensor_bundle QUIET)

# RtDetrImageEncoderNode
ament_auto_add_library(rtdetr_image_encoder_node SHARED src/rtdetr_image_encoder_node.cpp)
target_include_directories(rtdetr_image_encoder_node PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(rtdetr_image_encoder_node
  ${OpenCV_LIBRARIES} gpu_ros_detection_common::core)
rclcpp_components_register_nodes(rtdetr_image_encoder_node
  "gpu_ros::rtdetr::RtDetrImageEncoderNode")

# RtDetrPreprocessorNode
ament_auto_add_library(rtdetr_preprocessor_node SHARED src/rtdetr_preprocessor_node.cpp)
rclcpp_components_register_nodes(rtdetr_preprocessor_node
  "gpu_ros::rtdetr::RtDetrPreprocessorNode")

# RtDetrDecoderNode
ament_auto_add_library(rtdetr_decoder_core SHARED src/rtdetr_decoder.cpp)
ament_auto_add_library(rtdetr_decoder_node SHARED src/rtdetr_decoder_node.cpp)
rclcpp_components_register_nodes(rtdetr_decoder_node
  "gpu_ros::rtdetr::RtDetrDecoderNode")
target_link_libraries(rtdetr_decoder_node rtdetr_decoder_core)

if(TARGET gpu_ros_managed::hip AND TARGET gpu_ros_managed::ros AND
  TARGET gpu_ros_managed::core AND TARGET gpu_ros_managed::tensor_bundle AND
  TARGET gpu_ros_detection_common::hip)
  ament_auto_add_library(rtdetr_managed_hip_nodes SHARED
    src/rtdetr_managed_hip_nodes.cpp)
  target_link_libraries(rtdetr_managed_hip_nodes
    gpu_ros_managed::hip gpu_ros_managed::ros gpu_ros_managed::core
    gpu_ros_managed::tensor_bundle gpu_ros_detection_common::hip rtdetr_decoder_core)
  rclcpp_components_register_nodes(rtdetr_managed_hip_nodes
    "gpu_ros::rtdetr::RtDetrManagedHipImageEncoderNode"
    "gpu_ros::rtdetr::RtDetrManagedHipPreprocessorNode"
    "gpu_ros::rtdetr::RtDetrManagedHipDecoderNode")
else()
  message(STATUS "RT-DETR Managed HIP plugins disabled: HIP managed/common targets unavailable")
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_rtdetr_decoder test/test_rtdetr_decoder.cpp)
  if(TARGET test_rtdetr_decoder)
    target_link_libraries(test_rtdetr_decoder rtdetr_decoder_core)
  endif()

  option(
    BUILD_MIGRAPHX_POL_TEST
    "Build the standard ROS2 + ONNX Runtime MIGraphX proof-of-life test"
    OFF
  )
  if(BUILD_MIGRAPHX_POL_TEST)
    find_package(launch_testing_ament_cmake REQUIRED)
    add_launch_test(test/gpu_ros_std_rtdetr_pol_test.py TIMEOUT "300")
  endif()
endif()

install(PROGRAMS
  scripts/prepare_rtdetr_models.sh
  scripts/convert_rtdetr_fp16.py
  scripts/compare_detections.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_auto_package(INSTALL_TO_SHARE launch)
