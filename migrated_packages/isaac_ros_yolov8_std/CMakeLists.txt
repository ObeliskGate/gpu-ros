cmake_minimum_required(VERSION 3.22.1)
project(isaac_ros_yolov8_std LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc dnn)
ament_auto_find_build_dependencies()
find_package(isaac_ros_detection_common REQUIRED)
find_package(gpu_ros_managed_core QUIET)
find_package(gpu_ros_managed_hip QUIET)
find_package(gpu_ros_managed_ros QUIET)
find_package(gpu_ros_managed_tensor_list QUIET)

ament_auto_add_library(yolov8_image_encoder_core SHARED src/yolov8_image_encoder.cpp)
target_include_directories(yolov8_image_encoder_core PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolov8_image_encoder_core
  ${OpenCV_LIBRARIES} isaac_ros_detection_common)

ament_auto_add_library(yolov8_image_encoder_node SHARED src/yolov8_image_encoder_node.cpp)
target_link_libraries(yolov8_image_encoder_node yolov8_image_encoder_core)
rclcpp_components_register_nodes(yolov8_image_encoder_node
  "nvidia::isaac_ros::yolov8_std::YoloV8ImageEncoderNode")

ament_auto_add_library(yolov8_decoder_core SHARED src/yolov8_decoder.cpp)
target_include_directories(yolov8_decoder_core PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolov8_decoder_core ${OpenCV_LIBRARIES})

ament_auto_add_library(yolov8_decoder_node SHARED src/yolov8_decoder_node.cpp)
target_link_libraries(yolov8_decoder_node yolov8_decoder_core)
rclcpp_components_register_nodes(yolov8_decoder_node
  "nvidia::isaac_ros::yolov8_std::YoloV8DecoderNode")

if(TARGET gpu_ros_managed::hip AND TARGET gpu_ros_managed::ros AND
  TARGET gpu_ros_managed::core AND TARGET gpu_ros_managed::tensor_list AND
  TARGET isaac_ros_detection_common_hip)
  ament_auto_add_library(yolov8_managed_hip_nodes SHARED
    src/yolov8_managed_hip_nodes.cpp)
  target_link_libraries(yolov8_managed_hip_nodes
    gpu_ros_managed::hip gpu_ros_managed::ros gpu_ros_managed::core
    gpu_ros_managed::tensor_list isaac_ros_detection_common_hip yolov8_decoder_core)
  rclcpp_components_register_nodes(yolov8_managed_hip_nodes
    "nvidia::isaac_ros::yolov8_std::YoloV8ManagedHipImageEncoderNode"
    "nvidia::isaac_ros::yolov8_std::YoloV8ManagedHipDecoderNode")
else()
  message(STATUS "YOLOv8 Managed HIP plugins disabled: HIP managed/common targets unavailable")
endif()

option(
  BUILD_NVIDIA_YOLOV8_POL_TEST
  "Build the existing NVIDIA std YOLOv8 proof-of-life test (opt-in)"
  OFF
)
option(
  BUILD_YOLOV8_MIGRAPHX_POL_TEST
  "Build the AMD YOLOv8 MIGraphX tiny-model proof-of-life test"
  OFF
)

# The AMD profile deliberately does not provide isaac_ros_test, which is only
# needed by the legacy NVIDIA POL. Force the legacy option off for the AMD
# MIGraphX profile so a stale CMake cache cannot re-register that test.
if(NOT BUILD_NITROS_TRANSPORT AND ORT_ENABLE_MIGRAPHX)
  set(BUILD_NVIDIA_YOLOV8_POL_TEST OFF CACHE BOOL
    "Build the existing NVIDIA std YOLOv8 proof-of-life test (opt-in)" FORCE)
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_yolov8_image_encoder test/test_yolov8_image_encoder.cpp)
  if(TARGET test_yolov8_image_encoder)
    target_link_libraries(test_yolov8_image_encoder yolov8_image_encoder_core)
  endif()

  ament_add_gtest(test_yolov8_decoder test/test_yolov8_decoder.cpp)
  if(TARGET test_yolov8_decoder)
    target_link_libraries(test_yolov8_decoder yolov8_decoder_core)
  endif()

  if(BUILD_NVIDIA_YOLOV8_POL_TEST OR BUILD_YOLOV8_MIGRAPHX_POL_TEST)
    find_package(launch_testing_ament_cmake REQUIRED)
  endif()
  if(BUILD_NVIDIA_YOLOV8_POL_TEST)
    add_launch_test(test/isaac_ros_std_yolov8_pol_test.py TIMEOUT "600")
  endif()
  if(BUILD_YOLOV8_MIGRAPHX_POL_TEST)
    add_launch_test(test/isaac_ros_yolov8_migraphx_pol_test.py TIMEOUT "900")
  endif()
endif()

ament_auto_package(INSTALL_TO_SHARE launch)
