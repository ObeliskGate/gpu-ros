cmake_minimum_required(VERSION 3.22.1)
project(isaac_ros_yolov8_std LANGUAGES CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED)
ament_auto_find_build_dependencies()

ament_auto_add_library(yolov8_decoder_core SHARED src/yolov8_decoder.cpp)
target_include_directories(yolov8_decoder_core PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolov8_decoder_core ${OpenCV_LIBRARIES})

ament_auto_add_library(yolov8_decoder_node SHARED src/yolov8_decoder_node.cpp)
target_link_libraries(yolov8_decoder_node yolov8_decoder_core)
rclcpp_components_register_nodes(yolov8_decoder_node
  "nvidia::isaac_ros::yolov8_std::YoloV8DecoderNode")

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_yolov8_decoder test/test_yolov8_decoder.cpp)
  if(TARGET test_yolov8_decoder)
    target_link_libraries(test_yolov8_decoder yolov8_decoder_core)
  endif()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_std_yolov8_pol_test.py TIMEOUT "600")
endif()

ament_auto_package()
