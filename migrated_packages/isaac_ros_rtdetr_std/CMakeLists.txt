cmake_minimum_required(VERSION 3.22.1)
project(isaac_ros_rtdetr_std LANGUAGES CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# RtDetrPreprocessorNode
ament_auto_add_library(rtdetr_preprocessor_node SHARED src/rtdetr_preprocessor_node.cpp)
rclcpp_components_register_nodes(rtdetr_preprocessor_node
  "nvidia::isaac_ros::rtdetr_std::RtDetrPreprocessorNode")

# RtDetrDecoderNode
ament_auto_add_library(rtdetr_decoder_node SHARED src/rtdetr_decoder_node.cpp)
rclcpp_components_register_nodes(rtdetr_decoder_node
  "nvidia::isaac_ros::rtdetr_std::RtDetrDecoderNode")

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_std_rtdetr_pol_test.py TIMEOUT "600")
endif()

install(PROGRAMS
  scripts/prepare_rtdetr_models.sh
  scripts/convert_rtdetr_fp16.py
  scripts/compare_detections.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_auto_package(INSTALL_TO_SHARE launch)
