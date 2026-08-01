cmake_minimum_required(VERSION 3.22.1)
project(isaac_ros_detection_validation)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/compare_detection2d_bags.py
  scripts/run_nvidia_fixed_input_capture.sh
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_capture_runner test/test_capture_runner.py)
  ament_add_pytest_test(test_compare_detection2d_bags test/test_compare_detection2d_bags.py)
endif()

ament_package()
