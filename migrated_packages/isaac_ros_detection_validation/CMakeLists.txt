cmake_minimum_required(VERSION 3.22.1)
project(isaac_ros_detection_validation)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/compare_detection2d_bags.py
  scripts/run_amd_phase2a_fixed_input_capture.sh
  scripts/run_rocprof_hip_copy_probe.sh
  scripts/run_amd_transport_audit.sh
  scripts/run_amd_phase2b_benchmark_matrix.sh
  scripts/count_ros_messages.py
  scripts/run_amd_phase2b_managed_high_load.sh
  scripts/run_nvidia_fixed_input_capture.sh
  scripts/run_nvidia_c_vs_d_copy_audit.sh
  scripts/run_nvidia_transport_audit.sh
  scripts/run_nvidia_yolov8_transport_audit.sh
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_capture_runner test/test_capture_runner.py)
  ament_add_pytest_test(test_compare_detection2d_bags test/test_compare_detection2d_bags.py)
  ament_add_pytest_test(test_managed_topology test/test_managed_topology.py)
endif()

ament_package()
