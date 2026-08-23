cmake_minimum_required(VERSION 3.22.1)
project(gpu_ros_tensor_bundle_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Tensor.msg"
  "msg/TensorBundle.msg"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime std_msgs)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_tensor_bundle_roundtrip
    test/test_tensor_bundle_roundtrip.py)
endif()

ament_package()
