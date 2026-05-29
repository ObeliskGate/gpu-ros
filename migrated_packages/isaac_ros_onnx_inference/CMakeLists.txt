cmake_minimum_required(VERSION 3.22.1)
project(isaac_ros_onnx_inference LANGUAGES CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# ── ONNX Runtime ──────────────────────────────────────────────────────────────
# Use the CUDA-13-matched ORT 1.23.1 that ships with Triton in the base image.
# Headers are unpacked to /opt/onnxruntime/include (CUDA-version-independent);
# the shared library is Triton's, so the CUDA EP links the right cuBLAS/cuDNN.
add_library(onnxruntime::onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
  IMPORTED_LOCATION "/opt/tritonserver/backends/onnxruntime/libonnxruntime.so"
  INTERFACE_INCLUDE_DIRECTORIES "/opt/onnxruntime/include"
)

# ── NITROS transport (optional, NVIDIA-only) ──────────────────────────────────
option(BUILD_NITROS_TRANSPORT "Build NITROS pub/sub transport path (requires NVIDIA Isaac ROS)" ON)

# ── Inference core library ────────────────────────────────────────────────────
ament_auto_add_library(onnx_inference_core SHARED
  src/onnx_inference_core.cpp
)
target_link_libraries(onnx_inference_core onnxruntime::onnxruntime)

# ── ROS2 node + transport IO ──────────────────────────────────────────────────
set(node_sources
  src/onnx_inference_node.cpp
  src/std_tensor_list_io.cpp
  src/tensor_list_io_factory.cpp
)
if(BUILD_NITROS_TRANSPORT)
  list(APPEND node_sources
    src/nitros_tensor_list_io.cpp
    src/tensor_list_bridge_node.cpp)
endif()

ament_auto_add_library(onnx_inference_node SHARED ${node_sources})
target_link_libraries(onnx_inference_node onnx_inference_core)
rclcpp_components_register_nodes(onnx_inference_node
  "nvidia::isaac_ros::onnx_inference::OnnxInferenceNode")

if(BUILD_NITROS_TRANSPORT)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(onnx_inference_node PRIVATE BUILD_NITROS_TRANSPORT)
  target_link_libraries(onnx_inference_node CUDA::cudart)
  rclcpp_components_register_nodes(onnx_inference_node
    "nvidia::isaac_ros::onnx_inference::TensorListBridgeNode")
endif()

# ── Tests ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_onnx_inference_core
    test/test_onnx_inference_core.cpp
  )
  target_link_libraries(test_onnx_inference_core onnx_inference_core)

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_onnx_rtdetr_pol_test.py TIMEOUT "600")
endif()

ament_auto_package(INSTALL_TO_SHARE launch USE_SCOPED_HEADER_INSTALL_DIR)
