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
# The Isaac ROS base image ships ORT at /opt/onnxruntime with a pkg-config file.
# We use pkg-config so no manual path hints are needed.
find_package(PkgConfig REQUIRED)
pkg_check_modules(ONNXRUNTIME REQUIRED libonnxruntime)

add_library(onnxruntime::onnxruntime INTERFACE IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIRS}"
  INTERFACE_LINK_LIBRARIES "${ONNXRUNTIME_LINK_LIBRARIES}"
  INTERFACE_LINK_OPTIONS "${ONNXRUNTIME_LDFLAGS_OTHER}"
)

# ── NITROS transport (optional, NVIDIA-only) ──────────────────────────────────
option(BUILD_NITROS_TRANSPORT "Build NITROS pub/sub transport path (requires NVIDIA Isaac ROS)" ON)

# ── Inference core library ────────────────────────────────────────────────────
ament_auto_add_library(onnx_inference_core SHARED
  src/onnx_inference_core.cpp
)
target_link_libraries(onnx_inference_core onnxruntime::onnxruntime)

# ── ROS2 node (std transport) ─────────────────────────────────────────────────
ament_auto_add_library(onnx_inference_node SHARED
  src/onnx_inference_node.cpp
)
target_link_libraries(onnx_inference_node onnx_inference_core)
rclcpp_components_register_nodes(onnx_inference_node
  "nvidia::isaac_ros::onnx_inference::OnnxInferenceNode")

# ── Tests ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_onnx_inference_core
    test/test_onnx_inference_core.cpp
  )
  target_link_libraries(test_onnx_inference_core onnx_inference_core)
endif()

ament_auto_package(INSTALL_TO_SHARE launch)
