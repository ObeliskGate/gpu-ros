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
# Search order:
#   1. cmake package (installed via apt libonnxruntime-dev or prebuilt tarball)
#   2. manual hint via -DONNXRUNTIME_ROOT=/opt/onnxruntime
find_package(onnxruntime QUIET)
if(NOT onnxruntime_FOUND)
  # Fallback: locate headers + library manually
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS
      ${ONNXRUNTIME_ROOT}/include/onnxruntime
      /opt/onnxruntime/include/onnxruntime
      /usr/include/onnxruntime
  )
  find_library(ONNXRUNTIME_LIB onnxruntime
    HINTS
      ${ONNXRUNTIME_ROOT}/lib
      /opt/onnxruntime/lib
      /usr/lib
  )
  if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
    message(STATUS "Found ONNX Runtime (manual): ${ONNXRUNTIME_LIB}")
    add_library(onnxruntime::onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
      IMPORTED_LOCATION "${ONNXRUNTIME_LIB}"
      INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
    )
  else()
    message(FATAL_ERROR
      "ONNX Runtime not found. Install libonnxruntime-dev or set -DONNXRUNTIME_ROOT=<path>")
  endif()
endif()

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
