cmake_minimum_required(VERSION 3.22.1)
project(isaac_ros_onnx_inference LANGUAGES CXX)

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
find_package(gpu_ros_managed_core REQUIRED)
find_package(gpu_ros_managed_ros REQUIRED)
find_package(gpu_ros_managed_tensor_list REQUIRED)

# ── ONNX Runtime ──────────────────────────────────────────────────────────────
set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime installation root")
set(ONNXRUNTIME_INCLUDE_DIR "" CACHE PATH "Directory containing onnxruntime_cxx_api.h")
set(ONNXRUNTIME_LIBRARY "" CACHE FILEPATH "Path to libonnxruntime.so")

if(NOT ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
  set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
endif()
if(NOT ONNXRUNTIME_INCLUDE_DIR AND DEFINED ENV{ONNXRUNTIME_INCLUDE_DIR})
  set(ONNXRUNTIME_INCLUDE_DIR "$ENV{ONNXRUNTIME_INCLUDE_DIR}")
endif()
if(NOT ONNXRUNTIME_LIBRARY AND DEFINED ENV{ONNXRUNTIME_LIBRARY})
  set(ONNXRUNTIME_LIBRARY "$ENV{ONNXRUNTIME_LIBRARY}")
endif()

if(ONNXRUNTIME_ROOT)
  find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS "${ONNXRUNTIME_ROOT}"
    PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session
    NO_DEFAULT_PATH)
  find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS "${ONNXRUNTIME_ROOT}"
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH)
endif()

# NVIDIA Isaac ROS compatibility fallback: the CUDA-matched ORT library often
# ships inside Triton's backend while headers are unpacked separately.
if(NOT ONNXRUNTIME_INCLUDE_DIR AND EXISTS "/opt/onnxruntime/include/onnxruntime_cxx_api.h")
  set(ONNXRUNTIME_INCLUDE_DIR "/opt/onnxruntime/include")
endif()
if(NOT ONNXRUNTIME_LIBRARY AND EXISTS "/opt/tritonserver/backends/onnxruntime/libonnxruntime.so")
  set(ONNXRUNTIME_LIBRARY "/opt/tritonserver/backends/onnxruntime/libonnxruntime.so")
endif()

if(NOT ONNXRUNTIME_INCLUDE_DIR)
  find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session)
endif()
if(NOT ONNXRUNTIME_LIBRARY)
  find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime)
endif()

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR
    "ONNX Runtime was not found. Set ONNXRUNTIME_ROOT, or set "
    "ONNXRUNTIME_INCLUDE_DIR and ONNXRUNTIME_LIBRARY explicitly.")
endif()
if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
  message(FATAL_ERROR
    "ONNXRUNTIME_INCLUDE_DIR does not contain onnxruntime_cxx_api.h: "
    "${ONNXRUNTIME_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${ONNXRUNTIME_LIBRARY}")
  message(FATAL_ERROR
    "ONNXRUNTIME_LIBRARY does not exist: ${ONNXRUNTIME_LIBRARY}")
endif()

add_library(onnxruntime::onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
)
message(STATUS "ONNX Runtime include dir: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")

# Provider availability is explicit. Requesting a provider that was not enabled
# at build time throws at runtime instead of silently falling back to CPU.
option(ORT_ENABLE_CUDA "Enable CUDAExecutionProvider selection" ON)
option(ORT_ENABLE_ROCM "Enable ROCMExecutionProvider selection" OFF)
option(ORT_ENABLE_MIGRAPHX "Enable MIGraphXExecutionProvider selection" OFF)

# ── NITROS transport (optional, NVIDIA-only) ──────────────────────────────────
option(BUILD_NITROS_TRANSPORT "Build NITROS pub/sub transport path (requires NVIDIA Isaac ROS)" ON)

# ── Inference core library ────────────────────────────────────────────────────
ament_auto_add_library(onnx_inference_core SHARED
  src/onnx_inference_core.cpp
)
target_link_libraries(onnx_inference_core onnxruntime::onnxruntime)
target_compile_definitions(onnx_inference_core PRIVATE
  $<$<BOOL:${ORT_ENABLE_CUDA}>:ORT_CUDA_AVAILABLE>
  $<$<BOOL:${ORT_ENABLE_ROCM}>:ORT_ROCM_AVAILABLE>
  $<$<BOOL:${ORT_ENABLE_MIGRAPHX}>:ORT_MIGRAPHX_AVAILABLE>
)
if(ORT_ENABLE_CUDA OR BUILD_NITROS_TRANSPORT)
  find_package(gpu_ros_managed_cuda REQUIRED)
  if(NOT TARGET gpu_ros_managed::cuda)
    message(FATAL_ERROR
      "CUDA transport requires the gpu_ros_managed::cuda target and CUDA SDK.")
  endif()
endif()
if(ORT_ENABLE_CUDA)
  target_link_libraries(onnx_inference_core gpu_ros_managed::cuda)
  target_compile_definitions(onnx_inference_core PRIVATE GPU_ROS_MANAGED_CUDA)
endif()

# ── ROS2 node + transport IO ──────────────────────────────────────────────────
set(node_sources
  src/onnx_inference_node.cpp
  src/std_tensor_list_io.cpp
  src/managed_tensor_list_io.cpp
  src/tensor_list_io_factory.cpp
)
if(BUILD_NITROS_TRANSPORT)
  list(APPEND node_sources
    src/nitros_tensor_list_io.cpp
    src/nitros_managed_tensor_list_adapter.cpp
    src/nitros_managed_tensor_list_nodes.cpp
    src/tensor_list_bridge_node.cpp)
endif()

ament_auto_add_library(onnx_inference_node SHARED ${node_sources})
target_link_libraries(onnx_inference_node onnx_inference_core)
rclcpp_components_register_nodes(onnx_inference_node
  "nvidia::isaac_ros::onnx_inference::OnnxInferenceNode")

install(PROGRAMS
  scripts/compare_nsys_cuda_traces.py
  scripts/summarize_ort_profile.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_NITROS_TRANSPORT)
  find_package(isaac_ros_nitros QUIET)
  find_package(isaac_ros_managed_nitros QUIET)
  find_package(isaac_ros_nitros_tensor_list_type QUIET)
  find_package(CUDAToolkit QUIET)
  if(NOT isaac_ros_nitros_FOUND OR
    NOT isaac_ros_managed_nitros_FOUND OR
    NOT isaac_ros_nitros_tensor_list_type_FOUND OR
    NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR
      "BUILD_NITROS_TRANSPORT=ON requires isaac_ros_nitros, "
      "isaac_ros_managed_nitros, isaac_ros_nitros_tensor_list_type, and CUDA. "
      "Set -DBUILD_NITROS_TRANSPORT=OFF for AMD/CPU standard ROS2 builds.")
  endif()
  target_compile_definitions(onnx_inference_node PRIVATE BUILD_NITROS_TRANSPORT)
  ament_target_dependencies(onnx_inference_node
    isaac_ros_nitros
    isaac_ros_managed_nitros
    isaac_ros_nitros_tensor_list_type)
  target_link_libraries(onnx_inference_node CUDA::cudart)
  target_link_libraries(onnx_inference_node gpu_ros_managed::cuda)
  rclcpp_components_register_nodes(onnx_inference_node
    "nvidia::isaac_ros::onnx_inference::TensorListBridgeNode"
    "nvidia::isaac_ros::onnx_inference::NitrosToManagedTensorListNode"
    "nvidia::isaac_ros::onnx_inference::ManagedToNitrosTensorListNode")
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
  target_compile_definitions(test_onnx_inference_core PRIVATE
    $<$<BOOL:${ORT_ENABLE_CUDA}>:ORT_CUDA_AVAILABLE>
    $<$<BOOL:${ORT_ENABLE_ROCM}>:ORT_ROCM_AVAILABLE>
    $<$<BOOL:${ORT_ENABLE_MIGRAPHX}>:ORT_MIGRAPHX_AVAILABLE>
  )
  set(CUDA_IO_BINDING_TEST_MODEL
    "/workspaces/isaac_ros-dev/assets/models/yolov8/yolov8s.onnx"
    CACHE FILEPATH "Valid YOLOv8 ONNX model used by the CUDA I/O Binding test")
  if(ORT_ENABLE_CUDA AND BUILD_NITROS_TRANSPORT)
    if(EXISTS "${CUDA_IO_BINDING_TEST_MODEL}")
      target_compile_definitions(test_onnx_inference_core PRIVATE
        TEST_CUDA_IO_BINDING_MODEL_PATH="${CUDA_IO_BINDING_TEST_MODEL}")
      target_link_libraries(test_onnx_inference_core CUDA::cudart)
    else()
      message(WARNING
        "CUDA I/O Binding test disabled because its model does not exist: "
        "${CUDA_IO_BINDING_TEST_MODEL}")
    endif()
  endif()

  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_summarize_ort_profile
    test/test_summarize_ort_profile.py
  )
  ament_add_pytest_test(test_compare_nsys_cuda_traces
    test/test_compare_nsys_cuda_traces.py
  )

  if(BUILD_NITROS_TRANSPORT)
    find_package(launch_testing_ament_cmake REQUIRED)
    ament_add_gtest(test_nitros_managed_tensor_list_adapter
      test/test_nitros_managed_tensor_list_adapter.cpp)
    target_link_libraries(test_nitros_managed_tensor_list_adapter
      onnx_inference_node
      gpu_ros_managed::cuda
      CUDA::cudart)
    ament_target_dependencies(test_nitros_managed_tensor_list_adapter
      isaac_ros_nitros
      isaac_ros_managed_nitros
      isaac_ros_nitros_tensor_list_type)
    add_launch_test(test/isaac_ros_onnx_rtdetr_pol_test.py TIMEOUT "600")
    add_launch_test(test/isaac_ros_onnx_rtdetr_managed_pol_test.py TIMEOUT "600")
  endif()
endif()

ament_auto_package(INSTALL_TO_SHARE launch USE_SCOPED_HEADER_INSTALL_DIR)
