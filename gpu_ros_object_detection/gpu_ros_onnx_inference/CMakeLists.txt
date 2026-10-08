cmake_minimum_required(VERSION 3.22.1)
project(gpu_ros_onnx_inference LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
find_package(gpu_ros_managed_core REQUIRED)
find_package(gpu_ros_managed_ros REQUIRED)
find_package(gpu_ros_managed_tensor_bundle REQUIRED)
find_package(rosidl_buffer REQUIRED)

include(cmake/onnxruntime.cmake)

# Provider availability is explicit. Requesting a provider that was not enabled
# at build time throws at runtime instead of silently falling back to CPU.
option(ORT_ENABLE_CUDA "Enable CUDAExecutionProvider selection" OFF)
option(ORT_ENABLE_ROCM "Enable ROCMExecutionProvider selection" OFF)
option(ORT_ENABLE_MIGRAPHX "Enable MIGraphXExecutionProvider selection" OFF)
option(
  BUILD_MANAGED_HIP_TRANSPORT
  "Build explicit standard-ROS2 to Managed HIP TensorBundle adapters"
  ${ORT_ENABLE_MIGRAPHX})
option(
  BUILD_MIGRAPHX_POL_TEST
  "Build the direct strict Managed HIP proof-of-life test"
  OFF)

# Native CUDA Buffer transport is enabled only in the NVIDIA manifest profile.
option(BUILD_NATIVE_TENSOR_LIST_TRANSPORT "Build native CUDA TensorList transport" OFF)
if(BUILD_NATIVE_TENSOR_LIST_TRANSPORT)
  if(NOT "$ENV{GPU_ROS_NVIDIA_PROFILE}" STREQUAL "1")
    message(FATAL_ERROR "Native TensorList requires GPU_ROS_NVIDIA_PROFILE=1 before colcon discovery")
  endif()
  if(NOT ORT_ENABLE_CUDA)
    message(FATAL_ERROR "Native TensorList requires ORT_ENABLE_CUDA=ON")
  endif()
  find_package(gpu_ros_nvidia_tensor_bundle_compat REQUIRED)
  find_package(CUDAToolkit REQUIRED)
endif()

# Pure ORT policy is shared without importing the package's Managed dependencies.
add_library(onnx_session SHARED
  src/onnx_session.cpp
  src/tensor_contract.cpp
  src/binding_report.cpp
  src/onnx_binding.cpp)
add_library(gpu_ros::onnx_session ALIAS onnx_session)
target_include_directories(onnx_session PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_link_libraries(onnx_session PUBLIC onnxruntime::onnxruntime)
target_compile_definitions(onnx_session PRIVATE
  $<$<BOOL:${ORT_ENABLE_CUDA}>:ORT_CUDA_AVAILABLE>
  $<$<BOOL:${ORT_ENABLE_ROCM}>:ORT_ROCM_AVAILABLE>
  $<$<BOOL:${ORT_ENABLE_MIGRAPHX}>:ORT_MIGRAPHX_AVAILABLE>)

add_library(onnx_model_path OBJECT src/model_path.cpp)
set_target_properties(onnx_model_path PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ── Inference core library ────────────────────────────────────────────────────
ament_auto_add_library(onnx_inference_core SHARED src/onnx_inference_core.cpp)
target_link_libraries(onnx_inference_core onnx_session)
target_compile_definitions(onnx_inference_core PRIVATE
  $<$<BOOL:${ORT_ENABLE_CUDA}>:ORT_CUDA_AVAILABLE>
  $<$<BOOL:${ORT_ENABLE_ROCM}>:ORT_ROCM_AVAILABLE>
  $<$<BOOL:${ORT_ENABLE_MIGRAPHX}>:ORT_MIGRAPHX_AVAILABLE>
)
if(ORT_ENABLE_CUDA OR BUILD_NATIVE_TENSOR_LIST_TRANSPORT)
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
if(ORT_ENABLE_ROCM OR ORT_ENABLE_MIGRAPHX OR BUILD_MANAGED_HIP_TRANSPORT)
  find_package(gpu_ros_managed_hip REQUIRED)
  target_link_libraries(onnx_inference_core gpu_ros_managed::hip)
  target_compile_definitions(onnx_inference_core PRIVATE GPU_ROS_MANAGED_HIP)
endif()

# ── ROS2 node + transport IO ──────────────────────────────────────────────────
set(node_sources
  src/onnx_inference_node.cpp
  src/std_tensor_bundle_io.cpp
  src/managed_tensor_bundle_io.cpp
  src/tensor_bundle_io_factory.cpp
)
if(BUILD_MANAGED_HIP_TRANSPORT)
  list(APPEND node_sources src/managed_hip_tensor_bundle_nodes.cpp)
endif()
if(BUILD_NATIVE_TENSOR_LIST_TRANSPORT)
  list(APPEND node_sources
    src/nitros_managed_tensor_bundle_adapter.cpp
    src/nitros_managed_tensor_bundle_nodes.cpp
    src/tensor_bundle_bridge_node.cpp)
endif()

ament_auto_add_library(onnx_inference_node SHARED ${node_sources})
target_sources(onnx_inference_node PRIVATE $<TARGET_OBJECTS:onnx_model_path>)
target_link_libraries(onnx_inference_node onnx_inference_core)
rclcpp_components_register_nodes(onnx_inference_node
  "gpu_ros::onnx_inference::OnnxInferenceNode")
if(BUILD_MANAGED_HIP_TRANSPORT)
  rclcpp_components_register_nodes(onnx_inference_node
    "gpu_ros::onnx_inference::StdToManagedHipTensorBundleNode"
    "gpu_ros::onnx_inference::ManagedHipToStdTensorBundleNode")
endif()

install(PROGRAMS
  scripts/copy_audit_common.py
  scripts/compare_nsys_cuda_traces.py
  scripts/compare_nvidia_copy_traces.py
  scripts/analyze_rocprof_traces.py
  scripts/rocprofv3_json_loader.py
  scripts/summarize_ort_profile.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_NATIVE_TENSOR_LIST_TRANSPORT)
  target_compile_definitions(onnx_inference_node PRIVATE BUILD_NATIVE_TENSOR_LIST_TRANSPORT)
  ament_target_dependencies(onnx_inference_node gpu_ros_nvidia_tensor_bundle_compat)
  target_link_libraries(onnx_inference_node CUDA::cudart gpu_ros_managed::cuda)
  rclcpp_components_register_nodes(onnx_inference_node
    "gpu_ros::onnx_inference::TensorBundleBridgeNode"
    "gpu_ros::onnx_inference::NitrosToManagedTensorBundleNode"
    "gpu_ros::onnx_inference::ManagedToNitrosTensorBundleNode")

  add_library(onnx_native_executor STATIC src/native_onnx_executor.cpp)
  set_target_properties(onnx_native_executor PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(onnx_native_executor PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
  target_link_libraries(onnx_native_executor PUBLIC
    onnx_session gpu_ros::gpu_ros_nvidia_tensor_list_native CUDA::cudart)

  add_library(onnx_inference_native_node SHARED src/native_onnx_inference_node.cpp)
  target_sources(onnx_inference_native_node PRIVATE $<TARGET_OBJECTS:onnx_model_path>)
  target_include_directories(onnx_inference_native_node PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
  ament_target_dependencies(onnx_inference_native_node PUBLIC rclcpp rclcpp_components std_msgs)
  target_link_libraries(onnx_inference_native_node
    PUBLIC gpu_ros::gpu_ros_nvidia_tensor_list_native
    PRIVATE onnx_native_executor)
  rclcpp_components_register_nodes(onnx_inference_native_node
    "gpu_ros::onnx_inference::NativeOnnxInferenceNode")
  install(TARGETS onnx_inference_native_node EXPORT nativeInferenceTargets
    ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
  ament_export_targets(nativeInferenceTargets HAS_LIBRARY_TARGET NAMESPACE gpu_ros::)
endif()

# ── Tests ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_onnx_inference_core
    test/test_onnx_inference_core.cpp
    TIMEOUT 900
  )
  if(ORT_ENABLE_MIGRAPHX)
    # Compile the same implementation in the test executable, never the
    # production library, with the three package-private exception points.
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(cancellation_test_model
      "${CMAKE_CURRENT_BINARY_DIR}/test_models/managed_output_cancellation.onnx")
    add_custom_command(
      OUTPUT "${cancellation_test_model}"
      COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/test/generate_cancellation_model.py"
        "${cancellation_test_model}"
      DEPENDS test/generate_cancellation_model.py
      VERBATIM)
    add_custom_target(onnx_cancellation_test_model DEPENDS "${cancellation_test_model}")
    add_dependencies(test_onnx_inference_core onnx_cancellation_test_model)
    target_sources(test_onnx_inference_core PRIVATE src/onnx_inference_core.cpp)
    target_include_directories(test_onnx_inference_core PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/test")
    ament_target_dependencies(test_onnx_inference_core gpu_ros_managed_tensor_bundle)
    target_link_libraries(test_onnx_inference_core
      onnx_session gpu_ros_managed::hip)
    target_compile_definitions(test_onnx_inference_core PRIVATE
      GPU_ROS_MANAGED_HIP GPU_ROS_ORT_CANCELLATION_TEST
      TEST_HIP_CANCELLATION_MODEL_PATH="${cancellation_test_model}")
    if(ORT_ENABLE_CUDA)
      target_link_libraries(test_onnx_inference_core gpu_ros_managed::cuda)
      target_compile_definitions(test_onnx_inference_core PRIVATE GPU_ROS_MANAGED_CUDA)
    endif()
  else()
    target_link_libraries(test_onnx_inference_core onnx_inference_core)
  endif()
  ament_add_gtest(test_staging_pool_limits
    test/test_staging_pool_limits.cpp)
  target_include_directories(test_staging_pool_limits PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_definitions(test_onnx_inference_core PRIVATE
    $<$<BOOL:${ORT_ENABLE_CUDA}>:ORT_CUDA_AVAILABLE>
    $<$<BOOL:${ORT_ENABLE_ROCM}>:ORT_ROCM_AVAILABLE>
    $<$<BOOL:${ORT_ENABLE_MIGRAPHX}>:ORT_MIGRAPHX_AVAILABLE>
  )
  set(CUDA_IO_BINDING_TEST_MODEL
    "/workspaces/isaac_ros-dev/assets/models/yolov8/yolov8s.onnx"
    CACHE FILEPATH "Valid YOLOv8 ONNX model used by the CUDA I/O Binding test")
  if(ORT_ENABLE_CUDA AND BUILD_NATIVE_TENSOR_LIST_TRANSPORT)
    if(EXISTS "${CUDA_IO_BINDING_TEST_MODEL}")
      target_compile_definitions(test_onnx_inference_core PRIVATE
        TEST_CUDA_IO_BINDING_MODEL_PATH="${CUDA_IO_BINDING_TEST_MODEL}")
      target_link_libraries(test_onnx_inference_core CUDA::cudart)
    else()
      message(FATAL_ERROR
        "CUDA I/O Binding test requires its real model: "
        "${CUDA_IO_BINDING_TEST_MODEL}")
    endif()
  endif()

  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_onnxruntime_configure
    test/test_onnxruntime_configure.py
  )
  ament_add_pytest_test(test_summarize_ort_profile
    test/test_summarize_ort_profile.py
  )
  ament_add_pytest_test(test_compare_nsys_cuda_traces
    test/test_compare_nsys_cuda_traces.py
  )
  ament_add_pytest_test(test_compare_nvidia_copy_traces
    test/test_compare_nvidia_copy_traces.py
  )
  ament_add_pytest_test(test_analyze_rocprof_traces
    test/test_analyze_rocprof_traces.py
  )

  if(BUILD_NATIVE_TENSOR_LIST_TRANSPORT)
    find_package(launch_testing_ament_cmake REQUIRED)
    target_link_libraries(test_onnx_inference_core onnx_inference_node)
    target_compile_definitions(test_onnx_inference_core PRIVATE BUILD_NATIVE_TENSOR_LIST_TRANSPORT)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(dynamic_output_test_model "${CMAKE_CURRENT_BINARY_DIR}/test_models/dynamic_native_output.onnx")
    add_custom_command(
      OUTPUT "${dynamic_output_test_model}"
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test/generate_dynamic_output_model.py"
        "${dynamic_output_test_model}"
      DEPENDS test/generate_dynamic_output_model.py
      VERBATIM)
    add_custom_target(native_dynamic_output_test_model DEPENDS "${dynamic_output_test_model}")
    add_dependencies(test_onnx_inference_core native_dynamic_output_test_model)
    target_sources(test_onnx_inference_core PRIVATE test/test_native_output_binding.cpp)
    target_compile_definitions(test_onnx_inference_core PRIVATE
      TEST_DYNAMIC_OUTPUT_MODEL_PATH="${dynamic_output_test_model}")

    # Compile the real pure implementation with private failure injection.
    ament_add_gtest(test_native_onnx_executor
      test/test_native_onnx_executor.cpp
      src/native_onnx_executor.cpp
      src/native_onnx_inference_node.cpp
      TIMEOUT 900)
    if(TARGET test_native_onnx_executor)
      target_sources(test_native_onnx_executor PRIVATE $<TARGET_OBJECTS:onnx_model_path>)
      target_include_directories(test_native_onnx_executor PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/test")
      ament_target_dependencies(test_native_onnx_executor rclcpp rclcpp_components std_msgs)
      target_link_libraries(test_native_onnx_executor
        onnx_session gpu_ros::gpu_ros_nvidia_tensor_list_native CUDA::cudart)
      target_compile_definitions(test_native_onnx_executor PRIVATE
        GPU_ROS_NATIVE_EXECUTOR_TEST
        TEST_DYNAMIC_OUTPUT_MODEL_PATH="${dynamic_output_test_model}"
        TEST_CUDA_IO_BINDING_MODEL_PATH="${CUDA_IO_BINDING_TEST_MODEL}")
      add_dependencies(test_native_onnx_executor native_dynamic_output_test_model)
    endif()
    add_launch_test(test/gpu_ros_onnx_rtdetr_pol_test.py TIMEOUT "600")
    add_launch_test(test/gpu_ros_onnx_rtdetr_managed_pol_test.py TIMEOUT "600")
  endif()

  if(BUILD_MANAGED_HIP_TRANSPORT AND ORT_ENABLE_MIGRAPHX AND BUILD_MIGRAPHX_POL_TEST)
    find_package(launch_testing_ament_cmake REQUIRED)
    add_launch_test(test/gpu_ros_onnx_managed_hip_pol_test.py TIMEOUT "900")
  endif()
endif()

install(TARGETS onnx_session EXPORT onnxSessionTargets
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
ament_export_targets(onnxSessionTargets HAS_LIBRARY_TARGET NAMESPACE gpu_ros::)
ament_auto_package(
  CONFIG_EXTRAS cmake/onnxruntime.cmake
  INSTALL_TO_SHARE launch USE_SCOPED_HEADER_INSTALL_DIR)
