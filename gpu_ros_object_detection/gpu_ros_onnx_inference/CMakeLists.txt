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
find_package(gpu_ros_rosidl_buffer REQUIRED)

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
add_library(onnx_inference_core SHARED src/onnx_inference_core.cpp)
target_link_libraries(onnx_inference_core PUBLIC onnx_session)
ament_target_dependencies(onnx_inference_core PUBLIC std_msgs)
target_include_directories(onnx_inference_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
add_library(onnx_rosidl_buffer SHARED src/rosidl_buffer_adapter.cpp)
target_include_directories(onnx_rosidl_buffer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_link_libraries(onnx_rosidl_buffer PUBLIC onnx_inference_core gpu_ros::rosidl_buffer)
ament_target_dependencies(onnx_rosidl_buffer PUBLIC gpu_ros_tensor_bundle_msgs)
if(ORT_ENABLE_CUDA OR BUILD_NATIVE_TENSOR_LIST_TRANSPORT)
  find_package(gpu_ros_managed_cuda REQUIRED)
  if(NOT TARGET gpu_ros_managed::cuda)
    message(FATAL_ERROR
      "CUDA transport requires the gpu_ros_managed::cuda target and CUDA SDK.")
  endif()
endif()
if(ORT_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  find_package(cuda_buffer REQUIRED)
  find_package(rcutils REQUIRED)
  if(NOT TARGET gpu_ros::rosidl_buffer_cuda)
    message(FATAL_ERROR "ORT CUDA Buffer adapter requires GPU_ROS_ROSIDL_BUFFER_BUILD_CUDA=ON")
  endif()
  target_link_libraries(onnx_rosidl_buffer PRIVATE gpu_ros::rosidl_buffer_cuda CUDA::cudart)
  target_link_libraries(onnx_rosidl_buffer PRIVATE cuda_buffer::cuda_buffer rcutils::rcutils)
  target_compile_definitions(onnx_rosidl_buffer PRIVATE GPU_ROS_ROSIDL_CUDA)
endif()
if(ORT_ENABLE_ROCM OR ORT_ENABLE_MIGRAPHX OR BUILD_MANAGED_HIP_TRANSPORT)
  find_package(gpu_ros_managed_hip REQUIRED)
endif()

# ── ROS2 node + transport IO ──────────────────────────────────────────────────
set(node_sources
  src/onnx_inference_node.cpp
  src/std_tensor_bundle_io.cpp
  src/managed_tensor_bundle_io.cpp
  src/tensor_bundle_io_factory.cpp
  src/rosidl_buffer_tensor_bundle_io.cpp
)
if(BUILD_MANAGED_HIP_TRANSPORT)
  list(APPEND node_sources src/managed_hip_tensor_bundle_nodes.cpp)
endif()
if(BUILD_NATIVE_TENSOR_LIST_TRANSPORT)
  list(APPEND node_sources
    src/tensor_list_buffer_io.cpp
    src/nitros_managed_tensor_bundle_nodes.cpp
    src/tensor_bundle_bridge_node.cpp)
endif()

ament_auto_add_library(onnx_inference_node SHARED ${node_sources})
target_sources(onnx_inference_node PRIVATE $<TARGET_OBJECTS:onnx_model_path>)
target_link_libraries(onnx_inference_node onnx_inference_core onnx_rosidl_buffer)
if(ORT_ENABLE_CUDA)
  target_link_libraries(onnx_inference_node gpu_ros_managed::cuda gpu_ros::rosidl_buffer_cuda)
  target_compile_definitions(onnx_inference_node PRIVATE GPU_ROS_MANAGED_CUDA GPU_ROS_ROSIDL_CUDA)
endif()
if(ORT_ENABLE_ROCM OR ORT_ENABLE_MIGRAPHX OR BUILD_MANAGED_HIP_TRANSPORT)
  target_link_libraries(onnx_inference_node gpu_ros_managed::hip)
  target_compile_definitions(onnx_inference_node PRIVATE GPU_ROS_MANAGED_HIP)
endif()
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
  target_link_libraries(onnx_inference_node CUDA::cudart gpu_ros_managed::cuda
    gpu_ros::gpu_ros_nvidia_tensor_list_native gpu_ros::rosidl_buffer_cuda)
  rclcpp_components_register_nodes(onnx_inference_node
    "gpu_ros::onnx_inference::TensorBundleBridgeNode"
    "gpu_ros::onnx_inference::NitrosToManagedTensorBundleNode"
    "gpu_ros::onnx_inference::ManagedToNitrosTensorBundleNode")

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
  target_link_libraries(test_onnx_inference_core onnx_inference_core onnx_inference_node)
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

  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(dynamic_output_test_model "${CMAKE_CURRENT_BINARY_DIR}/test_models/dynamic_output.onnx")
  add_custom_command(
    OUTPUT "${dynamic_output_test_model}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test/generate_dynamic_output_model.py"
      "${dynamic_output_test_model}"
    DEPENDS test/generate_dynamic_output_model.py
    VERBATIM)
  add_custom_target(onnx_dynamic_output_test_model DEPENDS "${dynamic_output_test_model}")
  ament_add_gtest(test_onnx_rosidl_buffer
    test/test_onnx_rosidl_buffer.cpp src/onnx_inference_core.cpp src/onnx_inference_node.cpp
    TIMEOUT 900)
  target_sources(test_onnx_rosidl_buffer PRIVATE $<TARGET_OBJECTS:onnx_model_path>)
  target_include_directories(test_onnx_rosidl_buffer PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/test" "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(test_onnx_rosidl_buffer onnx_rosidl_buffer onnx_inference_node)
  target_compile_definitions(test_onnx_rosidl_buffer PRIVATE GPU_ROS_ORT_BINDING_TEST
    TEST_DYNAMIC_OUTPUT_MODEL_PATH="${dynamic_output_test_model}")
  add_dependencies(test_onnx_rosidl_buffer onnx_dynamic_output_test_model)
  ament_add_gtest(test_managed_output_binding
    test/test_managed_output_binding.cpp src/onnx_inference_core.cpp
    src/managed_tensor_bundle_io.cpp TIMEOUT 900)
  target_include_directories(test_managed_output_binding PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/test")
  target_link_libraries(test_managed_output_binding onnx_inference_node onnx_rosidl_buffer)
  target_compile_definitions(test_managed_output_binding PRIVATE GPU_ROS_ORT_BINDING_TEST GPU_ROS_ORT_MANAGED_TEST
    TEST_DYNAMIC_OUTPUT_MODEL_PATH="${dynamic_output_test_model}")
  add_dependencies(test_managed_output_binding onnx_dynamic_output_test_model)
  if(ORT_ENABLE_CUDA)
    foreach(test_target test_onnx_rosidl_buffer test_managed_output_binding)
      target_compile_definitions(${test_target} PRIVATE GPU_ROS_TEST_CUDA
        TEST_CUDA_IO_BINDING_MODEL_PATH="${CUDA_IO_BINDING_TEST_MODEL}")
      target_link_libraries(${test_target} CUDA::cudart gpu_ros_managed::cuda)
      ament_target_dependencies(${test_target} cuda_buffer)
    endforeach()
    target_compile_definitions(test_managed_output_binding PRIVATE GPU_ROS_MANAGED_CUDA)
  endif()
  if(ORT_ENABLE_MIGRAPHX)
    set(cancellation_test_model "${CMAKE_CURRENT_BINARY_DIR}/test_models/managed_output_cancellation.onnx")
    add_custom_command(OUTPUT "${cancellation_test_model}"
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test/generate_cancellation_model.py"
        "${cancellation_test_model}" DEPENDS test/generate_cancellation_model.py VERBATIM)
    add_custom_target(onnx_cancellation_test_model DEPENDS "${cancellation_test_model}")
    add_dependencies(test_managed_output_binding onnx_cancellation_test_model)
    target_compile_definitions(test_managed_output_binding PRIVATE GPU_ROS_TEST_HIP GPU_ROS_MANAGED_HIP
      TEST_HIP_CANCELLATION_MODEL_PATH="${cancellation_test_model}")
    target_link_libraries(test_managed_output_binding gpu_ros_managed::hip)
  endif()
  if(BUILD_NATIVE_TENSOR_LIST_TRANSPORT)
    find_package(launch_testing_ament_cmake REQUIRED)
    target_compile_definitions(test_onnx_rosidl_buffer PRIVATE GPU_ROS_TEST_TENSOR_LIST)
    ament_target_dependencies(test_onnx_rosidl_buffer isaac_ros_tensor_msgs)
    target_compile_definitions(test_managed_output_binding PRIVATE BUILD_NATIVE_TENSOR_LIST_TRANSPORT)
    ament_target_dependencies(test_managed_output_binding gpu_ros_nvidia_tensor_bundle_compat)
    add_launch_test(test/gpu_ros_onnx_rtdetr_pol_test.py TIMEOUT "600")
    add_launch_test(test/gpu_ros_onnx_rtdetr_managed_pol_test.py TIMEOUT "600")
  endif()

  if(BUILD_MANAGED_HIP_TRANSPORT AND ORT_ENABLE_MIGRAPHX AND BUILD_MIGRAPHX_POL_TEST)
    find_package(launch_testing_ament_cmake REQUIRED)
    add_launch_test(test/gpu_ros_onnx_managed_hip_pol_test.py TIMEOUT "900")
  endif()
endif()

install(TARGETS onnx_session onnx_inference_core onnx_rosidl_buffer EXPORT onnxSessionTargets
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
ament_export_targets(onnxSessionTargets HAS_LIBRARY_TARGET NAMESPACE gpu_ros::)
ament_export_dependencies(std_msgs gpu_ros_tensor_bundle_msgs gpu_ros_rosidl_buffer)
ament_auto_package(
  CONFIG_EXTRAS cmake/onnxruntime.cmake
  INSTALL_TO_SHARE launch USE_SCOPED_HEADER_INSTALL_DIR)
