# Copyright 2026 Boshen Chen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Phase 2B YOLOv8 staged-control benchmark.

This lane deliberately uses the same Managed HIP encoder, strict ORT contract,
and standard decoder as the direct lane.  The only transport additions are one
Managed->standard and one standard->Managed adapter on each side of ORT.
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))
import yolov8_common as common  # noqa: E402
import isaac_ros_yolov8_phase2b_amd_managed_graph as managed_graph  # noqa: E402

from launch_ros.actions import ComposableNodeContainer  # noqa: E402
from launch_ros.descriptions import ComposableNode  # noqa: E402

from ros2_benchmark import ROS2BenchmarkConfig  # noqa: E402


RESULTS_DIR = os.environ.get("OVG_RESULTS_ROOT", "/workspaces/ovg-results")
RESULTS_FILE = os.environ.get("R2B_RESULT_FILE") or (
    "isaac_ros_yolov8_phase2b_amd_staged_control.json")


def launch_setup(container_prefix, container_sigterm_timeout):
    """Build the YOLOv8 staged-control graph."""
    namespace = TestIsaacROSYoloV8Phase2bAmdStagedControl.generate_namespace()
    model_path = managed_graph.model_path_for_test(
        TestIsaacROSYoloV8Phase2bAmdStagedControl)

    encoder = ComposableNode(
        name="Yolov8ManagedHipImageEncoder",
        namespace=namespace,
        package="isaac_ros_yolov8_std",
        plugin=(
            "nvidia::isaac_ros::yolov8_std::"
            "YoloV8ManagedHipImageEncoderNode"),
        parameters=[{
            "tensor_name": common.ORT_INPUT_TENSOR_NAME,
            "output_width": common.NETWORK_RESOLUTION["width"],
            "output_height": common.NETWORK_RESOLUTION["height"],
            "gpu_device_id": 0,
            "managed_pool_capacity": 16,
            "managed_pool_wait_timeout_ms": 100,
        }],
    )
    managed_to_std_input = ComposableNode(
        name="ManagedHipToStdInput",
        namespace=namespace,
        package="isaac_ros_onnx_inference",
        plugin=(
            "nvidia::isaac_ros::onnx_inference::"
            "ManagedHipToStdTensorListNode"),
        parameters=[{"gpu_device_id": 0}],
        remappings=[
            ("tensor_input", "managed_tensor_output"),
            ("tensor_output", "staged_std_input"),
        ],
    )
    std_to_managed_input = ComposableNode(
        name="StdToManagedHipInput",
        namespace=namespace,
        package="isaac_ros_onnx_inference",
        plugin=(
            "nvidia::isaac_ros::onnx_inference::"
            "StdToManagedHipTensorListNode"),
        parameters=[{
            "gpu_device_id": 0,
            "managed_pool_capacity": 16,
            "managed_pool_wait_timeout_ms": 100,
        }],
        remappings=[
            ("tensor_input", "staged_std_input"),
            ("tensor_output", "staged_managed_input"),
        ],
    )
    onnx = ComposableNode(
        name="OnnxInference",
        namespace=namespace,
        package="isaac_ros_onnx_inference",
        plugin="nvidia::isaac_ros::onnx_inference::OnnxInferenceNode",
        parameters=[{
            "model_file_path": model_path,
            "execution_provider": "migraphx",
            "gpu_device_id": 0,
            "transport": "managed",
            "managed_io_contract": "hip_managed_strict",
            "managed_input_contracts": ["images=float32[1,3,640,640]"],
            "managed_output_contracts": ["output0=float32[1,84,8400]"],
            "managed_pool_capacity": 16,
            "managed_pool_wait_timeout_ms": 100,
        }],
        remappings=[
            ("tensor_input", "staged_managed_input"),
            ("tensor_output", "staged_managed_output"),
        ],
    )
    managed_to_std_output = ComposableNode(
        name="ManagedHipToStdOutput",
        namespace=namespace,
        package="isaac_ros_onnx_inference",
        plugin=(
            "nvidia::isaac_ros::onnx_inference::"
            "ManagedHipToStdTensorListNode"),
        parameters=[{"gpu_device_id": 0}],
        remappings=[
            ("tensor_input", "staged_managed_output"),
            ("tensor_output", "staged_std_output"),
        ],
    )
    decoder = ComposableNode(
        name="Yolov8StdDecoder",
        namespace=namespace,
        package="isaac_ros_yolov8_std",
        plugin="nvidia::isaac_ros::yolov8_std::YoloV8DecoderNode",
        parameters=[{
            "tensor_name": common.ORT_OUTPUT_TENSOR_NAME,
            "confidence_threshold": 0.25,
            "nms_threshold": 0.45,
            "num_classes": 80,
        }],
        remappings=[("tensor_sub", "staged_std_output")],
    )

    container = ComposableNodeContainer(
        name="container",
        namespace=namespace,
        package="rclcpp_components",
        executable="component_container_mt",
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            common.make_data_loader_node(namespace),
            managed_graph.make_std_playback_node(namespace),
            encoder,
            managed_to_std_input,
            std_to_managed_input,
            onnx,
            managed_to_std_output,
            decoder,
            common.make_monitor_node(namespace),
        ],
        output="screen",
    )
    return [container]


def generate_test_description():
    """Generate the staged-control launch-test description."""
    return TestIsaacROSYoloV8Phase2bAmdStagedControl.generate_test_description_with_nsys(
        launch_setup)


class TestIsaacROSYoloV8Phase2bAmdStagedControl(
    managed_graph.TestIsaacROSYoloV8Phase2bAmdManaged):
    """Benchmark the intentional Managed<->standard control lane."""

    config = ROS2BenchmarkConfig(
        benchmark_name=(
            "Isaac ROS YOLOv8 Phase 2B AMD "
            "(ORT MIGraphX + staged control)"
        ),
        input_data_path=common.ROSBAG_PATH,
        publisher_upper_frequency=1000.0,
        publisher_lower_frequency=1.0,
        additional_fixed_publisher_rate_tests=[10.0, 30.0, 60.0],
        playback_message_buffer_size=1,
        pre_trial_run_wait_time_sec=5.0,
        log_folder=RESULTS_DIR,
        log_file_name=RESULTS_FILE,
        custom_report_info={
            "data_resolution": common.IMAGE_RESOLUTION,
            "network_resolution": common.NETWORK_RESOLUTION,
            "model": common.MODEL_FILE_NAME,
            "inference_backend": "ONNX Runtime MIGraphX EP",
            "transport": "Managed HIP TensorList with std control lane",
            "staging": "Managed->std->Managed before and after ORT",
            "managed_io_contract": "hip_managed_strict",
            "build_type": "Release",
            "result_directory": RESULTS_DIR,
        },
    )
