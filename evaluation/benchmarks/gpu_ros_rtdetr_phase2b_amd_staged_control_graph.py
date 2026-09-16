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
"""Phase 2B RT-DETR staged-control benchmark."""

import os
import sys

sys.path.append(os.path.dirname(__file__))
import rtdetr_common as common  # noqa: E402
import gpu_ros_rtdetr_phase2b_amd_managed_graph as managed_graph  # noqa: E402

from launch_ros.actions import ComposableNodeContainer  # noqa: E402
from launch_ros.descriptions import ComposableNode  # noqa: E402

from ros2_benchmark import ROS2BenchmarkConfig  # noqa: E402


RESULTS_DIR = os.environ.get("OVG_RESULTS_ROOT", "/workspaces/ovg-results")
RESULTS_FILE = os.environ.get("R2B_RESULT_FILE") or (
    "gpu_ros_rtdetr_phase2b_amd_staged_control.json"
)


def launch_setup(container_prefix, container_sigterm_timeout):
    """Build the RT-DETR staged-control graph."""
    namespace = TestGpuRosRtDetrPhase2bAmdStagedControl.generate_namespace()
    model_path = os.path.join(
        TestGpuRosRtDetrPhase2bAmdStagedControl.get_assets_root_path(),
        "models",
        common.AMD_MODEL_FILE_NAME,
    )

    encoder = ComposableNode(
        name="RtdetrManagedHipImageEncoder",
        namespace=namespace,
        package="gpu_ros_rtdetr",
        plugin=("gpu_ros::rtdetr::RtDetrManagedHipImageEncoderNode"),
        parameters=[
            {
                "tensor_name": "input_tensor",
                "output_width": common.NETWORK_RESOLUTION["width"],
                "output_height": common.NETWORK_RESOLUTION["height"],
                "gpu_device_id": 0,
                "managed_pool_capacity": 16,
                "managed_pool_wait_timeout_ms": 100,
            }
        ],
        remappings=[("managed_tensor_output", "managed_tensor_image")],
    )
    hip_preprocessor = ComposableNode(
        name="RtdetrManagedHipPreprocessor",
        namespace=namespace,
        package="gpu_ros_rtdetr",
        plugin=("gpu_ros::rtdetr::RtDetrManagedHipPreprocessorNode"),
        parameters=[
            {
                "image_width": common.NETWORK_RESOLUTION["width"],
                "image_height": common.NETWORK_RESOLUTION["height"],
                "model_input_width": common.NETWORK_RESOLUTION["width"],
                "model_input_height": common.NETWORK_RESOLUTION["height"],
                "use_max_dim_for_orig_size": True,
                "gpu_device_id": 0,
                "managed_pool_capacity": 16,
                "managed_pool_wait_timeout_ms": 100,
            }
        ],
        remappings=[
            ("managed_tensor_input", "managed_tensor_image"),
            ("managed_tensor_output", "managed_tensor_preprocessed"),
        ],
    )
    managed_to_std_input = ComposableNode(
        name="ManagedHipToStdInput",
        namespace=namespace,
        package="gpu_ros_onnx_inference",
        plugin=("gpu_ros::onnx_inference::ManagedHipToStdTensorBundleNode"),
        parameters=[{"gpu_device_id": 0}],
        remappings=[
            ("tensor_input", "managed_tensor_preprocessed"),
            ("tensor_output", "staged_std_model_input"),
        ],
    )
    std_to_managed_input = ComposableNode(
        name="StdToManagedHipInput",
        namespace=namespace,
        package="gpu_ros_onnx_inference",
        plugin=("gpu_ros::onnx_inference::StdToManagedHipTensorBundleNode"),
        parameters=[
            {
                "gpu_device_id": 0,
                "managed_pool_capacity": 16,
                "managed_pool_wait_timeout_ms": 100,
            }
        ],
        remappings=[
            ("tensor_input", "staged_std_model_input"),
            ("tensor_output", "staged_managed_input"),
        ],
    )
    onnx = ComposableNode(
        name="OnnxInference",
        namespace=namespace,
        package="gpu_ros_onnx_inference",
        plugin="gpu_ros::onnx_inference::OnnxInferenceNode",
        parameters=[
            {
                "model_file_path": model_path,
                "execution_provider": "migraphx",
                "gpu_device_id": 0,
                "transport": "managed",
                "managed_io_contract": "hip_managed_strict",
                "managed_input_contracts": [
                    "images=float32[1,3,640,640]",
                    "orig_target_sizes=int64[1,2]",
                ],
                "managed_output_contracts": [
                    "labels=int64[1,300]",
                    "boxes=float32[1,300,4]",
                    "scores=float32[1,300]",
                ],
                "managed_pool_capacity": 16,
                "managed_pool_wait_timeout_ms": 100,
            }
        ],
        remappings=[
            ("tensor_input", "staged_managed_input"),
            ("tensor_output", "staged_managed_output"),
        ],
    )
    managed_to_std_output = ComposableNode(
        name="ManagedHipToStdOutput",
        namespace=namespace,
        package="gpu_ros_onnx_inference",
        plugin=("gpu_ros::onnx_inference::ManagedHipToStdTensorBundleNode"),
        parameters=[{"gpu_device_id": 0}],
        remappings=[
            ("tensor_input", "staged_managed_output"),
            ("tensor_output", "staged_std_output"),
        ],
    )
    decoder = ComposableNode(
        name="RtdetrStdDecoder",
        namespace=namespace,
        package="gpu_ros_rtdetr",
        plugin="gpu_ros::rtdetr::RtDetrDecoderNode",
        parameters=[{"confidence_threshold": 0.6}],
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
            hip_preprocessor,
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
    return TestGpuRosRtDetrPhase2bAmdStagedControl.generate_test_description_with_nsys(launch_setup)


class TestGpuRosRtDetrPhase2bAmdStagedControl(managed_graph.TestGpuRosRtDetrPhase2bAmdManaged):
    """Benchmark the intentional Managed<->standard RT-DETR control lane."""

    config = ROS2BenchmarkConfig(
        benchmark_name=("GPU ROS RT-DETR Phase 2B AMD (ORT MIGraphX + staged control)"),
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
            "model": common.AMD_MODEL_FILE_NAME,
            "inference_backend": "ONNX Runtime MIGraphX EP",
            "transport": "Managed HIP TensorBundle with std control lane",
            "staging": (
                "Managed->std->Managed before ORT; Managed->std->standard decoder after ORT"
            ),
            "managed_io_contract": "hip_managed_strict",
            "build_type": "Release",
            "result_directory": RESULTS_DIR,
        },
    )
