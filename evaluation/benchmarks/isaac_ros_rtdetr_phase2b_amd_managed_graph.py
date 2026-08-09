# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright 2026 Maintainer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""
Phase 2B AMD benchmark: RT-DETR with Managed HIP and MIGraphX.

The graph uses the standard ROS 2 image and TensorList boundary, explicit
host-to-device/device-to-host HIP staging, and Managed transport at the ORT
inference boundary.
"""

import os
import sys
import time

sys.path.append(os.path.dirname(__file__))
import rtdetr_common as common  # noqa: E402

from launch_ros.actions import ComposableNodeContainer  # noqa: E402
from launch_ros.descriptions import ComposableNode  # noqa: E402

import rclpy  # noqa: E402
from ros2_benchmark import BenchmarkMode, ROS2BenchmarkConfig, ROS2BenchmarkTest  # noqa: E402
from ros2_benchmark_interfaces.srv import PlayMessages  # noqa: E402
from vision_msgs.msg import Detection2DArray  # noqa: E402


RESULTS_DIR = os.environ.get("OVG_RESULTS_ROOT", "/workspaces/ovg-results")
RESULTS_FILE = os.environ.get("R2B_RESULT_FILE") or (
    "isaac_ros_rtdetr_phase2b_amd_managed.json")
MIGRAPHX_CACHE_PATH = os.environ.get(
    "ORT_MIGRAPHX_MODEL_CACHE_PATH",
    os.path.join(os.environ.get("OVG_CACHE_ROOT", "/workspaces/ovg-cache"), "migraphx"),
)
os.environ.setdefault("ORT_MIGRAPHX_MODEL_CACHE_PATH", MIGRAPHX_CACHE_PATH)
MIGRAPHX_WARMUP_TIMEOUT_SEC = float(
    os.environ.get("MIGRAPHX_WARMUP_TIMEOUT_SEC", "900"))


def make_std_playback_node(namespace):
    """Use generic ROS messages for the benchmark source topics."""
    return ComposableNode(
        name="PlaybackNode",
        namespace=namespace,
        package="ros2_benchmark",
        plugin="ros2_benchmark::PlaybackNode",
        parameters=[{
            "data_formats": ["sensor_msgs/msg/Image", "sensor_msgs/msg/CameraInfo"]
        }],
        remappings=[
            ("buffer/input0", "data_loader/image_raw"),
            ("input0", "image"),
            ("buffer/input1", "data_loader/camera_info"),
            ("input1", "camera_info"),
        ],
    )


def make_stager(namespace):
    """Create the standard TensorList to Managed HIP staging node."""
    return ComposableNode(
        name="StdToManagedHip",
        namespace=namespace,
        package="isaac_ros_onnx_inference",
        plugin=(
            "nvidia::isaac_ros::onnx_inference::"
            "StdToManagedHipTensorListNode"
        ),
        parameters=[{"gpu_device_id": 0}],
        remappings=[
            ("tensor_input", "tensor_pub"),
            ("tensor_output", "managed_tensor_input"),
        ],
    )


def make_unstager(namespace):
    """Create the Managed HIP to standard TensorList staging node."""
    return ComposableNode(
        name="ManagedHipToStd",
        namespace=namespace,
        package="isaac_ros_onnx_inference",
        plugin=(
            "nvidia::isaac_ros::onnx_inference::"
            "ManagedHipToStdTensorListNode"
        ),
        parameters=[{"gpu_device_id": 0}],
        remappings=[
            ("tensor_input", "managed_tensor_output"),
            ("tensor_output", "tensor_sub"),
        ],
    )


def launch_setup(container_prefix, container_sigterm_timeout):
    """Build the AMD RT-DETR Managed HIP benchmark graph."""
    namespace = TestIsaacROSRtDetrPhase2bAmdManaged.generate_namespace()

    image_encoder = ComposableNode(
        name="RtdetrImageEncoder",
        namespace=namespace,
        package="isaac_ros_rtdetr_std",
        plugin="nvidia::isaac_ros::rtdetr_std::RtDetrImageEncoderNode",
        parameters=[{
            "tensor_name": "input_tensor",
            "output_width": common.NETWORK_RESOLUTION["width"],
            "output_height": common.NETWORK_RESOLUTION["height"],
        }],
    )
    preprocessor = ComposableNode(
        name="RtdetrPreprocessor",
        namespace=namespace,
        package="isaac_ros_rtdetr_std",
        plugin="nvidia::isaac_ros::rtdetr_std::RtDetrPreprocessorNode",
        parameters=[{
            "image_width": common.NETWORK_RESOLUTION["width"],
            "image_height": common.NETWORK_RESOLUTION["height"],
            "use_max_dim_for_orig_size": True,
        }],
    )
    onnx = ComposableNode(
        name="OnnxInference",
        namespace=namespace,
        package="isaac_ros_onnx_inference",
        plugin="nvidia::isaac_ros::onnx_inference::OnnxInferenceNode",
        parameters=[{
            "model_file_path": os.path.join(
                TestIsaacROSRtDetrPhase2bAmdManaged.get_assets_root_path(),
                "models",
                common.MODEL_FILE_NAME,
            ),
            "execution_provider": "migraphx",
            "gpu_device_id": 0,
            "transport": "managed",
        }],
        remappings=[
            ("tensor_input", "managed_tensor_input"),
            ("tensor_output", "managed_tensor_output"),
        ],
    )
    decoder = ComposableNode(
        name="RtdetrDecoder",
        namespace=namespace,
        package="isaac_ros_rtdetr_std",
        plugin="nvidia::isaac_ros::rtdetr_std::RtDetrDecoderNode",
        parameters=[{"confidence_threshold": 0.6}],
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
            make_std_playback_node(namespace),
            image_encoder,
            preprocessor,
            make_stager(namespace),
            onnx,
            make_unstager(namespace),
            decoder,
            common.make_monitor_node(namespace),
        ],
        output="screen",
    )
    return [container]


def generate_test_description():
    """Generate the ROS 2 benchmark launch-test description."""
    return TestIsaacROSRtDetrPhase2bAmdManaged.generate_test_description_with_nsys(
        launch_setup)


class TestIsaacROSRtDetrPhase2bAmdManaged(ROS2BenchmarkTest):
    """Benchmark RT-DETR with AMD Managed HIP transport."""

    config = ROS2BenchmarkConfig(
        benchmark_name=(
            "Isaac ROS RT-DETR Phase 2B AMD "
            "(ORT MIGraphX + Managed HIP)"
        ),
        input_data_path=common.ROSBAG_PATH,
        publisher_upper_frequency=1000.0,
        publisher_lower_frequency=1.0,
        playback_message_buffer_size=1,
        pre_trial_run_wait_time_sec=5.0,
        log_folder=RESULTS_DIR,
        log_file_name=RESULTS_FILE,
        custom_report_info={
            "data_resolution": common.IMAGE_RESOLUTION,
            "network_resolution": common.NETWORK_RESOLUTION,
            "inference_backend": "ONNX Runtime MIGraphX EP",
            "transport": "Managed HIP TensorList",
            "staging": "explicit host-to-device/device-to-host",
            "build_type": "Release",
            "result_directory": RESULTS_DIR,
        },
    )

    def prepare_buffer(self):
        """Warm up lazy MIGraphX compilation before measured trials."""
        super().prepare_buffer()
        if getattr(self, "_migraphx_warmup_complete", False):
            return

        detection_received = False

        def on_detection(_message):
            nonlocal detection_received
            detection_received = True

        subscription = self.node.create_subscription(
            Detection2DArray, "detections_output", on_detection, 10)
        try:
            client = self.create_service_client_blocking(PlayMessages, "play_messages")
            request = PlayMessages.Request()
            request.playback_mode = BenchmarkMode.LOOPING.value
            request.target_publisher_rate = 1.0
            request.message_count = 1
            request.enforce_publisher_rate = False
            request.revise_timestamps_as_message_ids = False

            self.get_logger().info(
                "Starting one-frame MIGraphX Managed HIP warm-up; "
                "waiting for detections_output")
            future = client.call_async(request)
            deadline = time.monotonic() + MIGRAPHX_WARMUP_TIMEOUT_SEC
            while not detection_received and time.monotonic() < deadline:
                rclpy.spin_once(self.node, timeout_sec=0.5)
                if future.done() and future.exception() is not None:
                    raise RuntimeError(
                        "MIGraphX Managed HIP warm-up playback failed"
                    ) from future.exception()

            if not detection_received:
                raise RuntimeError(
                    "MIGraphX Managed HIP warm-up did not produce detections "
                    f"within {MIGRAPHX_WARMUP_TIMEOUT_SEC:.0f} seconds")
            self._migraphx_warmup_complete = True
            self.get_logger().info(
                "MIGraphX Managed HIP warm-up complete; "
                "starting measured benchmark")
        finally:
            self.node.destroy_subscription(subscription)

    def test_benchmark(self):
        """Run the configured benchmark sweep."""
        self.run_benchmark()

# Modified derived source; upstream NVIDIA Apache-2.0 attribution retained.
