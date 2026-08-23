# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modified for the direct AMD Managed HIP and MIGraphX path in 2026.
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
Phase 2B AMD benchmark: YOLOv8 with Managed HIP and MIGraphX.

The graph uses the direct same-process Managed HIP TensorBundle path end to end.
"""

import os
import sys
import time

sys.path.append(os.path.dirname(__file__))
import yolov8_common as common  # noqa: E402

from launch_ros.actions import ComposableNodeContainer  # noqa: E402
from launch_ros.descriptions import ComposableNode  # noqa: E402

import rclpy  # noqa: E402
from ros2_benchmark import BenchmarkMode, ROS2BenchmarkConfig, ROS2BenchmarkTest  # noqa: E402
from ros2_benchmark_interfaces.srv import PlayMessages  # noqa: E402
from vision_msgs.msg import Detection2DArray  # noqa: E402


RESULTS_DIR = os.environ.get("OVG_RESULTS_ROOT", "/workspaces/ovg-results")
RESULTS_FILE = os.environ.get("R2B_RESULT_FILE") or (
    "gpu_ros_yolov8_phase2b_amd_managed.json")
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


def model_path_for_test(test_class):
    """Return the required user-provided YOLOv8 model path."""
    model_path = os.path.join(
        test_class.get_assets_root_path(), "models", common.MODEL_FILE_NAME)
    if not os.path.isfile(model_path) or os.path.getsize(model_path) == 0:
        resolved_path = os.path.abspath(model_path)
        raise RuntimeError(
            "YOLOv8 ONNX asset is missing:\n"
            f"{resolved_path}\n"
            "Provide OVG_YOLOV8_ONNX_SOURCE and run phase2 assets import-yolov8."
        )
    return model_path


def launch_setup(container_prefix, container_sigterm_timeout):
    """Build the AMD YOLOv8 Managed HIP benchmark graph."""
    namespace = TestGpuRosYoloV8Phase2bAmdManaged.generate_namespace()
    model_path = model_path_for_test(TestGpuRosYoloV8Phase2bAmdManaged)

    image_encoder = ComposableNode(
        name="Yolov8ImageEncoder",
        namespace=namespace,
        package="gpu_ros_yolov8",
        plugin=(
            "gpu_ros::yolov8::"
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
    onnx = ComposableNode(
        name="OnnxInference",
        namespace=namespace,
        package="gpu_ros_onnx_inference",
        plugin="gpu_ros::onnx_inference::OnnxInferenceNode",
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
            ("tensor_input", "managed_tensor_output"),
            ("tensor_output", "managed_tensor_output_ort"),
        ],
    )
    decoder = ComposableNode(
        name="Yolov8Decoder",
        namespace=namespace,
        package="gpu_ros_yolov8",
        plugin=(
            "gpu_ros::yolov8::"
            "YoloV8ManagedHipDecoderNode"),
        parameters=[{
            "gpu_device_id": 0,
            "tensor_name": common.ORT_OUTPUT_TENSOR_NAME,
            "confidence_threshold": 0.25,
            "nms_threshold": 0.45,
            "num_classes": 80,
        }],
        remappings=[("managed_tensor_input", "managed_tensor_output_ort")],
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
            onnx,
            decoder,
            common.make_monitor_node(namespace),
        ],
        output="screen",
    )
    return [container]


def generate_test_description():
    """Generate the ROS 2 benchmark launch-test description."""
    return TestGpuRosYoloV8Phase2bAmdManaged.generate_test_description_with_nsys(
        launch_setup)


class TestGpuRosYoloV8Phase2bAmdManaged(ROS2BenchmarkTest):
    """Benchmark YOLOv8 with AMD Managed HIP transport."""

    config = ROS2BenchmarkConfig(
        benchmark_name=(
            "GPU ROS YOLOv8 Phase 2B AMD "
            "(ORT MIGraphX + Managed HIP)"
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
            "model_sha256": (
                "d6e22418dd1acc69a232a1b297c01dfc785842fd11a4a84546c84e14cdeb235c"
            ),
            "inference_backend": "ONNX Runtime MIGraphX EP",
            "transport": "Managed HIP TensorBundle",
            "staging": "none; direct Managed HIP TensorBundle",
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
