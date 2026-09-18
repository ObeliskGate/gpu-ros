# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modified for the AMD standard-ROS and MIGraphX path in 2026.
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
"""Phase 2a AMD benchmark: RT-DETR std image encoder + ORT MIGraphX.

This benchmark covers only the AMD target path:
Image -> std RT-DETR image encoder -> std RtDetrPreprocessor ->
OnnxInferenceNode(transport=std, execution_provider=migraphx) ->
std RtDetrDecoder -> Detection2DArray.
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

RESULTS_DIR = os.environ.get('OVG_RESULTS_ROOT', '/workspaces/ovg-results')
RESULTS_FILE = os.environ.get('R2B_RESULT_FILE') or 'gpu_ros_rtdetr_phase2a_amd.json'
MIGRAPHX_WARMUP_TIMEOUT_SEC = float(os.environ.get('MIGRAPHX_WARMUP_TIMEOUT_SEC', '900'))


def make_std_playback_node(namespace):
    """Use generic ROS messages for the benchmark source topics."""
    return ComposableNode(
        name='PlaybackNode',
        namespace=namespace,
        package='ros2_benchmark',
        plugin='ros2_benchmark::PlaybackNode',
        parameters=[{'data_formats': ['sensor_msgs/msg/Image', 'sensor_msgs/msg/CameraInfo']}],
        remappings=[
            ('buffer/input0', 'data_loader/image_raw'),
            ('input0', 'image'),
            ('buffer/input1', 'data_loader/camera_info'),
            ('input1', 'camera_info'),
        ],
    )


def launch_setup(container_prefix, container_sigterm_timeout):
    ns = TestGpuRosRtDetrPhase2aAmd.generate_namespace()

    image_encoder_node = ComposableNode(
        name='RtdetrImageEncoder',
        namespace=ns,
        package='gpu_ros_rtdetr',
        plugin='gpu_ros::rtdetr::RtDetrImageEncoderNode',
        parameters=[
            {
                'tensor_name': 'input_tensor',
                'output_width': common.NETWORK_RESOLUTION['width'],
                'output_height': common.NETWORK_RESOLUTION['height'],
            }
        ],
    )

    preprocessor_node = ComposableNode(
        name='RtdetrPreprocessor',
        namespace=ns,
        package='gpu_ros_rtdetr',
        plugin='gpu_ros::rtdetr::RtDetrPreprocessorNode',
        parameters=[
            {
                'image_width': common.NETWORK_RESOLUTION['width'],
                'image_height': common.NETWORK_RESOLUTION['height'],
                'use_max_dim_for_orig_size': True,
            }
        ],
    )

    onnx_node = ComposableNode(
        name='OnnxInference',
        namespace=ns,
        package='gpu_ros_onnx_inference',
        plugin='gpu_ros::onnx_inference::OnnxInferenceNode',
        parameters=[
            {
                'model_file_path': os.path.join(
                    TestGpuRosRtDetrPhase2aAmd.get_assets_root_path(),
                    'models',
                    common.AMD_MODEL_FILE_NAME,
                ),
                'execution_provider': 'migraphx',
                'transport': 'std',
            }
        ],
        remappings=[('tensor_input', 'tensor_pub'), ('tensor_output', 'tensor_sub')],
    )

    decoder_node = ComposableNode(
        name='RtdetrDecoder',
        namespace=ns,
        package='gpu_ros_rtdetr',
        plugin='gpu_ros::rtdetr::RtDetrDecoderNode',
    )

    container = ComposableNodeContainer(
        name='container',
        namespace=ns,
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            common.make_data_loader_node(ns),
            make_std_playback_node(ns),
            image_encoder_node,
            preprocessor_node,
            onnx_node,
            decoder_node,
            common.make_monitor_node(ns),
        ],
        output='screen',
    )
    return [container]


def generate_test_description():
    return TestGpuRosRtDetrPhase2aAmd.generate_test_description_with_nsys(launch_setup)


class TestGpuRosRtDetrPhase2aAmd(ROS2BenchmarkTest):
    """Phase 2a AMD target: ONNX Runtime MIGraphX + standard ROS2 transport."""

    config = ROS2BenchmarkConfig(
        benchmark_name='GPU ROS RT-DETR Phase 2A AMD (ORT MIGraphX + std ROS2)',
        input_data_path=common.ROSBAG_PATH,
        publisher_upper_frequency=1000.0,
        publisher_lower_frequency=1.0,
        additional_fixed_publisher_rate_tests=[10.0, 30.0, 60.0],
        playback_message_buffer_size=1,
        pre_trial_run_wait_time_sec=5.0,
        log_folder=RESULTS_DIR,
        log_file_name=RESULTS_FILE,
        custom_report_info={
            'data_resolution': common.IMAGE_RESOLUTION,
            'network_resolution': common.NETWORK_RESOLUTION,
            'inference_backend': 'ONNX Runtime MIGraphX EP',
            'transport': 'standard ROS2 TensorBundle',
            'build_type': 'Release',
            'result_directory': RESULTS_DIR,
        },
    )

    def prepare_buffer(self):
        """Buffer the input and finish lazy MIGraphX compilation before measurement."""
        super().prepare_buffer()
        if getattr(self, '_migraphx_warmup_complete', False):
            return

        detection_received = False

        def on_detection(_message):
            nonlocal detection_received
            detection_received = True

        subscription = self.node.create_subscription(
            Detection2DArray, 'detections_output', on_detection, 10
        )
        try:
            client = self.create_service_client_blocking(PlayMessages, 'play_messages')
            request = PlayMessages.Request()
            request.playback_mode = BenchmarkMode.LOOPING.value
            request.target_publisher_rate = 1.0
            request.message_count = 1
            request.enforce_publisher_rate = False
            request.revise_timestamps_as_message_ids = False

            self.get_logger().info(
                'Starting one-frame MIGraphX warm-up; waiting for detections_output'
            )
            future = client.call_async(request)
            deadline = time.monotonic() + MIGRAPHX_WARMUP_TIMEOUT_SEC
            while not detection_received and time.monotonic() < deadline:
                rclpy.spin_once(self.node, timeout_sec=0.5)
                if future.done() and future.exception() is not None:
                    raise RuntimeError('MIGraphX warm-up playback failed') from future.exception()

            if not detection_received:
                raise RuntimeError(
                    'MIGraphX warm-up did not produce detections within '
                    f'{MIGRAPHX_WARMUP_TIMEOUT_SEC:.0f} seconds'
                )
            self._migraphx_warmup_complete = True
            self.get_logger().info('MIGraphX warm-up complete; starting measured benchmark')
        finally:
            self.node.destroy_subscription(subscription)

    def test_benchmark(self):
        self.run_benchmark()
