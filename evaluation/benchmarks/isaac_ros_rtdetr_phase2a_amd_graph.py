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
"""Phase 2a AMD benchmark: RT-DETR std image encoder + ORT MIGraphX.

This benchmark covers only the AMD target path:
Image -> std RT-DETR image encoder -> std RtDetrPreprocessor ->
OnnxInferenceNode(transport=std, execution_provider=migraphx) ->
std RtDetrDecoder -> Detection2DArray.
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))
import rtdetr_common as common  # noqa: E402

from launch_ros.actions import ComposableNodeContainer  # noqa: E402
from launch_ros.descriptions import ComposableNode  # noqa: E402

from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest  # noqa: E402

RESULTS_DIR = 'migrated_packages/benchmark_results'
RESULTS_FILE = 'phase2a-rtdetr-amd-migraphx.json'


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
            ('input1', 'camera_info')
        ]
    )


def launch_setup(container_prefix, container_sigterm_timeout):
    ns = TestIsaacROSRtDetrPhase2aAmd.generate_namespace()

    image_encoder_node = ComposableNode(
        name='RtdetrImageEncoder',
        namespace=ns,
        package='isaac_ros_rtdetr_std',
        plugin='nvidia::isaac_ros::rtdetr_std::RtDetrImageEncoderNode',
        parameters=[{
            'tensor_name': 'input_tensor',
            'output_width': common.NETWORK_RESOLUTION['width'],
            'output_height': common.NETWORK_RESOLUTION['height'],
        }],
    )

    preprocessor_node = ComposableNode(
        name='RtdetrPreprocessor',
        namespace=ns,
        package='isaac_ros_rtdetr_std',
        plugin='nvidia::isaac_ros::rtdetr_std::RtDetrPreprocessorNode',
        parameters=[{
            'image_width': common.NETWORK_RESOLUTION['width'],
            'image_height': common.NETWORK_RESOLUTION['height'],
            'use_max_dim_for_orig_size': True,
        }],
    )

    onnx_node = ComposableNode(
        name='OnnxInference',
        namespace=ns,
        package='isaac_ros_onnx_inference',
        plugin='nvidia::isaac_ros::onnx_inference::OnnxInferenceNode',
        parameters=[{
            'model_file_path': os.path.join(
                TestIsaacROSRtDetrPhase2aAmd.get_assets_root_path(),
                'models', common.MODEL_FILE_NAME),
            'execution_provider': 'migraphx',
            'ort_profile_prefix': os.environ.get('ORT_PROFILE_PREFIX', ''),
            'transport': 'std',
        }],
        remappings=[('tensor_input', 'tensor_pub'), ('tensor_output', 'tensor_sub')]
    )

    decoder_node = ComposableNode(
        name='RtdetrDecoder',
        namespace=ns,
        package='isaac_ros_rtdetr_std',
        plugin='nvidia::isaac_ros::rtdetr_std::RtDetrDecoderNode',
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
    return TestIsaacROSRtDetrPhase2aAmd.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSRtDetrPhase2aAmd(ROS2BenchmarkTest):
    """Phase 2a AMD target: ONNX Runtime MIGraphX + standard ROS2 transport."""

    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS RT-DETR Phase 2a AMD (ORT MIGraphX + std ROS2)',
        input_data_path=common.ROSBAG_PATH,
        publisher_upper_frequency=1000.0,
        publisher_lower_frequency=10.0,
        playback_message_buffer_size=1,
        pre_trial_run_wait_time_sec=5.0,
        log_folder=RESULTS_DIR,
        log_file_name=RESULTS_FILE,
        custom_report_info={
            'data_resolution': common.IMAGE_RESOLUTION,
            'network_resolution': common.NETWORK_RESOLUTION,
            'inference_backend': 'ONNX Runtime MIGraphX EP',
            'transport': 'standard ROS2 TensorList',
            'result_path': os.path.join(RESULTS_DIR, RESULTS_FILE),
        }
    )

    def test_benchmark(self):
        self.run_benchmark()

# Modified derived source; upstream NVIDIA Apache-2.0 attribution retained.
