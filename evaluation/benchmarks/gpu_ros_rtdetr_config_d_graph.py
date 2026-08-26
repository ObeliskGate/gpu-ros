# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modified from the NVIDIA Isaac ROS RT-DETR benchmark composition; see
# THIRD_PARTY_NOTICES.md for the exact pinned revision and local boundary.
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
"""Config D benchmark: ONNX Runtime (CUDA EP) + standard ROS2 migrated pipeline.

The vendor-neutral target stack. Shares the framework + 6-node preprocessing
with configs A/B/C (see rtdetr_common); only the inference segment differs:
our std-ROS2 RtDetrPreprocessor + OnnxInferenceNode(std) + RtDetrDecoder.
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))
import rtdetr_common as common  # noqa: E402

from launch_ros.actions import ComposableNodeContainer  # noqa: E402
from launch_ros.descriptions import ComposableNode  # noqa: E402

from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest  # noqa: E402


def launch_setup(container_prefix, container_sigterm_timeout):
    ns = TestGpuRosRtDetrConfigD.generate_namespace()

    tensor_list_adapter_node = ComposableNode(
        name='NvidiaTensorListToTensorBundle',
        namespace=ns,
        package='gpu_ros_nvidia_tensor_bundle_compat',
        plugin=(
            'gpu_ros::nvidia_tensor_bundle_compat::'
            'NvidiaTensorListToTensorBundleNode'),
        remappings=[
            ('tensor_input', 'reshaped_tensor'),
            ('tensor_output', 'tensor_bundle_input'),
        ],
    )

    preprocessor_node = ComposableNode(
        name='RtdetrPreprocessor',
        namespace=ns,
        package='gpu_ros_rtdetr',
        plugin='gpu_ros::rtdetr::RtDetrPreprocessorNode',
        remappings=[('encoded_tensor', 'tensor_bundle_input')]
    )

    onnx_node = ComposableNode(
        name='OnnxInference',
        namespace=ns,
        package='gpu_ros_onnx_inference',
        plugin='gpu_ros::onnx_inference::OnnxInferenceNode',
        parameters=[{
            'model_file_path': os.path.join(
                TestGpuRosRtDetrConfigD.get_assets_root_path(),
                'models', common.MODEL_FILE_NAME),
            'execution_provider': 'cuda',
            'transport': 'std',
        }],
        remappings=[('tensor_input', 'tensor_pub'), ('tensor_output', 'tensor_sub')]
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
            common.make_playback_node(ns),
            *common.make_preprocessing_nodes(ns),
            tensor_list_adapter_node, preprocessor_node, onnx_node, decoder_node,
            common.make_monitor_node(ns),
        ],
        output='screen',
    )
    return [container]


def generate_test_description():
    return TestGpuRosRtDetrConfigD.generate_test_description_with_nsys(launch_setup)


class TestGpuRosRtDetrConfigD(ROS2BenchmarkTest):
    """Config D: ONNX Runtime + standard ROS2 migrated pipeline."""

    config = ROS2BenchmarkConfig(
        benchmark_name='GPU ROS RT-DETR (NVIDIA reference D: ORT + std ROS2)',
        input_data_path=common.ROSBAG_PATH,
        publisher_upper_frequency=1000.0,
        publisher_lower_frequency=10.0,
        playback_message_buffer_size=1,
        pre_trial_run_wait_time_sec=5.0,
        custom_report_info={
            'data_resolution': common.IMAGE_RESOLUTION,
            'network_resolution': common.NETWORK_RESOLUTION,
            'build_type': 'Release',
        }
    )

    def test_benchmark(self):
        self.run_benchmark()
