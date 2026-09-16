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
"""Config C benchmark: ONNX Runtime (CUDA EP) + NITROS transport.

Isolates the inference-backend cost under NITROS transport: upstream NITROS
RtDetrPreprocessor/Decoder, with only the inference node swapped from TensorRT
to OnnxInferenceNode(transport=nitros). Shares framework + preprocessing with
the other configs (see rtdetr_common).
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))
import rtdetr_common as common  # noqa: E402

from launch_ros.actions import ComposableNodeContainer  # noqa: E402
from launch_ros.descriptions import ComposableNode  # noqa: E402

from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest  # noqa: E402


def launch_setup(container_prefix, container_sigterm_timeout):
    ns = TestGpuRosRtDetrConfigC.generate_namespace()

    preprocessor_node = ComposableNode(
        name='RtdetrPreprocessor',
        namespace=ns,
        package='isaac_ros_rtdetr',
        plugin='nvidia::isaac_ros::rtdetr::RtDetrPreprocessorNode',
        parameters=[{'image_size': common.NETWORK_RESOLUTION['width']}],
        remappings=[('encoded_tensor', 'reshaped_tensor')],
    )

    onnx_node = ComposableNode(
        name='OnnxInference',
        namespace=ns,
        package='gpu_ros_onnx_inference',
        plugin='gpu_ros::onnx_inference::OnnxInferenceNode',
        parameters=[
            {
                'model_file_path': os.path.join(
                    TestGpuRosRtDetrConfigC.get_assets_root_path(), 'models', common.MODEL_FILE_NAME
                ),
                'execution_provider': 'cuda',
                'transport': 'nitros',
            }
        ],
        remappings=[('tensor_input', 'tensor_pub'), ('tensor_output', 'tensor_sub')],
    )

    decoder_node = ComposableNode(
        name='RtdetrDecoder',
        namespace=ns,
        package='isaac_ros_rtdetr',
        plugin='nvidia::isaac_ros::rtdetr::RtDetrDecoderNode',
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
            preprocessor_node,
            onnx_node,
            decoder_node,
            common.make_monitor_node(ns),
        ],
        output='screen',
    )
    return [container]


def generate_test_description():
    return TestGpuRosRtDetrConfigC.generate_test_description_with_nsys(launch_setup)


class TestGpuRosRtDetrConfigC(ROS2BenchmarkTest):
    """Config C: ONNX Runtime + NITROS transport."""

    config = ROS2BenchmarkConfig(
        benchmark_name='GPU ROS RT-DETR (NVIDIA reference C: ORT + NITROS)',
        input_data_path=common.ROSBAG_PATH,
        publisher_upper_frequency=1000.0,
        publisher_lower_frequency=10.0,
        playback_message_buffer_size=1,
        pre_trial_run_wait_time_sec=5.0,
        custom_report_info={
            'data_resolution': common.IMAGE_RESOLUTION,
            'network_resolution': common.NETWORK_RESOLUTION,
            'build_type': 'Release',
        },
    )

    def test_benchmark(self):
        self.run_benchmark()
