# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Config D benchmark: ONNX Runtime CUDA + standard TensorList + std YOLOv8 decoder."""

import os
import sys

sys.path.append(os.path.dirname(__file__))
import yolov8_common as common  # noqa: E402

from launch_ros.actions import ComposableNodeContainer  # noqa: E402
from launch_ros.descriptions import ComposableNode  # noqa: E402

from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest  # noqa: E402


def launch_setup(container_prefix, container_sigterm_timeout):
    ns = TestIsaacROSYoloV8ConfigD.generate_namespace()

    onnx_node = ComposableNode(
        name='OnnxInference',
        namespace=ns,
        package='isaac_ros_onnx_inference',
        plugin='nvidia::isaac_ros::onnx_inference::OnnxInferenceNode',
        parameters=[{
            'model_file_path': os.path.join(
                TestIsaacROSYoloV8ConfigD.get_assets_root_path(),
                'models', common.MODEL_FILE_NAME),
            'execution_provider': 'cuda',
            'transport': 'std',
        }],
        remappings=[('tensor_input', 'reshaped_tensor'), ('tensor_output', 'tensor_sub')]
    )

    decoder_node = ComposableNode(
        name='YoloV8Decoder',
        namespace=ns,
        package='isaac_ros_yolov8_std',
        plugin='nvidia::isaac_ros::yolov8_std::YoloV8DecoderNode',
        parameters=[{
            'tensor_name': common.ORT_OUTPUT_TENSOR_NAME,
            'confidence_threshold': 0.25,
            'nms_threshold': 0.45,
            'num_classes': 80,
        }]
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
            *common.make_preprocessing_nodes(ns, common.ORT_INPUT_TENSOR_NAME),
            onnx_node, decoder_node,
            common.make_monitor_node(ns),
        ],
        output='screen',
    )
    return [container]


def generate_test_description():
    return TestIsaacROSYoloV8ConfigD.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSYoloV8ConfigD(ROS2BenchmarkTest):
    """Config D: ONNX Runtime CUDA + standard TensorList + std YOLOv8 decoder."""

    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS YOLOv8 (D: ORT + std ROS2)',
        input_data_path=common.ROSBAG_PATH,
        publisher_upper_frequency=1000.0,
        publisher_lower_frequency=10.0,
        playback_message_buffer_size=1,
        pre_trial_run_wait_time_sec=5.0,
        custom_report_info={
            'data_resolution': common.IMAGE_RESOLUTION,
            'network_resolution': common.NETWORK_RESOLUTION
        }
    )

    def test_benchmark(self):
        self.run_benchmark()

# Modified derived source; upstream NVIDIA Apache-2.0 attribution retained.
