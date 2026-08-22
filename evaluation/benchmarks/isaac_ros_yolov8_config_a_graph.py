# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modified from the NVIDIA Isaac ROS YOLOv8 benchmark composition; see
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
"""Config A benchmark: TensorRT + NITROS + upstream YOLOv8 decoder."""

import os
import sys

from isaac_ros_benchmark import TRTConverter

sys.path.append(os.path.dirname(__file__))
import yolov8_common as common  # noqa: E402

from launch_ros.actions import ComposableNodeContainer  # noqa: E402
from launch_ros.descriptions import ComposableNode  # noqa: E402

from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest  # noqa: E402


def launch_setup(container_prefix, container_sigterm_timeout):
    ns = TestIsaacROSYoloV8ConfigA.generate_namespace()

    tensor_rt_node = ComposableNode(
        name='TensorRt',
        namespace=ns,
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'model_file_path': os.path.join(
                TestIsaacROSYoloV8ConfigA.get_assets_root_path(),
                'models', common.MODEL_FILE_NAME),
            'engine_file_path': common.ENGINE_FILE_PATH,
            'input_tensor_names': [common.TRT_INPUT_TENSOR_NAME],
            'input_binding_names': [common.INPUT_BINDING_NAME],
            'output_binding_names': [common.OUTPUT_BINDING_NAME],
            'output_tensor_names': [common.TRT_OUTPUT_TENSOR_NAME],
            'verbose': False,
            'force_engine_update': False
        }],
        remappings=[('tensor_pub', 'reshaped_tensor')]
    )

    decoder_node = ComposableNode(
        name='YoloV8Decoder',
        namespace=ns,
        package='isaac_ros_yolov8',
        plugin='nvidia::isaac_ros::yolov8::YoloV8DecoderNode',
        parameters=[{
            'tensor_name': common.TRT_OUTPUT_TENSOR_NAME,
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
            *common.make_preprocessing_nodes(ns, common.TRT_INPUT_TENSOR_NAME),
            tensor_rt_node, decoder_node,
            common.make_monitor_node(ns),
        ],
        output='screen',
    )
    return [container]


def generate_test_description():
    model_path = os.path.join(
        TestIsaacROSYoloV8ConfigA.get_assets_root_path(), 'models', common.MODEL_FILE_NAME)
    if not os.path.isfile(common.ENGINE_FILE_PATH):
        TRTConverter()([
            f'--onnx={model_path}',
            f'--saveEngine={common.ENGINE_FILE_PATH}',
            '--fp16',
            '--skipInference',
        ])
    return TestIsaacROSYoloV8ConfigA.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSYoloV8ConfigA(ROS2BenchmarkTest):
    """Config A: TensorRT + NITROS + upstream YOLOv8 decoder."""

    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS YOLOv8 (A: TRT + NITROS)',
        input_data_path=common.ROSBAG_PATH,
        publisher_upper_frequency=1000.0,
        publisher_lower_frequency=10.0,
        playback_message_buffer_size=1,
        pre_trial_run_wait_time_sec=5.0,
        custom_report_info={
            'data_resolution': common.IMAGE_RESOLUTION,
            'network_resolution': common.NETWORK_RESOLUTION,
            'build_type': 'Release',
            'inference_precision': 'FP16',
        }
    )

    def test_benchmark(self):
        self.run_benchmark()
