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
"""Shared building blocks for the RT-DETR 2x2 benchmark matrix (configs A-D).

The framework nodes (data loader / playback / monitor) and the 6-node
preprocessing chain are identical across all four configurations and live here
so there is a single source of truth — only the inference segment differs per
config. Mirrors upstream isaac_ros_rtdetr_graph.py.
"""

from launch_ros.descriptions import ComposableNode

from ros2_benchmark import ImageResolution, Resolution

IMAGE_RESOLUTION = ImageResolution.HD
NETWORK_SIZE = 640  # RT-DETR architecture requires square network resolution
NETWORK_RESOLUTION = Resolution(NETWORK_SIZE, NETWORK_SIZE)
ROSBAG_PATH = 'datasets/r2bdataset2024_v1/r2b_robotarm'
MODEL_FILE_NAME = 'synthetica_detr_v1.0.0_onnx/sdetr_grasp_fp16.onnx'
ENGINE_FILE_PATH = '/tmp/sdetr_grasp_fp16.plan'


def make_preprocessing_nodes(namespace):
    """Return the 6-node preprocessing chain (raw image -> reshaped tensor)."""
    resize_node = ComposableNode(
        name='ResizeNode',
        namespace=namespace,
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': IMAGE_RESOLUTION['width'],
            'input_height': IMAGE_RESOLUTION['height'],
            'output_width': NETWORK_RESOLUTION['width'],
            'output_height': NETWORK_RESOLUTION['height'],
            'keep_aspect_ratio': True,
            'encoding_desired': 'rgb8',
            'disable_padding': True
        }]
    )

    pad_node = ComposableNode(
        name='PadNode',
        namespace=namespace,
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::PadNode',
        parameters=[{
            'output_image_width': NETWORK_RESOLUTION['width'],
            'output_image_height': NETWORK_RESOLUTION['height'],
            'padding_type': 'BOTTOM_RIGHT'
        }],
        remappings=[('image', 'resize/image')]
    )

    image_format_converter_node = ComposableNode(
        name='ImageFormatConverter',
        namespace=namespace,
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        parameters=[{
            'encoding_desired': 'rgb8',
            'image_width': NETWORK_RESOLUTION['width'],
            'image_height': NETWORK_RESOLUTION['height']
        }],
        remappings=[('image_raw', 'padded_image'), ('image', 'image_rgb')]
    )

    image_to_tensor_node = ComposableNode(
        name='ImageToTensorNode',
        namespace=namespace,
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{'scale': False, 'tensor_name': 'image'}],
        remappings=[('image', 'image_rgb'), ('tensor', 'normalized_tensor')]
    )

    interleave_to_planar_node = ComposableNode(
        name='InterleavedToPlanarNode',
        namespace=namespace,
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[{
            'input_tensor_shape': [NETWORK_RESOLUTION['width'], NETWORK_RESOLUTION['height'], 3]
        }],
        remappings=[('interleaved_tensor', 'normalized_tensor')]
    )

    reshape_node = ComposableNode(
        name='ReshapeNode',
        namespace=namespace,
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[{
            'output_tensor_name': 'input_tensor',
            'input_tensor_shape': [3, NETWORK_RESOLUTION['height'], NETWORK_RESOLUTION['width']],
            'output_tensor_shape': [
                1, 3, NETWORK_RESOLUTION['height'], NETWORK_RESOLUTION['width']]
        }],
        remappings=[('tensor', 'planar_tensor')],
    )

    return [
        resize_node, pad_node, image_format_converter_node,
        image_to_tensor_node, interleave_to_planar_node, reshape_node
    ]


def make_data_loader_node(namespace):
    return ComposableNode(
        name='DataLoaderNode',
        namespace=namespace,
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[
            ('camera_1/color/image_raw', 'data_loader/image_raw'),
            ('camera_1/color/camera_info', 'data_loader/camera_info')
        ]
    )


def make_playback_node(namespace):
    return ComposableNode(
        name='PlaybackNode',
        namespace=namespace,
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosPlaybackNode',
        parameters=[{'data_formats': ['nitros_image_bgr8', 'nitros_camera_info']}],
        remappings=[
            ('buffer/input0', 'data_loader/image_raw'),
            ('input0', 'image'),
            ('buffer/input1', 'data_loader/camera_info'),
            ('input1', 'camera_info')
        ]
    )


def make_monitor_node(namespace):
    return ComposableNode(
        name='MonitorNode',
        namespace=namespace,
        package='ros2_benchmark',
        plugin='ros2_benchmark::MonitorNode',
        parameters=[{'monitor_data_format': 'vision_msgs/msg/Detection2DArray'}],
        remappings=[('output', 'detections_output')],
    )

# Modified derived source; upstream NVIDIA Apache-2.0 attribution retained.
