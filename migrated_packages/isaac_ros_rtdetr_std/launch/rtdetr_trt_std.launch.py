# Copyright 2026 Maintainer
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

"""
RT-DETR config B: TensorRT + standard ROS2 transport.

Upstream NITROS preprocess chain + upstream TensorRT inference node, but our
std-ROS2 RtDetrPreprocessor/Decoder. The TensorRT NITROS publisher bridges to
our std decoder via NITROS auto-compat. Isolates the NITROS transport cost on
the TensorRT backend (compare against config A).
"""

import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

MODEL_INPUT_SIZE = 640
MODEL_NUM_CHANNELS = 3


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument('engine_file_path', default_value='',
                              description='Absolute path to the FP16 TensorRT engine'),
        DeclareLaunchArgument('input_image_width', default_value='640'),
        DeclareLaunchArgument('input_image_height', default_value='480'),
        DeclareLaunchArgument('confidence_threshold', default_value='0.6'),
    ]

    input_image_width = LaunchConfiguration('input_image_width')
    input_image_height = LaunchConfiguration('input_image_height')
    engine_file_path = LaunchConfiguration('engine_file_path')
    confidence_threshold = LaunchConfiguration('confidence_threshold')

    resize_node = ComposableNode(
        name='resize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': input_image_width,
            'input_height': input_image_height,
            'output_width': MODEL_INPUT_SIZE,
            'output_height': MODEL_INPUT_SIZE,
            'keep_aspect_ratio': True,
            'encoding_desired': 'rgb8',
            'disable_padding': True
        }],
        remappings=[('image', 'image_rect'), ('camera_info', 'camera_info_rect')],
    )

    pad_node = ComposableNode(
        name='pad_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::PadNode',
        parameters=[{
            'output_image_width': MODEL_INPUT_SIZE,
            'output_image_height': MODEL_INPUT_SIZE,
            'padding_type': 'BOTTOM_RIGHT'
        }],
        remappings=[('image', 'resize/image')]
    )

    image_format_node = ComposableNode(
        name='image_format_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        parameters=[{
            'encoding_desired': 'rgb8',
            'image_width': MODEL_INPUT_SIZE,
            'image_height': MODEL_INPUT_SIZE
        }],
        remappings=[('image_raw', 'padded_image'), ('image', 'image_rgb')]
    )

    image_to_tensor_node = ComposableNode(
        name='image_to_tensor_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{'scale': False, 'tensor_name': 'image'}],
        remappings=[('image', 'image_rgb'), ('tensor', 'normalized_tensor')]
    )

    interleave_to_planar_node = ComposableNode(
        name='interleaved_to_planar_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[{
            'input_tensor_shape': [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, MODEL_NUM_CHANNELS]
        }],
        remappings=[('interleaved_tensor', 'normalized_tensor')]
    )

    reshape_node = ComposableNode(
        name='reshape_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[{
            'output_tensor_name': 'input_tensor',
            'input_tensor_shape': [MODEL_NUM_CHANNELS, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
            'output_tensor_shape': [1, MODEL_NUM_CHANNELS, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]
        }],
        remappings=[('tensor', 'planar_tensor')],
    )

    # Our std-ROS2 preprocessor (receives upstream NITROS via auto-compat).
    # Publishes std-ROS2 on 'tensor_pub'.
    rtdetr_preprocessor_node = ComposableNode(
        name='rtdetr_preprocessor',
        package='isaac_ros_rtdetr_std',
        plugin='nvidia::isaac_ros::rtdetr_std::RtDetrPreprocessorNode',
        parameters=[{
            'image_width': input_image_width,
            'image_height': input_image_height,
            # Match baseline: default use_max_dim_for_orig_size=true ([640,640]).
        }],
        remappings=[('encoded_tensor', 'reshaped_tensor')]
    )

    # std -> NITROS bridge: the TensorRT node only speaks NITROS, and a std
    # publisher cannot feed a NITROS subscriber directly. Converts the
    # preprocessor's std output to NITROS on 'bridged_tensor'.
    bridge_node = ComposableNode(
        name='tensor_list_bridge',
        package='isaac_ros_onnx_inference',
        plugin='nvidia::isaac_ros::onnx_inference::TensorListBridgeNode',
        parameters=[{
            'input_transport': 'std',
            'enable_timing': True,
            'timing_log_every': 500,
        }],
        remappings=[
            ('tensor_input', 'tensor_pub'),
            ('tensor_output', 'bridged_tensor'),
        ]
    )

    # Upstream TensorRT NITROS inference node. Reads NITROS from the bridge.
    tensor_rt_node = ComposableNode(
        name='tensor_rt',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'engine_file_path': engine_file_path,
            'input_tensor_names': ['images', 'orig_target_sizes'],
            'input_binding_names': ['images', 'orig_target_sizes'],
            'output_binding_names': ['labels', 'boxes', 'scores'],
            'output_tensor_names': ['labels', 'boxes', 'scores'],
            'verbose': False,
            'force_engine_update': False
        }],
        remappings=[('tensor_pub', 'bridged_tensor')]
    )

    # Our std-ROS2 decoder (receives TensorRT NITROS pub via auto-compat).
    rtdetr_decoder_node = ComposableNode(
        name='rtdetr_decoder',
        package='isaac_ros_rtdetr_std',
        plugin='nvidia::isaac_ros::rtdetr_std::RtDetrDecoderNode',
        parameters=[{'confidence_threshold': confidence_threshold}],
    )

    container = ComposableNodeContainer(
        name='rtdetr_container',
        namespace='rtdetr_container',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            resize_node, pad_node, image_format_node,
            image_to_tensor_node, interleave_to_planar_node, reshape_node,
            rtdetr_preprocessor_node, bridge_node, tensor_rt_node, rtdetr_decoder_node
        ],
        output='screen'
    )

    return launch.LaunchDescription(launch_args + [container])
