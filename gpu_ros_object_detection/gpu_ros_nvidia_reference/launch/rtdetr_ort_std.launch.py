# Copyright 2026 Boshen Chen
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
RT-DETR config D: ONNX Runtime + standard ROS2 migrated pipeline.

This is the vendor-neutral target.

Forks the upstream isaac_ros_rtdetr_oss launch, replacing the final three nodes
(preprocessor / inference / decoder) with our standard-ROS2 + ORT versions.
The upstream 6-node NITROS preprocessing chain is reused, then crosses an
explicit NVIDIA TensorList-to-project-TensorBundle boundary before entering
the project standard path. All nodes share one component container.
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
        DeclareLaunchArgument(
            'model_file_path',
            default_value='',
            description='Absolute path to the RT-DETR ONNX file',
        ),
        DeclareLaunchArgument('model_profile', default_value='auto'),
        DeclareLaunchArgument('model_assets_root', default_value='/workspaces/ovg-assets'),
        DeclareLaunchArgument(
            'input_image_width', default_value='640', description='The input image width'
        ),
        DeclareLaunchArgument(
            'input_image_height', default_value='480', description='The input image height'
        ),
        DeclareLaunchArgument(
            'execution_provider',
            default_value='cuda',
            description='ONNX Runtime execution provider (cuda/rocm/migraphx/cpu)',
        ),
        DeclareLaunchArgument(
            'ort_profile_prefix',
            default_value='',
            description='Enable ORT profiling and write JSON using this path prefix',
        ),
        DeclareLaunchArgument(
            'ort_profile_frames',
            default_value='0',
            description='Number of inference frames to include in the ORT profile',
        ),
        DeclareLaunchArgument(
            'binding_report_path',
            default_value='',
            description='Write the first-frame ORT pointer/lifetime report here',
        ),
        DeclareLaunchArgument(
            'confidence_threshold',
            default_value='0.6',
            description='Minimum score for a bounding box to be published',
        ),
    ]

    input_image_width = LaunchConfiguration('input_image_width')
    input_image_height = LaunchConfiguration('input_image_height')
    model_file_path = LaunchConfiguration('model_file_path')
    model_profile = LaunchConfiguration('model_profile')
    model_assets_root = LaunchConfiguration('model_assets_root')
    execution_provider = LaunchConfiguration('execution_provider')
    ort_profile_prefix = LaunchConfiguration('ort_profile_prefix')
    ort_profile_frames = LaunchConfiguration('ort_profile_frames')
    binding_report_path = LaunchConfiguration('binding_report_path')
    confidence_threshold = LaunchConfiguration('confidence_threshold')

    resize_node = ComposableNode(
        name='resize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[
            {
                'input_width': input_image_width,
                'input_height': input_image_height,
                'output_width': MODEL_INPUT_SIZE,
                'output_height': MODEL_INPUT_SIZE,
                'keep_aspect_ratio': True,
                'encoding_desired': 'rgb8',
                'disable_padding': True,
            }
        ],
        remappings=[('image', 'image_rect'), ('camera_info', 'camera_info_rect')],
    )

    pad_node = ComposableNode(
        name='pad_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::PadNode',
        parameters=[
            {
                'output_image_width': MODEL_INPUT_SIZE,
                'output_image_height': MODEL_INPUT_SIZE,
                'padding_type': 'BOTTOM_RIGHT',
            }
        ],
        remappings=[('image', 'resize/image')],
    )

    image_format_node = ComposableNode(
        name='image_format_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        parameters=[
            {
                'encoding_desired': 'rgb8',
                'image_width': MODEL_INPUT_SIZE,
                'image_height': MODEL_INPUT_SIZE,
            }
        ],
        remappings=[('image_raw', 'padded_image'), ('image', 'image_rgb')],
    )

    image_to_tensor_node = ComposableNode(
        name='image_to_tensor_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{'scale': False, 'tensor_name': 'image'}],
        remappings=[('image', 'image_rgb'), ('tensor', 'normalized_tensor')],
    )

    interleave_to_planar_node = ComposableNode(
        name='interleaved_to_planar_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[
            {'input_tensor_shape': [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, MODEL_NUM_CHANNELS]}
        ],
        remappings=[('interleaved_tensor', 'normalized_tensor')],
    )

    reshape_node = ComposableNode(
        name='reshape_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[
            {
                'output_tensor_name': 'input_tensor',
                'input_tensor_shape': [MODEL_NUM_CHANNELS, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
                'output_tensor_shape': [1, MODEL_NUM_CHANNELS, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
            }
        ],
        remappings=[('tensor', 'planar_tensor')],
    )

    # The NVIDIA reshape node publishes TensorList. Convert it explicitly at
    # the NVIDIA-only boundary before entering the project-owned std path.
    tensor_list_adapter_node = ComposableNode(
        name='nvidia_tensor_list_to_tensor_bundle',
        package='gpu_ros_nvidia_tensor_bundle_compat',
        plugin=('gpu_ros::nvidia_tensor_bundle_compat::NvidiaTensorListToTensorBundleNode'),
        remappings=[
            ('tensor_input', 'reshaped_tensor'),
            ('tensor_output', 'tensor_bundle_input'),
        ],
    )

    # --- std-ROS2 + ORT replacements for the final three nodes ---

    rtdetr_preprocessor_node = ComposableNode(
        name='rtdetr_preprocessor',
        package='gpu_ros_rtdetr',
        plugin='gpu_ros::rtdetr::RtDetrPreprocessorNode',
        parameters=[
            {
                'image_width': input_image_width,
                'image_height': input_image_height,
                # Match the baseline graph: it passes a bogus 'image_size' param the
                # node ignores, so it falls back to use_max_dim_for_orig_size=true
                # (orig_target_sizes=[640,640]). Keep that default for A/D parity.
            }
        ],
        remappings=[('encoded_tensor', 'tensor_bundle_input')],
    )

    onnx_node = ComposableNode(
        name='onnx_inference',
        package='gpu_ros_onnx_inference',
        plugin='gpu_ros::onnx_inference::OnnxInferenceNode',
        parameters=[
            {
                'model_file_path': model_file_path,
                'model_profile': model_profile,
                'model_assets_root': model_assets_root,
                'execution_provider': execution_provider,
                'ort_profile_prefix': ort_profile_prefix,
                'ort_profile_frames': ort_profile_frames,
                'binding_report_path': binding_report_path,
                'transport': 'std',
            }
        ],
        remappings=[
            ('tensor_input', 'tensor_pub'),
            ('tensor_output', 'tensor_sub'),
        ],
    )

    rtdetr_decoder_node = ComposableNode(
        name='rtdetr_decoder',
        package='gpu_ros_rtdetr',
        plugin='gpu_ros::rtdetr::RtDetrDecoderNode',
        parameters=[{'confidence_threshold': confidence_threshold}],
    )

    container = ComposableNodeContainer(
        name='rtdetr_container',
        namespace='rtdetr_container',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            resize_node,
            pad_node,
            image_format_node,
            image_to_tensor_node,
            interleave_to_planar_node,
            reshape_node,
            tensor_list_adapter_node,
            rtdetr_preprocessor_node,
            onnx_node,
            rtdetr_decoder_node,
        ],
        output='screen',
    )

    return launch.LaunchDescription(launch_args + [container])
