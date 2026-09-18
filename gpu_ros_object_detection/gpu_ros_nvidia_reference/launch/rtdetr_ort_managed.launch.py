# Copyright 2026 Boshen Chen
# NVIDIA/Isaac-specific reference launch owned by gpu_ros_nvidia_reference.
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

"""RT-DETR: official NITROS pre/post-processing with Managed ORT transport."""

import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    model_file_path = LaunchConfiguration('model_file_path')
    model_profile = LaunchConfiguration('model_profile')
    model_assets_root = LaunchConfiguration('model_assets_root')
    input_image_width = LaunchConfiguration('input_image_width')
    input_image_height = LaunchConfiguration('input_image_height')
    confidence_threshold = LaunchConfiguration('confidence_threshold')
    ort_profile_prefix = LaunchConfiguration('ort_profile_prefix')
    ort_profile_frames = LaunchConfiguration('ort_profile_frames')
    binding_report_path = LaunchConfiguration('binding_report_path')

    arguments = [
        DeclareLaunchArgument('model_file_path', default_value=''),
        DeclareLaunchArgument('model_profile', default_value='auto'),
        DeclareLaunchArgument('model_assets_root', default_value='/workspaces/ovg-assets'),
        DeclareLaunchArgument('input_image_width', default_value='640'),
        DeclareLaunchArgument('input_image_height', default_value='480'),
        DeclareLaunchArgument('confidence_threshold', default_value='0.6'),
        DeclareLaunchArgument('ort_profile_prefix', default_value=''),
        DeclareLaunchArgument('ort_profile_frames', default_value='0'),
        DeclareLaunchArgument('binding_report_path', default_value=''),
    ]

    resize = ComposableNode(
        name='resize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[
            {
                'input_width': input_image_width,
                'input_height': input_image_height,
                'output_width': 640,
                'output_height': 640,
                'keep_aspect_ratio': True,
                'encoding_desired': 'rgb8',
                'disable_padding': True,
            }
        ],
        remappings=[('image', 'image_rect'), ('camera_info', 'camera_info_rect')],
    )
    pad = ComposableNode(
        name='pad_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::PadNode',
        parameters=[
            {'output_image_width': 640, 'output_image_height': 640, 'padding_type': 'BOTTOM_RIGHT'}
        ],
        remappings=[('image', 'resize/image')],
    )
    format_converter = ComposableNode(
        name='image_format_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        parameters=[{'encoding_desired': 'rgb8', 'image_width': 640, 'image_height': 640}],
        remappings=[('image_raw', 'padded_image'), ('image', 'image_rgb')],
    )
    image_to_tensor = ComposableNode(
        name='image_to_tensor_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{'scale': False, 'tensor_name': 'image'}],
        remappings=[('image', 'image_rgb'), ('tensor', 'normalized_tensor')],
    )
    interleaved_to_planar = ComposableNode(
        name='interleaved_to_planar_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[{'input_tensor_shape': [640, 640, 3]}],
        remappings=[('interleaved_tensor', 'normalized_tensor')],
    )
    reshape = ComposableNode(
        name='reshape_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[
            {
                'output_tensor_name': 'input_tensor',
                'input_tensor_shape': [3, 640, 640],
                'output_tensor_shape': [1, 3, 640, 640],
            }
        ],
        remappings=[('tensor', 'planar_tensor')],
    )
    preprocessor = ComposableNode(
        name='rtdetr_preprocessor',
        package='isaac_ros_rtdetr',
        plugin='nvidia::isaac_ros::rtdetr::RtDetrPreprocessorNode',
        parameters=[{'image_width': input_image_width, 'image_height': input_image_height}],
        remappings=[('encoded_tensor', 'reshaped_tensor')],
    )
    nitros_to_managed = ComposableNode(
        name='nitros_to_managed',
        package='gpu_ros_onnx_inference',
        plugin='gpu_ros::onnx_inference::NitrosToManagedTensorBundleNode',
        remappings=[('tensor_input', 'tensor_pub'), ('tensor_output', 'managed_tensor_input')],
    )
    onnx = ComposableNode(
        name='onnx_inference',
        package='gpu_ros_onnx_inference',
        plugin='gpu_ros::onnx_inference::OnnxInferenceNode',
        parameters=[
            {
                'model_file_path': model_file_path,
                'model_profile': model_profile,
                'model_assets_root': model_assets_root,
                'execution_provider': 'cuda',
                'ort_profile_prefix': ort_profile_prefix,
                'ort_profile_frames': ort_profile_frames,
                'binding_report_path': binding_report_path,
                'transport': 'managed',
            }
        ],
        remappings=[
            ('tensor_input', 'managed_tensor_input'),
            ('tensor_output', 'managed_tensor_output'),
        ],
    )
    managed_to_nitros = ComposableNode(
        name='managed_to_nitros',
        package='gpu_ros_onnx_inference',
        plugin='gpu_ros::onnx_inference::ManagedToNitrosTensorBundleNode',
        remappings=[('tensor_input', 'managed_tensor_output'), ('tensor_output', 'tensor_sub')],
    )
    decoder = ComposableNode(
        name='rtdetr_decoder',
        package='isaac_ros_rtdetr',
        plugin='nvidia::isaac_ros::rtdetr::RtDetrDecoderNode',
        parameters=[{'confidence_threshold': confidence_threshold}],
    )

    container = ComposableNodeContainer(
        name='rtdetr_managed_container',
        namespace='rtdetr_managed_container',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            resize,
            pad,
            format_converter,
            image_to_tensor,
            interleaved_to_planar,
            reshape,
            preprocessor,
            nitros_to_managed,
            onnx,
            managed_to_nitros,
            decoder,
        ],
        output='screen',
    )
    return launch.LaunchDescription(arguments + [container])
