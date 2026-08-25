# Copyright 2026 Boshen Chen
# NVIDIA/Isaac-specific reference launch owned by the compatibility package.
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

"""Fixed-input YOLOv8 graph for ORT CUDA NITROS/Managed comparisons."""

import launch
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


MODEL_INPUT_SIZE = 640
MODEL_NUM_CHANNELS = 3


def launch_setup(context):
    """Create the graph after resolving the selected transport."""
    transport = LaunchConfiguration('transport').perform(context)
    if transport not in ('nitros', 'managed'):
        raise RuntimeError("transport must be either 'nitros' or 'managed'")

    model_file_path = LaunchConfiguration('model_file_path')
    execution_provider = LaunchConfiguration('execution_provider')
    ort_profile_prefix = LaunchConfiguration('ort_profile_prefix')
    ort_profile_frames = LaunchConfiguration('ort_profile_frames')
    binding_report_path = LaunchConfiguration('binding_report_path')
    input_image_width = LaunchConfiguration('input_image_width')
    input_image_height = LaunchConfiguration('input_image_height')
    confidence_threshold = LaunchConfiguration('confidence_threshold')
    nms_threshold = LaunchConfiguration('nms_threshold')

    resize = ComposableNode(
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
            'disable_padding': True,
        }],
        remappings=[('image', 'image_rect'), ('camera_info', 'camera_info_rect')],
    )
    pad = ComposableNode(
        name='pad_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::PadNode',
        parameters=[{
            'output_image_width': MODEL_INPUT_SIZE,
            'output_image_height': MODEL_INPUT_SIZE,
            'padding_type': 'BOTTOM_RIGHT',
        }],
        remappings=[('image', 'resize/image')],
    )
    format_converter = ComposableNode(
        name='image_format_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        parameters=[{
            'encoding_desired': 'rgb8',
            'image_width': MODEL_INPUT_SIZE,
            'image_height': MODEL_INPUT_SIZE,
        }],
        remappings=[('image_raw', 'padded_image'), ('image', 'image_rgb')],
    )
    image_to_tensor = ComposableNode(
        name='image_to_tensor_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{'scale': True, 'tensor_name': 'image'}],
        remappings=[('image', 'image_rgb'), ('tensor', 'normalized_tensor')],
    )
    interleaved_to_planar = ComposableNode(
        name='interleaved_to_planar_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[{
            'input_tensor_shape': [
                MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, MODEL_NUM_CHANNELS,
            ],
        }],
        remappings=[('interleaved_tensor', 'normalized_tensor')],
    )
    reshape = ComposableNode(
        name='reshape_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[{
            'output_tensor_name': 'images',
            'input_tensor_shape': [
                MODEL_NUM_CHANNELS, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE,
            ],
            'output_tensor_shape': [
                1, MODEL_NUM_CHANNELS, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE,
            ],
        }],
        remappings=[('tensor', 'planar_tensor')],
    )

    if transport == 'nitros':
        inference_nodes = [ComposableNode(
            name='onnx_inference',
            package='gpu_ros_onnx_inference',
            plugin='gpu_ros::onnx_inference::OnnxInferenceNode',
            parameters=[{
                'model_file_path': model_file_path,
                'execution_provider': execution_provider,
                'ort_profile_prefix': ort_profile_prefix,
                'ort_profile_frames': ort_profile_frames,
                'binding_report_path': binding_report_path,
                'transport': 'nitros',
            }],
            remappings=[
                ('tensor_input', 'reshaped_tensor'),
                ('tensor_output', 'tensor_sub'),
            ],
        )]
    else:
        inference_nodes = [
            ComposableNode(
                name='nitros_to_managed',
                package='gpu_ros_onnx_inference',
                plugin=(
                    'gpu_ros::onnx_inference::'
                    'NitrosToManagedTensorBundleNode'
                ),
                remappings=[
                    ('tensor_input', 'reshaped_tensor'),
                    ('tensor_output', 'managed_tensor_input'),
                ],
            ),
            ComposableNode(
                name='onnx_inference',
                package='gpu_ros_onnx_inference',
                plugin='gpu_ros::onnx_inference::OnnxInferenceNode',
                parameters=[{
                    'model_file_path': model_file_path,
                    'execution_provider': execution_provider,
                    'ort_profile_prefix': ort_profile_prefix,
                    'ort_profile_frames': ort_profile_frames,
                    'binding_report_path': binding_report_path,
                    'transport': 'managed',
                }],
                remappings=[
                    ('tensor_input', 'managed_tensor_input'),
                    ('tensor_output', 'managed_tensor_output'),
                ],
            ),
            ComposableNode(
                name='managed_to_nitros',
                package='gpu_ros_onnx_inference',
                plugin=(
                    'gpu_ros::onnx_inference::'
                    'ManagedToNitrosTensorBundleNode'
                ),
                remappings=[
                    ('tensor_input', 'managed_tensor_output'),
                    ('tensor_output', 'tensor_sub'),
                ],
            ),
        ]

    decoder = ComposableNode(
        name='yolov8_decoder',
        package='isaac_ros_yolov8',
        plugin='nvidia::isaac_ros::yolov8::YoloV8DecoderNode',
        parameters=[{
            'tensor_name': 'output0',
            'confidence_threshold': confidence_threshold,
            'nms_threshold': nms_threshold,
            'num_classes': 80,
        }],
    )

    container = ComposableNodeContainer(
        name='yolov8_container',
        namespace='yolov8_container',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            resize,
            pad,
            format_converter,
            image_to_tensor,
            interleaved_to_planar,
            reshape,
            *inference_nodes,
            decoder,
        ],
        output='screen',
    )
    return [container]


def generate_launch_description():
    """Generate the fixed-input YOLOv8 launch description."""
    arguments = [
        DeclareLaunchArgument('transport', default_value='nitros'),
        DeclareLaunchArgument('model_file_path', default_value=''),
        DeclareLaunchArgument('execution_provider', default_value='cuda'),
        DeclareLaunchArgument('ort_profile_prefix', default_value=''),
        DeclareLaunchArgument('ort_profile_frames', default_value='0'),
        DeclareLaunchArgument('binding_report_path', default_value=''),
        DeclareLaunchArgument('input_image_width', default_value='1280'),
        DeclareLaunchArgument('input_image_height', default_value='720'),
        DeclareLaunchArgument('confidence_threshold', default_value='0.25'),
        DeclareLaunchArgument('nms_threshold', default_value='0.45'),
    ]
    return launch.LaunchDescription(arguments + [OpaqueFunction(function=launch_setup)])
