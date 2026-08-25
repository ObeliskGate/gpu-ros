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
"""Production RT-DETR direct Managed HIP graph."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue


MODEL_INPUT_SIZE = 640


def generate_launch_description():
    """Generate the RT-DETR direct Managed HIP graph."""
    model_file_path = LaunchConfiguration('model_file_path')
    model_profile = LaunchConfiguration('model_profile')
    model_assets_root = LaunchConfiguration('model_assets_root')
    image_topic = LaunchConfiguration('image_topic')
    namespace = LaunchConfiguration('namespace')
    input_image_width = LaunchConfiguration('input_image_width')
    input_image_height = LaunchConfiguration('input_image_height')
    use_max_dim_for_orig_size = LaunchConfiguration('use_max_dim_for_orig_size')
    gpu_device_id = LaunchConfiguration('gpu_device_id')
    ort_profile_prefix = LaunchConfiguration('ort_profile_prefix')
    binding_report_path = LaunchConfiguration('binding_report_path')
    confidence_threshold = LaunchConfiguration('confidence_threshold')

    encoder = ComposableNode(
        name='rtdetr_managed_hip_image_encoder',
        package='gpu_ros_rtdetr',
        plugin=(
            'gpu_ros::rtdetr::'
            'RtDetrManagedHipImageEncoderNode'),
        parameters=[{
            'tensor_name': 'input_tensor',
            'output_width': MODEL_INPUT_SIZE,
            'output_height': MODEL_INPUT_SIZE,
            'gpu_device_id': ParameterValue(gpu_device_id, value_type=int),
            'managed_pool_capacity': 16,
            'managed_pool_wait_timeout_ms': 100,
        }],
        remappings=[
            ('image', image_topic),
            ('managed_tensor_output', 'managed_tensor_image'),
        ],
    )
    preprocessor = ComposableNode(
        name='rtdetr_managed_hip_preprocessor',
        package='gpu_ros_rtdetr',
        plugin=(
            'gpu_ros::rtdetr::'
            'RtDetrManagedHipPreprocessorNode'),
        parameters=[{
            'input_image_tensor_name': 'input_tensor',
            'output_image_tensor_name': 'images',
            'output_size_tensor_name': 'orig_target_sizes',
            'image_width': ParameterValue(input_image_width, value_type=int),
            'image_height': ParameterValue(input_image_height, value_type=int),
            'model_input_width': MODEL_INPUT_SIZE,
            'model_input_height': MODEL_INPUT_SIZE,
            'use_max_dim_for_orig_size': ParameterValue(
                use_max_dim_for_orig_size, value_type=bool),
            'gpu_device_id': ParameterValue(gpu_device_id, value_type=int),
            'managed_pool_capacity': 16,
            'managed_pool_wait_timeout_ms': 100,
        }],
        remappings=[('managed_tensor_input', 'managed_tensor_image')],
    )
    onnx = ComposableNode(
        name='onnx_inference',
        package='gpu_ros_onnx_inference',
        plugin='gpu_ros::onnx_inference::OnnxInferenceNode',
        parameters=[{
            'model_file_path': model_file_path,
            'model_profile': model_profile,
            'model_assets_root': model_assets_root,
            'execution_provider': 'migraphx',
            'gpu_device_id': ParameterValue(gpu_device_id, value_type=int),
            'ort_profile_prefix': ort_profile_prefix,
            'binding_report_path': binding_report_path,
            'transport': 'managed',
            'managed_io_contract': 'hip_managed_strict',
            'managed_input_contracts': [
                'images=float32[1,3,640,640]',
                'orig_target_sizes=int64[1,2]',
            ],
            'managed_output_contracts': [
                'labels=int64[1,300]',
                'boxes=float32[1,300,4]',
                'scores=float32[1,300]',
            ],
            'managed_pool_capacity': 16,
            'managed_pool_wait_timeout_ms': 100,
        }],
        remappings=[
            ('tensor_input', 'managed_tensor_output'),
            ('tensor_output', 'managed_tensor_output_ort'),
        ],
    )
    decoder = ComposableNode(
        name='rtdetr_managed_hip_decoder',
        package='gpu_ros_rtdetr',
        plugin=(
            'gpu_ros::rtdetr::'
            'RtDetrManagedHipDecoderNode'),
        parameters=[{
            'gpu_device_id': ParameterValue(gpu_device_id, value_type=int),
            'confidence_threshold': ParameterValue(
                confidence_threshold, value_type=float),
        }],
        remappings=[('managed_tensor_input', 'managed_tensor_output_ort')],
    )

    container = ComposableNodeContainer(
        name='rtdetr_managed_amd_container',
        namespace=namespace,
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[encoder, preprocessor, onnx, decoder],
        output='screen',
    )

    arguments = [
        DeclareLaunchArgument('model_file_path', default_value=''),
        DeclareLaunchArgument('model_profile', default_value='auto'),
        DeclareLaunchArgument(
            'model_assets_root', default_value='/workspaces/ovg-assets'),
        DeclareLaunchArgument('image_topic', default_value='image'),
        DeclareLaunchArgument('namespace', default_value='rtdetr_managed'),
        DeclareLaunchArgument('input_image_width', default_value='640'),
        DeclareLaunchArgument('input_image_height', default_value='640'),
        DeclareLaunchArgument('use_max_dim_for_orig_size', default_value='false'),
        DeclareLaunchArgument('gpu_device_id', default_value='0'),
        DeclareLaunchArgument('ort_profile_prefix', default_value=''),
        DeclareLaunchArgument('binding_report_path', default_value=''),
        DeclareLaunchArgument('confidence_threshold', default_value='0.6'),
    ]
    return LaunchDescription(arguments + [container])
