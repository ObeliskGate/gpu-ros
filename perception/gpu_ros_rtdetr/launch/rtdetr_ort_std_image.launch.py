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
RT-DETR Phase 2a target: std image encoder + ORT std transport.

Pipeline:
Image -> RtDetrImageEncoderNode -> RtDetrPreprocessorNode ->
OnnxInferenceNode(transport=std) -> RtDetrDecoderNode -> Detection2DArray.
"""

import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue

MODEL_INPUT_SIZE = 640


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'model_file_path',
            default_value='',
            description='Absolute path to the RT-DETR ONNX file'),
        DeclareLaunchArgument(
            'image_topic',
            default_value='image',
            description='Input sensor_msgs/Image topic, relative or absolute'),
        DeclareLaunchArgument(
            'namespace',
            default_value='rtdetr',
            description='Namespace for the composable pipeline'),
        DeclareLaunchArgument(
            'input_image_width',
            default_value='640',
            description='Width seed used by RtDetrPreprocessor for orig_target_sizes'),
        DeclareLaunchArgument(
            'input_image_height',
            default_value='640',
            description='Height seed used by RtDetrPreprocessor for orig_target_sizes'),
        DeclareLaunchArgument(
            'use_max_dim_for_orig_size',
            default_value='false',
            description=(
                'Use a square max-dimension orig_target_sizes value. The image-input '
                'pipeline defaults false to preserve the source image aspect ratio.')),
        DeclareLaunchArgument(
            'execution_provider',
            default_value='migraphx',
            description='ONNX Runtime execution provider (migraphx/rocm/cuda/cpu)'),
        DeclareLaunchArgument(
            'gpu_device_id',
            default_value='0',
            description='GPU device id for providers that support it'),
        DeclareLaunchArgument(
            'ort_profile_prefix',
            default_value='',
            description='Enable ORT profiling and write JSON using this path prefix'),
        DeclareLaunchArgument(
            'binding_report_path',
            default_value='',
            description='Write the first-frame ORT pointer/lifetime report here'),
        DeclareLaunchArgument(
            'confidence_threshold',
            default_value='0.6',
            description='Minimum score for a bounding box to be published'),
    ]

    model_file_path = LaunchConfiguration('model_file_path')
    image_topic = LaunchConfiguration('image_topic')
    namespace = LaunchConfiguration('namespace')
    input_image_width = LaunchConfiguration('input_image_width')
    input_image_height = LaunchConfiguration('input_image_height')
    use_max_dim_for_orig_size = LaunchConfiguration('use_max_dim_for_orig_size')
    execution_provider = LaunchConfiguration('execution_provider')
    gpu_device_id = LaunchConfiguration('gpu_device_id')
    ort_profile_prefix = LaunchConfiguration('ort_profile_prefix')
    binding_report_path = LaunchConfiguration('binding_report_path')
    confidence_threshold = LaunchConfiguration('confidence_threshold')

    image_encoder_node = ComposableNode(
        name='rtdetr_image_encoder',
        package='gpu_ros_rtdetr',
        plugin='gpu_ros::rtdetr::RtDetrImageEncoderNode',
        parameters=[{
            'tensor_name': 'input_tensor',
            'output_width': MODEL_INPUT_SIZE,
            'output_height': MODEL_INPUT_SIZE,
        }],
        remappings=[('image', image_topic)],
    )

    rtdetr_preprocessor_node = ComposableNode(
        name='rtdetr_preprocessor',
        package='gpu_ros_rtdetr',
        plugin='gpu_ros::rtdetr::RtDetrPreprocessorNode',
        parameters=[{
            'image_width': input_image_width,
            'image_height': input_image_height,
            'use_max_dim_for_orig_size': ParameterValue(
                use_max_dim_for_orig_size, value_type=bool),
        }],
    )

    onnx_node = ComposableNode(
        name='onnx_inference',
        package='gpu_ros_onnx_inference',
        plugin='gpu_ros::onnx_inference::OnnxInferenceNode',
        parameters=[{
            'model_file_path': model_file_path,
            'execution_provider': execution_provider,
            'gpu_device_id': gpu_device_id,
            'ort_profile_prefix': ort_profile_prefix,
            'binding_report_path': binding_report_path,
            'transport': 'std',
        }],
        remappings=[
            ('tensor_input', 'tensor_pub'),
            ('tensor_output', 'tensor_sub'),
        ]
    )

    rtdetr_decoder_node = ComposableNode(
        name='rtdetr_decoder',
        package='gpu_ros_rtdetr',
        plugin='gpu_ros::rtdetr::RtDetrDecoderNode',
        parameters=[{'confidence_threshold': confidence_threshold}],
    )

    container = ComposableNodeContainer(
        name='rtdetr_phase2a_container',
        namespace=namespace,
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            image_encoder_node,
            rtdetr_preprocessor_node,
            onnx_node,
            rtdetr_decoder_node,
        ],
        output='screen'
    )

    return launch.LaunchDescription(launch_args + [container])
