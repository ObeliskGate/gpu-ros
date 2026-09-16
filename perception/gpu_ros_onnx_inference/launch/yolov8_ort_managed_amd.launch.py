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
"""Production YOLOv8 direct Managed HIP graph."""

import os
from pathlib import Path

import launch
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


MODEL_INPUT_SIZE = 640
DEFAULT_MODEL_PATH = os.path.join(
    os.environ.get('OVG_ASSETS_ROOT', '/workspaces/ovg-assets'), 'models', 'yolov8', 'yolov8s.onnx'
)


def _launch_setup(context):
    model_file_path = str(
        Path(LaunchConfiguration('model_file_path').perform(context))
        .expanduser()
        .resolve(strict=False)
    )
    if not os.path.isfile(model_file_path) or os.path.getsize(model_file_path) == 0:
        raise RuntimeError(
            f'YOLOv8 ONNX asset is missing: {model_file_path}. '
            'Provide the user-owned Phase 2A asset.'
        )

    image_topic = LaunchConfiguration('image_topic').perform(context)
    namespace = LaunchConfiguration('namespace').perform(context)
    gpu_device_id = int(LaunchConfiguration('gpu_device_id').perform(context))
    ort_profile_prefix = LaunchConfiguration('ort_profile_prefix').perform(context)
    binding_report_path = LaunchConfiguration('binding_report_path').perform(context)
    confidence_threshold = float(LaunchConfiguration('confidence_threshold').perform(context))
    nms_threshold = float(LaunchConfiguration('nms_threshold').perform(context))

    encoder = ComposableNode(
        name='yolov8_managed_hip_image_encoder',
        package='gpu_ros_yolov8',
        plugin=('gpu_ros::yolov8::YoloV8ManagedHipImageEncoderNode'),
        parameters=[
            {
                'tensor_name': 'images',
                'output_width': MODEL_INPUT_SIZE,
                'output_height': MODEL_INPUT_SIZE,
                'gpu_device_id': gpu_device_id,
                'managed_pool_capacity': 16,
                'managed_pool_wait_timeout_ms': 100,
            }
        ],
        remappings=[('image', image_topic)],
    )
    onnx = ComposableNode(
        name='onnx_inference',
        package='gpu_ros_onnx_inference',
        plugin='gpu_ros::onnx_inference::OnnxInferenceNode',
        parameters=[
            {
                'model_file_path': model_file_path,
                'execution_provider': 'migraphx',
                'gpu_device_id': gpu_device_id,
                'ort_profile_prefix': ort_profile_prefix,
                'binding_report_path': binding_report_path,
                'transport': 'managed',
                'managed_io_contract': 'hip_managed_strict',
                'managed_input_contracts': ['images=float32[1,3,640,640]'],
                'managed_output_contracts': ['output0=float32[1,84,8400]'],
                'managed_pool_capacity': 16,
                'managed_pool_wait_timeout_ms': 100,
            }
        ],
        remappings=[
            ('tensor_input', 'managed_tensor_output'),
            ('tensor_output', 'managed_tensor_output_ort'),
        ],
    )
    decoder = ComposableNode(
        name='yolov8_managed_hip_decoder',
        package='gpu_ros_yolov8',
        plugin=('gpu_ros::yolov8::YoloV8ManagedHipDecoderNode'),
        parameters=[
            {
                'gpu_device_id': gpu_device_id,
                'tensor_name': 'output0',
                'confidence_threshold': confidence_threshold,
                'nms_threshold': nms_threshold,
                'num_classes': 80,
            }
        ],
        remappings=[('managed_tensor_input', 'managed_tensor_output_ort')],
    )

    container = ComposableNodeContainer(
        name='yolov8_managed_amd_container',
        namespace=namespace,
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[encoder, onnx, decoder],
        output='screen',
    )
    return [container]


def generate_launch_description():
    """Generate the YOLOv8 direct Managed HIP graph."""
    arguments = [
        DeclareLaunchArgument('model_file_path', default_value=DEFAULT_MODEL_PATH),
        DeclareLaunchArgument('image_topic', default_value='image'),
        DeclareLaunchArgument('namespace', default_value='yolov8_managed'),
        DeclareLaunchArgument('gpu_device_id', default_value='0'),
        DeclareLaunchArgument('ort_profile_prefix', default_value=''),
        DeclareLaunchArgument('binding_report_path', default_value=''),
        DeclareLaunchArgument('confidence_threshold', default_value='0.25'),
        DeclareLaunchArgument('nms_threshold', default_value='0.45'),
    ]
    return launch.LaunchDescription(arguments + [OpaqueFunction(function=_launch_setup)])
