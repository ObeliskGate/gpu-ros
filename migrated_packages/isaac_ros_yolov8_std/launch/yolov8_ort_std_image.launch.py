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

"""YOLOv8 standard ROS2 image path for the AMD Phase 2a target."""

import os
from pathlib import Path

import launch
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


MODEL_INPUT_SIZE = 640
DEFAULT_MODEL_PATH = os.path.join(
    os.environ.get('OVG_ASSETS_ROOT', '/workspaces/ovg-assets'),
    'models',
    'yolov8',
    'yolov8s.onnx',
)


def _missing_model_error(model_path: str) -> RuntimeError:
    return RuntimeError(
        'YOLOv8 ONNX asset is missing:\n'
        f'{model_path}\n'
        'Provide OVG_YOLOV8_ONNX_SOURCE and run phase2 assets import-yolov8.'
    )


def _launch_setup(context):
    model_file_path = LaunchConfiguration('model_file_path').perform(context)
    resolved_model_path = str(Path(model_file_path).expanduser().resolve(strict=False))
    if not os.path.isfile(resolved_model_path) or os.path.getsize(resolved_model_path) == 0:
        raise _missing_model_error(resolved_model_path)

    image_topic = LaunchConfiguration('image_topic').perform(context)
    namespace = LaunchConfiguration('namespace').perform(context)
    execution_provider = LaunchConfiguration('execution_provider').perform(context)
    ort_profile_prefix = LaunchConfiguration('ort_profile_prefix').perform(context)
    binding_report_path = LaunchConfiguration('binding_report_path').perform(context)
    confidence_threshold = float(
        LaunchConfiguration('confidence_threshold').perform(context))
    nms_threshold = float(
        LaunchConfiguration('nms_threshold').perform(context))

    image_encoder_node = ComposableNode(
        name='yolov8_image_encoder',
        package='isaac_ros_yolov8_std',
        plugin='nvidia::isaac_ros::yolov8_std::YoloV8ImageEncoderNode',
        parameters=[{
            'tensor_name': 'images',
            'output_width': MODEL_INPUT_SIZE,
            'output_height': MODEL_INPUT_SIZE,
        }],
        remappings=[('image', image_topic)],
    )

    onnx_node = ComposableNode(
        name='onnx_inference',
        package='isaac_ros_onnx_inference',
        plugin='nvidia::isaac_ros::onnx_inference::OnnxInferenceNode',
        parameters=[{
            'model_file_path': resolved_model_path,
            'execution_provider': execution_provider,
            'ort_profile_prefix': ort_profile_prefix,
            'binding_report_path': binding_report_path,
            'transport': 'std',
        }],
        remappings=[
            ('tensor_input', 'encoded_tensor'),
            ('tensor_output', 'tensor_sub'),
        ],
    )

    decoder_node = ComposableNode(
        name='yolov8_decoder',
        package='isaac_ros_yolov8_std',
        plugin='nvidia::isaac_ros::yolov8_std::YoloV8DecoderNode',
        parameters=[{
            'tensor_name': 'output0',
            'confidence_threshold': confidence_threshold,
            'nms_threshold': nms_threshold,
            'num_classes': 80,
        }],
    )

    container = ComposableNodeContainer(
        name='yolov8_phase2a_container',
        namespace=namespace,
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[image_encoder_node, onnx_node, decoder_node],
        output='screen',
    )
    return [container]


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'model_file_path',
            default_value=DEFAULT_MODEL_PATH,
            description='Absolute path to the user-provided YOLOv8s ONNX file',
        ),
        DeclareLaunchArgument(
            'image_topic',
            default_value='image',
            description='Input sensor_msgs/Image topic, relative or absolute',
        ),
        DeclareLaunchArgument(
            'namespace',
            default_value='yolov8',
            description='Namespace for the composable pipeline',
        ),
        DeclareLaunchArgument(
            'execution_provider',
            default_value='migraphx',
            description='ONNX Runtime execution provider (migraphx/rocm/cpu)',
        ),
        DeclareLaunchArgument(
            'ort_profile_prefix',
            default_value='',
            description='Enable ORT profiling and write JSON using this path prefix',
        ),
        DeclareLaunchArgument(
            'binding_report_path',
            default_value='',
            description='Write the first-frame ORT pointer/lifetime report here',
        ),
        DeclareLaunchArgument(
            'confidence_threshold',
            default_value='0.25',
            description='Minimum score for a bounding box to be published',
        ),
        DeclareLaunchArgument(
            'nms_threshold',
            default_value='0.45',
            description='IoU threshold used by YOLOv8 decoder NMS',
        ),
    ]
    return launch.LaunchDescription(launch_args + [OpaqueFunction(function=_launch_setup)])
