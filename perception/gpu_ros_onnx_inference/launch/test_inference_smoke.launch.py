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

# Smoke test launch: starts OnnxInferenceNode with no model (logs warning and exits cleanly).
# Used to verify the node can be loaded as a component.
import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    node = ComposableNode(
        name='onnx_inference_node',
        package='gpu_ros_onnx_inference',
        plugin='gpu_ros::onnx_inference::OnnxInferenceNode',
        parameters=[{
            'model_file_path': '',
            'execution_provider': 'cpu',
            'gpu_device_id': 0,
        }],
    )
    container = ComposableNodeContainer(
        name='onnx_inference_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[node],
        output='screen',
    )
    return launch.LaunchDescription([container])
