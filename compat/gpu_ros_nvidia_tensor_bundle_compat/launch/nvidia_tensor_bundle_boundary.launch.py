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

"""Explicit NVIDIA-only TensorList/TensorBundle boundary smoke path.

The two adapters are intentionally visible here.  NVIDIA reference graphs can
include this boundary where an official TensorList publisher or subscriber is
present; AMD profiles must not build or install this package.
"""

from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    to_nvidia = ComposableNode(
        package='gpu_ros_nvidia_tensor_bundle_compat',
        plugin=(
            'gpu_ros::nvidia_tensor_bundle_compat::'
            'TensorBundleToNvidiaTensorListNode'),
        name='tensor_bundle_to_nvidia_tensor_list',
        remappings=[
            ('tensor_input', 'tensor_bundle_input'),
            ('tensor_output', 'nvidia_tensor_list_output'),
        ],
    )
    from_nvidia = ComposableNode(
        package='gpu_ros_nvidia_tensor_bundle_compat',
        plugin=(
            'gpu_ros::nvidia_tensor_bundle_compat::'
            'NvidiaTensorListToTensorBundleNode'),
        name='nvidia_tensor_list_to_tensor_bundle',
        remappings=[
            ('tensor_input', 'nvidia_tensor_list_output'),
            ('tensor_output', 'tensor_bundle_output'),
        ],
    )
    container = ComposableNodeContainer(
        name='tensor_bundle_boundary_container',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[to_nvidia, from_nvidia],
        output='screen',
    )
    return LaunchDescription([container])
