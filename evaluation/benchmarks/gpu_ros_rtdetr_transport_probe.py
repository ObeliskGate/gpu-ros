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
"""Load both Managed boundaries; C++ adapter tests verify pointer identity."""

import unittest

import launch
import launch_testing
import pytest
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


@pytest.mark.launch_test
def generate_test_description():
    container = ComposableNodeContainer(
        name='managed_transport_probe',
        namespace='managed_transport_probe',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                name='nitros_to_managed', package='gpu_ros_onnx_inference',
                plugin='gpu_ros::onnx_inference::NitrosToManagedTensorBundleNode',
                parameters=[{'nitros_to_managed.enable_timing': True,
                             'nitros_to_managed.timing_log_every': 1}],
                remappings=[('tensor_input', 'probe_nitros_input'),
                            ('tensor_output', 'probe_managed_output')]),
            ComposableNode(
                name='managed_to_nitros', package='gpu_ros_onnx_inference',
                plugin='gpu_ros::onnx_inference::ManagedToNitrosTensorBundleNode',
                parameters=[{'managed_to_nitros.enable_timing': True,
                             'managed_to_nitros.timing_log_every': 1}],
                remappings=[('tensor_input', 'probe_managed_output'),
                            ('tensor_output', 'probe_nitros_output')]),
        ],
        output='screen')
    return launch.LaunchDescription([container, launch_testing.actions.ReadyToTest()]), {
        'container': container,
    }


class TestManagedTransportProbe(unittest.TestCase):
    """Ensure both components load in one CUDA/NITROS process.

    `test_nitros_managed_tensor_bundle_adapter` exercises a real payload and
    asserts NITROS -> Managed -> NITROS pointer identity. This launch probe
    catches component registration and startup regressions separately.
    """

    def test_components_start(self, proc_output, container):
        proc_output.assertWaitFor(
            'Starting Managed Nitros Publisher', process=container, timeout=30)
