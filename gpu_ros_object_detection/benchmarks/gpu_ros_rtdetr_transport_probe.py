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
"""Verify native TensorList/Managed component endpoints.

The C++ tests exercise a real payload and verify pointer identity.
"""

import time
import unittest

import launch
import launch_testing
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import pytest
import rclpy


@pytest.mark.launch_test
def generate_test_description():
    container = ComposableNodeContainer(
        name='managed_transport_probe',
        namespace='managed_transport_probe',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                name='nitros_to_managed',
                package='gpu_ros_onnx_inference',
                plugin='gpu_ros::onnx_inference::NitrosToManagedTensorBundleNode',
                parameters=[
                    {
                        'nitros_to_managed.enable_timing': True,
                        'nitros_to_managed.timing_log_every': 1,
                    }
                ],
                remappings=[
                    ('tensor_input', 'probe_nitros_input'),
                    ('tensor_output', 'probe_managed_output'),
                ],
            ),
            ComposableNode(
                name='managed_to_nitros',
                package='gpu_ros_onnx_inference',
                plugin='gpu_ros::onnx_inference::ManagedToNitrosTensorBundleNode',
                parameters=[
                    {
                        'managed_to_nitros.enable_timing': True,
                        'managed_to_nitros.timing_log_every': 1,
                    }
                ],
                remappings=[
                    ('tensor_input', 'probe_managed_output'),
                    ('tensor_output', 'probe_nitros_output'),
                ],
            ),
        ],
        output='screen',
    )
    return launch.LaunchDescription([container, launch_testing.actions.ReadyToTest()]), {
        'container': container,
    }


class TestManagedTransportProbe(unittest.TestCase):
    """Verify both native boundary components expose ready ROS endpoints.

    The adapter CTest exercises a real payload and verifies pointer identity.
    This launch probe independently catches component registration and graph
    endpoint startup regressions without depending on implementation log text.
    """

    @classmethod
    def setUpClass(cls):
        rclpy.init(args=[])
        cls.node = rclpy.create_node('native_transport_probe_client')

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_native_transport_endpoints_are_ready(self):
        expected_endpoints = (
            ('/probe_nitros_input', 'subscription'),
            ('/probe_managed_output', 'publisher'),
            ('/probe_managed_output', 'subscription'),
            ('/probe_nitros_output', 'publisher'),
        )
        deadline = time.monotonic() + 30.0
        missing = list(expected_endpoints)
        while missing and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            missing = []
            topics = [name for name, _types in self.node.get_topic_names_and_types()]
            for topic_suffix, direction in expected_endpoints:
                matching_topics = [topic for topic in topics if topic.endswith(topic_suffix)]
                if direction == 'publisher':
                    is_ready = any(
                        self.node.get_publishers_info_by_topic(topic) for topic in matching_topics
                    )
                else:
                    is_ready = any(
                        self.node.get_subscriptions_info_by_topic(topic)
                        for topic in matching_topics
                    )
                if not is_ready:
                    missing.append((topic_suffix, direction))
        self.assertEqual(
            missing,
            [],
            f'native boundary endpoints did not become ready: {missing}',
        )
