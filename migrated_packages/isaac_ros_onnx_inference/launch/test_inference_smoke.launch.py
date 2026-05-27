# Copyright 2026 - Apache-2.0
# Smoke test launch: starts OnnxInferenceNode with no model (logs warning and exits cleanly).
# Used to verify the node can be loaded as a component.
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import launch


def generate_launch_description():
    node = ComposableNode(
        name='onnx_inference_node',
        package='isaac_ros_onnx_inference',
        plugin='nvidia::isaac_ros::onnx_inference::OnnxInferenceNode',
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
