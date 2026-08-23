# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Modified to compare the managed transport around the official NVIDIA graph
# in 2026;
# see THIRD_PARTY_NOTICES.md for the external source boundary.
# SPDX-License-Identifier: Apache-2.0
"""Managed RT-DETR: official NITROS pre/post-processing and ORT CUDA."""

import os
import sys

sys.path.append(os.path.dirname(__file__))
import rtdetr_common as common  # noqa: E402

from launch_ros.actions import ComposableNodeContainer  # noqa: E402
from launch_ros.descriptions import ComposableNode  # noqa: E402
from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest  # noqa: E402


def launch_setup(container_prefix, container_sigterm_timeout):
    ns = TestGpuRosRtDetrManaged.generate_namespace()
    preprocessor = ComposableNode(
        name='RtdetrPreprocessor', namespace=ns, package='isaac_ros_rtdetr',
        plugin='nvidia::isaac_ros::rtdetr::RtDetrPreprocessorNode',
        parameters=[{'image_size': common.NETWORK_RESOLUTION['width']}],
        remappings=[('encoded_tensor', 'reshaped_tensor')])
    nitros_to_managed = ComposableNode(
        name='NitrosToManaged', namespace=ns, package='gpu_ros_onnx_inference',
        plugin='gpu_ros::onnx_inference::NitrosToManagedTensorBundleNode',
        parameters=[{'nitros_to_managed.enable_timing': True}],
        remappings=[('tensor_input', 'tensor_pub'), ('tensor_output', 'managed_tensor_input')])
    onnx = ComposableNode(
        name='OnnxInference', namespace=ns, package='gpu_ros_onnx_inference',
        plugin='gpu_ros::onnx_inference::OnnxInferenceNode',
        parameters=[{
            'model_file_path': os.path.join(
                TestGpuRosRtDetrManaged.get_assets_root_path(), 'models', common.MODEL_FILE_NAME),
            'execution_provider': 'cuda', 'transport': 'managed'}],
        remappings=[('tensor_input', 'managed_tensor_input'),
                    ('tensor_output', 'managed_tensor_output')])
    managed_to_nitros = ComposableNode(
        name='ManagedToNitros', namespace=ns, package='gpu_ros_onnx_inference',
        plugin='gpu_ros::onnx_inference::ManagedToNitrosTensorBundleNode',
        parameters=[{'managed_to_nitros.enable_timing': True}],
        remappings=[('tensor_input', 'managed_tensor_output'), ('tensor_output', 'tensor_sub')])
    decoder = ComposableNode(
        name='RtdetrDecoder', namespace=ns, package='isaac_ros_rtdetr',
        plugin='nvidia::isaac_ros::rtdetr::RtDetrDecoderNode')
    container = ComposableNodeContainer(
        name='container', namespace=ns, package='rclcpp_components',
        executable='component_container_mt', prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            common.make_data_loader_node(ns), common.make_playback_node(ns),
            *common.make_preprocessing_nodes(ns), preprocessor, nitros_to_managed,
            onnx, managed_to_nitros, decoder, common.make_monitor_node(ns)],
        output='screen')
    return [container]


def generate_test_description():
    return TestGpuRosRtDetrManaged.generate_test_description_with_nsys(launch_setup)


class TestGpuRosRtDetrManaged(ROS2BenchmarkTest):
    """ORT CUDA transport=managed versus Config C NITROS transport."""

    config = ROS2BenchmarkConfig(
        benchmark_name='GPU ROS RT-DETR (NVIDIA reference M: ORT CUDA + Managed)',
        input_data_path=common.ROSBAG_PATH,
        publisher_upper_frequency=1000.0,
        publisher_lower_frequency=10.0,
        playback_message_buffer_size=1,
        pre_trial_run_wait_time_sec=5.0,
        custom_report_info={
            'data_resolution': common.IMAGE_RESOLUTION,
            'network_resolution': common.NETWORK_RESOLUTION,
            'build_type': 'Release',
            'payload_copies': 0,
        })

    def test_benchmark(self):
        self.run_benchmark()
