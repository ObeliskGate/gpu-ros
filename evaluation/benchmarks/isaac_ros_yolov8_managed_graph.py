# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 Maintainer
# SPDX-License-Identifier: Apache-2.0
"""Managed YOLOv8: official NITROS pre/post-processing and ORT CUDA."""

import os
import sys

sys.path.append(os.path.dirname(__file__))
import yolov8_common as common  # noqa: E402

from launch_ros.actions import ComposableNodeContainer  # noqa: E402
from launch_ros.descriptions import ComposableNode  # noqa: E402
from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest  # noqa: E402


def launch_setup(container_prefix, container_sigterm_timeout):
    """Create the Managed YOLOv8 benchmark graph."""
    ns = TestIsaacROSYoloV8Managed.generate_namespace()
    nitros_to_managed = ComposableNode(
        name='NitrosToManaged',
        namespace=ns,
        package='isaac_ros_onnx_inference',
        plugin='nvidia::isaac_ros::onnx_inference::NitrosToManagedTensorListNode',
        parameters=[{'nitros_to_managed.enable_timing': True}],
        remappings=[
            ('tensor_input', 'reshaped_tensor'),
            ('tensor_output', 'managed_tensor_input'),
        ],
    )
    onnx = ComposableNode(
        name='OnnxInference',
        namespace=ns,
        package='isaac_ros_onnx_inference',
        plugin='nvidia::isaac_ros::onnx_inference::OnnxInferenceNode',
        parameters=[{
            'model_file_path': os.path.join(
                TestIsaacROSYoloV8Managed.get_assets_root_path(),
                'models', common.MODEL_FILE_NAME,
            ),
            'execution_provider': 'cuda',
            'transport': 'managed',
        }],
        remappings=[
            ('tensor_input', 'managed_tensor_input'),
            ('tensor_output', 'managed_tensor_output'),
        ],
    )
    managed_to_nitros = ComposableNode(
        name='ManagedToNitros',
        namespace=ns,
        package='isaac_ros_onnx_inference',
        plugin='nvidia::isaac_ros::onnx_inference::ManagedToNitrosTensorListNode',
        parameters=[{'managed_to_nitros.enable_timing': True}],
        remappings=[
            ('tensor_input', 'managed_tensor_output'),
            ('tensor_output', 'tensor_sub'),
        ],
    )
    decoder = ComposableNode(
        name='YoloV8Decoder',
        namespace=ns,
        package='isaac_ros_yolov8',
        plugin='nvidia::isaac_ros::yolov8::YoloV8DecoderNode',
        parameters=[{
            'tensor_name': common.ORT_OUTPUT_TENSOR_NAME,
            'confidence_threshold': 0.25,
            'nms_threshold': 0.45,
            'num_classes': 80,
        }],
    )
    container = ComposableNodeContainer(
        name='container',
        namespace=ns,
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            common.make_data_loader_node(ns),
            common.make_playback_node(ns),
            *common.make_preprocessing_nodes(ns, common.ORT_INPUT_TENSOR_NAME),
            nitros_to_managed,
            onnx,
            managed_to_nitros,
            decoder,
            common.make_monitor_node(ns),
        ],
        output='screen',
    )
    return [container]


def generate_test_description():
    """Generate the ros2_benchmark launch-test description."""
    return TestIsaacROSYoloV8Managed.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSYoloV8Managed(ROS2BenchmarkTest):
    """ORT CUDA Managed transport versus Config C NITROS transport."""

    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS YOLOv8 (M: ORT CUDA + Managed)',
        input_data_path=common.ROSBAG_PATH,
        publisher_upper_frequency=1000.0,
        publisher_lower_frequency=10.0,
        playback_message_buffer_size=1,
        pre_trial_run_wait_time_sec=5.0,
        custom_report_info={
            'data_resolution': common.IMAGE_RESOLUTION,
            'network_resolution': common.NETWORK_RESOLUTION,
            'build_type': 'Release',
            'inference_precision': 'FP32',
            'payload_copies': 0,
        },
    )

    def test_benchmark(self):
        """Run the configured benchmark matrix."""
        self.run_benchmark()

# Modified derived source; upstream NVIDIA Apache-2.0 attribution retained.
