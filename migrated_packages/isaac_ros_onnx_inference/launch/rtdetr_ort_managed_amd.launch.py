# Copyright 2026 Maintainer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""RT-DETR standard host graph with explicit Managed HIP staging."""

import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue


MODEL_INPUT_SIZE = 640


def generate_launch_description():
    """Generate the RT-DETR Managed HIP graph."""
    model_file_path = LaunchConfiguration('model_file_path')
    image_topic = LaunchConfiguration('image_topic')
    namespace = LaunchConfiguration('namespace')
    input_image_width = LaunchConfiguration('input_image_width')
    input_image_height = LaunchConfiguration('input_image_height')
    use_max_dim_for_orig_size = LaunchConfiguration('use_max_dim_for_orig_size')
    gpu_device_id = LaunchConfiguration('gpu_device_id')
    ort_profile_prefix = LaunchConfiguration('ort_profile_prefix')
    confidence_threshold = LaunchConfiguration('confidence_threshold')

    encoder = ComposableNode(
        name='rtdetr_image_encoder',
        package='isaac_ros_rtdetr_std',
        plugin='nvidia::isaac_ros::rtdetr_std::RtDetrImageEncoderNode',
        parameters=[{
            'tensor_name': 'input_tensor',
            'output_width': MODEL_INPUT_SIZE,
            'output_height': MODEL_INPUT_SIZE,
        }],
        remappings=[('image', image_topic)],
    )
    preprocessor = ComposableNode(
        name='rtdetr_preprocessor',
        package='isaac_ros_rtdetr_std',
        plugin='nvidia::isaac_ros::rtdetr_std::RtDetrPreprocessorNode',
        parameters=[{
            'image_width': input_image_width,
            'image_height': input_image_height,
            'use_max_dim_for_orig_size': ParameterValue(
                use_max_dim_for_orig_size, value_type=bool),
        }],
    )
    std_to_managed = ComposableNode(
        name='std_to_managed_hip',
        package='isaac_ros_onnx_inference',
        plugin=(
            'nvidia::isaac_ros::onnx_inference::'
            'StdToManagedHipTensorListNode'),
        parameters=[{'gpu_device_id': gpu_device_id}],
        remappings=[
            ('tensor_input', 'tensor_pub'),
            ('tensor_output', 'managed_tensor_input'),
        ],
    )
    onnx = ComposableNode(
        name='onnx_inference',
        package='isaac_ros_onnx_inference',
        plugin='nvidia::isaac_ros::onnx_inference::OnnxInferenceNode',
        parameters=[{
            'model_file_path': model_file_path,
            'execution_provider': 'migraphx',
            'gpu_device_id': gpu_device_id,
            'ort_profile_prefix': ort_profile_prefix,
            'transport': 'managed',
        }],
        remappings=[
            ('tensor_input', 'managed_tensor_input'),
            ('tensor_output', 'managed_tensor_output'),
        ],
    )
    managed_to_std = ComposableNode(
        name='managed_hip_to_std',
        package='isaac_ros_onnx_inference',
        plugin=(
            'nvidia::isaac_ros::onnx_inference::'
            'ManagedHipToStdTensorListNode'),
        parameters=[{'gpu_device_id': gpu_device_id}],
        remappings=[
            ('tensor_input', 'managed_tensor_output'),
            ('tensor_output', 'tensor_sub'),
        ],
    )
    decoder = ComposableNode(
        name='rtdetr_decoder',
        package='isaac_ros_rtdetr_std',
        plugin='nvidia::isaac_ros::rtdetr_std::RtDetrDecoderNode',
        parameters=[{'confidence_threshold': confidence_threshold}],
    )

    container = ComposableNodeContainer(
        name='rtdetr_managed_amd_container',
        namespace=namespace,
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            encoder, preprocessor, std_to_managed, onnx, managed_to_std, decoder],
        output='screen',
    )

    arguments = [
        DeclareLaunchArgument('model_file_path', default_value=''),
        DeclareLaunchArgument('image_topic', default_value='image'),
        DeclareLaunchArgument('namespace', default_value='rtdetr_managed'),
        DeclareLaunchArgument('input_image_width', default_value='640'),
        DeclareLaunchArgument('input_image_height', default_value='640'),
        DeclareLaunchArgument('use_max_dim_for_orig_size', default_value='false'),
        DeclareLaunchArgument('gpu_device_id', default_value='0'),
        DeclareLaunchArgument('ort_profile_prefix', default_value=''),
        DeclareLaunchArgument('confidence_threshold', default_value='0.6'),
    ]
    return launch.LaunchDescription(arguments + [container])
