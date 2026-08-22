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

"""
Proof-Of-Life test for OnnxInferenceNode in the NITROS transport path (config C).

Forks the upstream isaac_ros_rtdetr POL test, replacing only the TensorRTNode
with our OnnxInferenceNode (transport=nitros). The 6 upstream NITROS preprocess
nodes + NITROS RtDetrPreprocessor + NITROS RtDetrDecoder are reused as-is.

Verifies the ORT node can drop into a real RT-DETR NITROS graph and produce a
Detection2DArray. Uses a random-weight RT-DETR-shaped ONNX (data not checked).
"""

import os
import pathlib
import time

from isaac_ros_test import IsaacROSBaseTest, JSONConversion, MockModelGenerator
from launch_ros.actions.composable_node_container import ComposableNodeContainer
from launch_ros.descriptions.composable_node import ComposableNode
import pytest
import rclpy
from sensor_msgs.msg import CameraInfo, Image
import torch
from vision_msgs.msg import Detection2DArray


MODEL_ONNX_PATH = '/tmp/rtdetr_ort_pol_model.onnx'
MODEL_GENERATION_TIMEOUT_SEC = 300
INIT_WAIT_SEC = 10


@pytest.mark.rostest
def generate_rtdetr_pol_description(test_class, transport):
    """Generate the shared C or Managed RT-DETR proof-of-life graph."""
    MockModelGenerator.generate(
        input_bindings=[
            MockModelGenerator.Binding('images', [-1, 3, 640, 640], torch.float32),
            MockModelGenerator.Binding('orig_target_sizes', [-1, 2], torch.int64)
        ],
        output_bindings=[
            MockModelGenerator.Binding('labels', [-1, 300], torch.int64),
            MockModelGenerator.Binding('boxes', [-1, 300, 4], torch.float32),
            MockModelGenerator.Binding('scores', [-1, 300], torch.float32)
        ],
        output_onnx_path=MODEL_ONNX_PATH
    )

    ns = test_class.generate_namespace()

    resize_node = ComposableNode(
        name='resize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        namespace=ns,
        parameters=[{
            'input_width': 640,
            'input_height': 480,
            'output_width': 640,
            'output_height': 640,
            'keep_aspect_ratio': True,
            'encoding_desired': 'rgb8',
            'disable_padding': True
        }]
    )

    pad_node = ComposableNode(
        name='pad_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::PadNode',
        namespace=ns,
        parameters=[{
            'output_image_width': 640,
            'output_image_height': 640,
            'padding_type': 'BOTTOM_RIGHT'
        }],
        remappings=[('image', 'resize/image')]
    )

    image_format_node = ComposableNode(
        name='image_format_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        namespace=ns,
        parameters=[{
            'encoding_desired': 'rgb8',
            'image_width': 640,
            'image_height': 640
        }],
        remappings=[('image_raw', 'padded_image'), ('image', 'image_rgb')]
    )

    image_to_tensor_node = ComposableNode(
        name='image_to_tensor_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        namespace=ns,
        parameters=[{'scale': False, 'tensor_name': 'image'}],
        remappings=[('image', 'image_rgb'), ('tensor', 'normalized_tensor')]
    )

    interleave_to_planar_node = ComposableNode(
        name='interleaved_to_planar_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        namespace=ns,
        parameters=[{'input_tensor_shape': [640, 640, 3]}],
        remappings=[('interleaved_tensor', 'normalized_tensor')]
    )

    reshape_node = ComposableNode(
        name='reshape_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        namespace=ns,
        parameters=[{
            'output_tensor_name': 'input_tensor',
            'input_tensor_shape': [3, 640, 640],
            'output_tensor_shape': [1, 3, 640, 640]
        }],
        remappings=[('tensor', 'planar_tensor')]
    )

    rtdetr_preprocessor_node = ComposableNode(
        name='rtdetr_preprocessor',
        package='isaac_ros_rtdetr',
        plugin='nvidia::isaac_ros::rtdetr::RtDetrPreprocessorNode',
        namespace=ns,
        remappings=[('encoded_tensor', 'reshaped_tensor')]
    )

    # Config C replaces TensorRT directly. Managed keeps the official NITROS
    # preprocessor/decoder and inserts explicit zero-payload-copy boundaries.
    onnx_node = ComposableNode(
        name='onnx_inference',
        package='isaac_ros_onnx_inference',
        plugin='nvidia::isaac_ros::onnx_inference::OnnxInferenceNode',
        namespace=ns,
        parameters=[{
            'model_file_path': MODEL_ONNX_PATH,
            'execution_provider': 'cuda',
            'transport': transport,
        }],
        remappings=[
            ('tensor_input', 'tensor_pub'),
            ('tensor_output', 'tensor_sub'),
        ]
    )

    rtdetr_decoder_node = ComposableNode(
        name='rtdetr_decoder',
        package='isaac_ros_rtdetr',
        plugin='nvidia::isaac_ros::rtdetr::RtDetrDecoderNode',
        namespace=ns
    )

    inference_nodes = [onnx_node]
    if transport == 'managed':
        nitros_to_managed_node = ComposableNode(
            name='nitros_to_managed', package='isaac_ros_onnx_inference',
            plugin='nvidia::isaac_ros::onnx_inference::NitrosToManagedTensorListNode',
            namespace=ns,
            remappings=[('tensor_input', 'tensor_pub'),
                        ('tensor_output', 'managed_tensor_input')])
        onnx_node = ComposableNode(
            name='onnx_inference', package='isaac_ros_onnx_inference',
            plugin='nvidia::isaac_ros::onnx_inference::OnnxInferenceNode', namespace=ns,
            parameters=[{
                'model_file_path': MODEL_ONNX_PATH,
                'execution_provider': 'cuda',
                'transport': 'managed',
            }],
            remappings=[('tensor_input', 'managed_tensor_input'),
                        ('tensor_output', 'managed_tensor_output')])
        managed_to_nitros_node = ComposableNode(
            name='managed_to_nitros', package='isaac_ros_onnx_inference',
            plugin='nvidia::isaac_ros::onnx_inference::ManagedToNitrosTensorListNode',
            namespace=ns,
            remappings=[('tensor_input', 'managed_tensor_output'),
                        ('tensor_output', 'tensor_sub')])
        inference_nodes = [nitros_to_managed_node, onnx_node, managed_to_nitros_node]

    container = ComposableNodeContainer(
        name='rtdetr_container',
        namespace='rtdetr_container',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            resize_node, pad_node, image_format_node,
            image_to_tensor_node, interleave_to_planar_node, reshape_node,
            rtdetr_preprocessor_node, *inference_nodes, rtdetr_decoder_node
        ],
        output='screen'
    )

    return test_class.generate_test_description([container])


def generate_test_description():
    """Generate Config C's direct NITROS proof-of-life graph."""
    return generate_rtdetr_pol_description(IsaacROSOnnxRtDetrPOLTest, 'nitros')


class IsaacROSOnnxRtDetrPOLTest(IsaacROSBaseTest):
    """Validate OnnxInferenceNode produces detections inside a RT-DETR NITROS graph."""

    filepath = pathlib.Path(os.path.dirname(__file__))
    INIT_WAIT_SEC = 10

    @IsaacROSBaseTest.for_each_test_case()
    def test_object_detection(self, test_folder):
        """Expect the pipeline to produce a detection array given an image."""
        self.node._logger.info(f'Generating model (timeout={MODEL_GENERATION_TIMEOUT_SEC}s)')
        start_time = time.time()
        while not os.path.isfile(MODEL_ONNX_PATH):
            if time.time() - start_time > MODEL_GENERATION_TIMEOUT_SEC:
                self.fail('Model generation timed out')
            time.sleep(1)

        received_messages = {}
        self.generate_namespace_lookup(['image', 'camera_info', 'detections_output'])

        image_pub = self.node.create_publisher(
            Image, self.namespaces['image'], self.DEFAULT_QOS)
        camera_info_pub = self.node.create_publisher(
            CameraInfo, self.namespaces['camera_info'], self.DEFAULT_QOS)
        subs = self.create_logging_subscribers(
            [('detections_output', Detection2DArray)], received_messages)

        try:
            image = JSONConversion.load_image_from_json(test_folder / 'image.json')
            camera_info = JSONConversion.load_camera_info_from_json(
                test_folder / 'camera_info.json')
            timestamp = self.node.get_clock().now().to_msg()
            image.header.stamp = timestamp
            camera_info.header.stamp = timestamp

            end_time = time.time() + 60
            done = False
            while time.time() < end_time:
                image_pub.publish(image)
                camera_info_pub.publish(camera_info)
                rclpy.spin_once(self.node, timeout_sec=0.1)
                if 'detections_output' in received_messages:
                    done = True
                    break

            self.assertTrue(done, "Didn't receive output on detections_output topic!")
        finally:
            self.node.destroy_subscription(subs)
            self.node.destroy_publisher(image_pub)
