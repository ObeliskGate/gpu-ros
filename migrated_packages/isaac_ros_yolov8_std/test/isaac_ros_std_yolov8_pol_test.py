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
Proof-Of-Life test for the std-ROS2 YOLOv8 pipeline.

Generates a tiny constant-output YOLOv8-shaped ONNX, then runs:
image_proc/tensor_proc -> OnnxInferenceNode(transport=std) -> std YOLOv8 decoder.
"""

from array import array
import os
import pathlib
import time

from isaac_ros_test import IsaacROSBaseTest
from launch_ros.actions.composable_node_container import ComposableNodeContainer
from launch_ros.descriptions.composable_node import ComposableNode
import pytest
import rclpy
from sensor_msgs.msg import CameraInfo, Image
import torch
from vision_msgs.msg import Detection2DArray


MODEL_ONNX_PATH = '/tmp/yolov8_std_pol_model.onnx'
MODEL_GENERATION_TIMEOUT_SEC = 300


class ConstantYoloV8(torch.nn.Module):
    """Minimal YOLOv8-shaped model with one high-confidence detection."""

    def __init__(self):
        super().__init__()
        output = torch.zeros((1, 84, 8400), dtype=torch.float32)
        output[0, 0, 0] = 100.0
        output[0, 1, 0] = 120.0
        output[0, 2, 0] = 40.0
        output[0, 3, 0] = 50.0
        output[0, 4, 0] = 0.95
        self.register_buffer('output', output)

    def forward(self, images):
        return self.output + images[:, :1, :1, :1].sum() * 0.0


def generate_model():
    model = ConstantYoloV8()
    dummy_input = torch.zeros((1, 3, 640, 640), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        MODEL_ONNX_PATH,
        input_names=['images'],
        output_names=['output0'],
        opset_version=17,
        do_constant_folding=False)


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description for the all-std-ROS2 YOLOv8 POL test."""
    generate_model()

    ns = IsaacROSStdYoloV8POLTest.generate_namespace()

    resize_node = ComposableNode(
        name='resize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        namespace=ns,
        parameters=[{
            'input_width': 640,
            'input_height': 640,
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
        parameters=[{'scale': True, 'tensor_name': 'image'}],
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
            'output_tensor_name': 'images',
            'input_tensor_shape': [3, 640, 640],
            'output_tensor_shape': [1, 3, 640, 640]
        }],
        remappings=[('tensor', 'planar_tensor')]
    )

    onnx_node = ComposableNode(
        name='onnx_inference',
        package='isaac_ros_onnx_inference',
        plugin='nvidia::isaac_ros::onnx_inference::OnnxInferenceNode',
        namespace=ns,
        parameters=[{
            'model_file_path': MODEL_ONNX_PATH,
            'execution_provider': 'cuda',
            'transport': 'std',
        }],
        remappings=[
            ('tensor_input', 'reshaped_tensor'),
            ('tensor_output', 'tensor_sub'),
        ]
    )

    yolov8_decoder_node = ComposableNode(
        name='yolov8_decoder',
        package='isaac_ros_yolov8_std',
        plugin='nvidia::isaac_ros::yolov8_std::YoloV8DecoderNode',
        namespace=ns,
        parameters=[{
            'tensor_name': 'output0',
            'confidence_threshold': 0.25,
            'nms_threshold': 0.45,
            'num_classes': 80,
        }]
    )

    container = ComposableNodeContainer(
        name='yolov8_container',
        namespace='yolov8_container',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            resize_node, pad_node, image_format_node,
            image_to_tensor_node, interleave_to_planar_node, reshape_node,
            onnx_node, yolov8_decoder_node
        ],
        output='screen'
    )

    return IsaacROSStdYoloV8POLTest.generate_test_description([container])


class IsaacROSStdYoloV8POLTest(IsaacROSBaseTest):
    """Validate ORT(std) + std YOLOv8 decoder publishes detections."""

    filepath = pathlib.Path(os.path.dirname(__file__))
    INIT_WAIT_SEC = 10

    def test_object_detection(self):
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
            image = Image()
            image.height = 640
            image.width = 640
            image.encoding = 'rgb8'
            image.step = 640 * 3
            image.data = array('B', [0]) * (image.height * image.step)

            camera_info = CameraInfo()
            camera_info.width = image.width
            camera_info.height = image.height
            camera_info.distortion_model = 'plumb_bob'

            end_time = time.time() + 60
            done = False
            while time.time() < end_time:
                timestamp = self.node.get_clock().now().to_msg()
                image.header.stamp = timestamp
                camera_info.header.stamp = timestamp
                image_pub.publish(image)
                camera_info_pub.publish(camera_info)
                rclpy.spin_once(self.node, timeout_sec=0.1)
                if 'detections_output' in received_messages:
                    done = True
                    break

            self.assertTrue(done, "Didn't receive output on detections_output topic!")
            self.assertGreaterEqual(
                len(received_messages['detections_output'].detections), 1)
        finally:
            self.node.destroy_subscription(subs)
            self.node.destroy_publisher(image_pub)
            self.node.destroy_publisher(camera_info_pub)
