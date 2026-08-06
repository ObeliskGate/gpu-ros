# Copyright 2026 Maintainer
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

"""Tiny-model YOLOv8 std ROS2 + MIGraphX proof-of-life test."""

from array import array
import os
import pathlib
import time
import unittest

import launch
from launch_ros.actions.composable_node_container import ComposableNodeContainer
from launch_ros.descriptions.composable_node import ComposableNode
import launch_testing.actions
import onnx
from onnx import helper, TensorProto
import pytest
import rclpy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray


MODEL_PATH = pathlib.Path('/tmp/yolov8_std_migraphx_pol.onnx')
POL_TIMEOUT_SEC = float(os.environ.get('YOLOV8_MIGRAPHX_POL_TIMEOUT_SEC', '900'))
NAMESPACE = 'yolov8_migraphx_pol'


def generate_test_model():
    """Create a tiny deterministic YOLOv8-shaped model for MIGraphX."""
    output_shape = [1, 84, 8400]
    output_values = [0.0] * (84 * 8400)
    output_values[0] = 100.0
    output_values[8400] = 120.0
    output_values[2 * 8400] = 40.0
    output_values[3 * 8400] = 50.0
    output_values[4 * 8400] = 0.95

    inputs = [
        helper.make_tensor_value_info(
            'images', TensorProto.FLOAT, [1, 3, 640, 640]),
    ]
    outputs = [
        helper.make_tensor_value_info(
            'output0', TensorProto.FLOAT, output_shape),
    ]
    initializers = [
        helper.make_tensor(
            'output_base', TensorProto.FLOAT, output_shape, output_values),
        helper.make_tensor('zero', TensorProto.FLOAT, [], [0.0]),
    ]
    nodes = [
        helper.make_node('ReduceMean', ['images'], ['image_mean'], keepdims=0),
        helper.make_node('Mul', ['image_mean', 'zero'], ['image_zero']),
        helper.make_node('Add', ['output_base', 'image_zero'], ['output0']),
    ]
    graph = helper.make_graph(
        nodes,
        'yolov8_std_migraphx_pol',
        inputs,
        outputs,
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid('', 17)],
        producer_name='isaac_ros_yolov8_std_test',
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, MODEL_PATH)


@pytest.mark.launch_test
def generate_test_description():
    """Launch the complete AMD target path with a deterministic tiny model."""
    generate_test_model()

    image_encoder_node = ComposableNode(
        name='yolov8_image_encoder',
        namespace=NAMESPACE,
        package='isaac_ros_yolov8_std',
        plugin='nvidia::isaac_ros::yolov8_std::YoloV8ImageEncoderNode',
        parameters=[{
            'tensor_name': 'images',
            'output_width': 640,
            'output_height': 640,
        }],
    )

    onnx_node = ComposableNode(
        name='onnx_inference',
        namespace=NAMESPACE,
        package='isaac_ros_onnx_inference',
        plugin='nvidia::isaac_ros::onnx_inference::OnnxInferenceNode',
        parameters=[{
            'model_file_path': str(MODEL_PATH),
            'execution_provider': 'migraphx',
            'transport': 'std',
        }],
        remappings=[
            ('tensor_input', 'encoded_tensor'),
            ('tensor_output', 'tensor_sub'),
        ],
    )

    decoder_node = ComposableNode(
        name='yolov8_decoder',
        namespace=NAMESPACE,
        package='isaac_ros_yolov8_std',
        plugin='nvidia::isaac_ros::yolov8_std::YoloV8DecoderNode',
        parameters=[{
            'tensor_name': 'output0',
            'confidence_threshold': 0.25,
            'nms_threshold': 0.45,
            'num_classes': 80,
        }],
    )

    container = ComposableNodeContainer(
        name='yolov8_migraphx_pol_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[image_encoder_node, onnx_node, decoder_node],
        output='screen',
    )
    return launch.LaunchDescription([
        container,
        launch_testing.actions.ReadyToTest(),
    ])


class TestYoloV8MigraphxProofOfLife(unittest.TestCase):
    """Check message flow, header propagation, and decoding with a tiny model."""

    @classmethod
    def setUpClass(cls):
        """Create the ROS test client."""
        rclpy.init()
        cls.node = rclpy.create_node('yolov8_migraphx_pol_test_client')

    @classmethod
    def tearDownClass(cls):
        """Destroy the ROS test client and generated model."""
        cls.node.destroy_node()
        rclpy.shutdown()
        MODEL_PATH.unlink(missing_ok=True)

    def test_graph_publishes_detection_array(self):
        received_messages = []
        image_topic = f'/{NAMESPACE}/image'
        detection_topic = f'/{NAMESPACE}/detections_output'
        image_pub = self.node.create_publisher(Image, image_topic, 10)
        detection_sub = self.node.create_subscription(
            Detection2DArray, detection_topic, received_messages.append, 10)

        image = Image()
        image.height = 640
        image.width = 640
        image.encoding = 'rgb8'
        image.step = image.width * 3
        image.data = array('B', [0]) * (image.height * image.step)
        image.header.frame_id = 'camera'

        try:
            deadline = time.monotonic() + POL_TIMEOUT_SEC
            next_publish = 0.0
            while time.monotonic() < deadline:
                now = time.monotonic()
                if now >= next_publish:
                    image.header.stamp = self.node.get_clock().now().to_msg()
                    image_pub.publish(image)
                    next_publish = now + 0.25
                rclpy.spin_once(self.node, timeout_sec=0.1)
                if received_messages:
                    break

            self.assertTrue(
                received_messages,
                'The tiny YOLOv8 graph did not publish Detection2DArray',
            )
            output = received_messages[-1]
            self.assertEqual(output.header.frame_id, 'camera')
            self.assertGreaterEqual(len(output.detections), 1)
        finally:
            self.node.destroy_subscription(detection_sub)
            self.node.destroy_publisher(image_pub)
