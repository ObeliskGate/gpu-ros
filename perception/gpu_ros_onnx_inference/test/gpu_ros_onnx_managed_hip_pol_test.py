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

"""Proof-of-life for the AMD standard-ROS2 -> Managed HIP graphs."""

from array import array
import os
import pathlib
import shutil
import time
import unittest

import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import launch_testing.actions
import onnx
from onnx import helper, TensorProto
import pytest
import rclpy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray


TEST_ARTIFACT_ROOT = pathlib.Path(
    f'/tmp/gpu_ros_onnx_managed_hip_pol_{os.getpid()}')
YOLO_MODEL_PATH = TEST_ARTIFACT_ROOT / 'yolov8.onnx'
RTDETR_MODEL_PATH = TEST_ARTIFACT_ROOT / 'rtdetr.onnx'
MIGRAPHX_CACHE_PATH = TEST_ARTIFACT_ROOT / 'migraphx_cache'
YOLO_NAMESPACE = 'yolov8_managed_hip_pol'
RTDETR_NAMESPACE = 'rtdetr_managed_hip_pol'


def _write_yolov8_model():
    """Write a deterministic static-output YOLO-shaped model."""
    output_shape = [1, 84, 8400]
    output_values = [0.0] * (84 * 8400)
    output_values[0] = 100.0
    output_values[8400] = 120.0
    output_values[2 * 8400] = 40.0
    output_values[3 * 8400] = 50.0
    output_values[4 * 8400] = 0.95
    graph = helper.make_graph(
        [
            helper.make_node('ReduceMean', ['images'], ['image_mean'], keepdims=0),
            helper.make_node('Mul', ['image_mean', 'zero'], ['image_zero']),
            helper.make_node('Add', ['output_base', 'image_zero'], ['output0']),
        ],
        'yolov8_managed_hip_pol',
        [helper.make_tensor_value_info('images', TensorProto.FLOAT, [-1, 3, 640, 640])],
        [helper.make_tensor_value_info('output0', TensorProto.FLOAT, [-1, 84, 8400])],
        initializer=[
            helper.make_tensor('output_base', TensorProto.FLOAT, output_shape, output_values),
            helper.make_tensor('zero', TensorProto.FLOAT, [], [0.0]),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid('', 17)],
        producer_name='gpu_ros_onnx_inference_managed_hip_test')
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, YOLO_MODEL_PATH)


def _write_rtdetr_model():
    """Write a deterministic static-output RT-DETR-shaped model."""
    labels_shape = [1, 300]
    boxes_shape = [1, 300, 4]
    scores_shape = [1, 300]
    labels_values = [0] * 300
    labels_values[0] = 7
    boxes_values = [0.0] * (300 * 4)
    boxes_values[:4] = [10.0, 20.0, 30.0, 50.0]
    scores_values = [0.0] * 300
    scores_values[0] = 0.95
    graph = helper.make_graph(
        [
            helper.make_node('Identity', ['labels_base'], ['labels']),
            helper.make_node('ReduceMean', ['images'], ['image_mean'], keepdims=0),
            helper.make_node('Mul', ['image_mean', 'zero'], ['image_zero']),
            helper.make_node('Add', ['boxes_base', 'image_zero'], ['boxes']),
            helper.make_node('Add', ['scores_base', 'image_zero'], ['scores']),
        ],
        'rtdetr_managed_hip_pol',
        [
            helper.make_tensor_value_info('images', TensorProto.FLOAT, [-1, 3, 640, 640]),
            helper.make_tensor_value_info('orig_target_sizes', TensorProto.INT64, [-1, 2]),
        ],
        [
            helper.make_tensor_value_info('labels', TensorProto.INT64, [-1, 300]),
            helper.make_tensor_value_info('boxes', TensorProto.FLOAT, [-1, 300, 4]),
            helper.make_tensor_value_info('scores', TensorProto.FLOAT, [-1, 300]),
        ],
        initializer=[
            helper.make_tensor('labels_base', TensorProto.INT64, labels_shape, labels_values),
            helper.make_tensor('boxes_base', TensorProto.FLOAT, boxes_shape, boxes_values),
            helper.make_tensor('scores_base', TensorProto.FLOAT, scores_shape, scores_values),
            helper.make_tensor('zero', TensorProto.FLOAT, [], [0.0]),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid('', 17)],
        producer_name='gpu_ros_onnx_inference_managed_hip_test')
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, RTDETR_MODEL_PATH)


def _yolov8_nodes():
    namespace = YOLO_NAMESPACE
    return [
        ComposableNode(
            package='gpu_ros_yolov8',
            plugin=(
                'gpu_ros::yolov8::'
                'YoloV8ManagedHipImageEncoderNode'),
            name='image_encoder', namespace=namespace,
            parameters=[{
                'tensor_name': 'images', 'output_width': 640, 'output_height': 640,
                'gpu_device_id': 0,
                'managed_pool_capacity': 16,
                'managed_pool_wait_timeout_ms': 100,
            }],
        ),
        ComposableNode(
            package='gpu_ros_onnx_inference',
            plugin='gpu_ros::onnx_inference::OnnxInferenceNode',
            name='inference', namespace=namespace,
            parameters=[{
                'model_file_path': str(YOLO_MODEL_PATH),
                'execution_provider': 'migraphx',
                'gpu_device_id': 0,
                'transport': 'managed',
                'managed_io_contract': 'hip_managed_strict',
                'managed_input_contracts': ['images=float32[1,3,640,640]'],
                'managed_output_contracts': ['output0=float32[1,84,8400]'],
                'managed_pool_capacity': 16,
                'managed_pool_wait_timeout_ms': 100,
            }],
            remappings=[
                ('tensor_input', 'managed_tensor_output'),
                ('tensor_output', 'managed_output'),
            ],
        ),
        ComposableNode(
            package='gpu_ros_yolov8',
            plugin=(
                'gpu_ros::yolov8::'
                'YoloV8ManagedHipDecoderNode'),
            name='decoder', namespace=namespace,
            parameters=[{
                'tensor_name': 'output0',
                'gpu_device_id': 0,
                'confidence_threshold': 0.25,
                'nms_threshold': 0.45,
                'num_classes': 80,
            }],
            remappings=[('managed_tensor_input', 'managed_output')],
        ),
    ]


def _rtdetr_nodes():
    namespace = RTDETR_NAMESPACE
    return [
        ComposableNode(
            package='gpu_ros_rtdetr',
            plugin=(
                'gpu_ros::rtdetr::'
                'RtDetrManagedHipImageEncoderNode'),
            name='image_encoder', namespace=namespace,
            parameters=[{
                'tensor_name': 'input_tensor', 'output_width': 640, 'output_height': 640,
                'gpu_device_id': 0,
                'managed_pool_capacity': 16,
                'managed_pool_wait_timeout_ms': 100,
            }],
            remappings=[('managed_tensor_output', 'managed_tensor_image')],
        ),
        ComposableNode(
            package='gpu_ros_rtdetr',
            plugin=(
                'gpu_ros::rtdetr::'
                'RtDetrManagedHipPreprocessorNode'),
            name='preprocessor', namespace=namespace,
            parameters=[{
                'image_width': 640,
                'image_height': 640,
                'model_input_width': 640,
                'model_input_height': 640,
                'gpu_device_id': 0,
                'managed_pool_capacity': 16,
                'managed_pool_wait_timeout_ms': 100,
            }],
            remappings=[('managed_tensor_input', 'managed_tensor_image')],
        ),
        ComposableNode(
            package='gpu_ros_onnx_inference',
            plugin='gpu_ros::onnx_inference::OnnxInferenceNode',
            name='inference', namespace=namespace,
            parameters=[{
                'model_file_path': str(RTDETR_MODEL_PATH),
                'execution_provider': 'migraphx',
                'gpu_device_id': 0,
                'transport': 'managed',
                'managed_io_contract': 'hip_managed_strict',
                'managed_input_contracts': [
                    'images=float32[1,3,640,640]',
                    'orig_target_sizes=int64[1,2]',
                ],
                'managed_output_contracts': [
                    'labels=int64[1,300]',
                    'boxes=float32[1,300,4]',
                    'scores=float32[1,300]',
                ],
                'managed_pool_capacity': 16,
                'managed_pool_wait_timeout_ms': 100,
            }],
            remappings=[
                ('tensor_input', 'managed_tensor_output'),
                ('tensor_output', 'managed_output'),
            ],
        ),
        ComposableNode(
            package='gpu_ros_rtdetr',
            plugin=(
                'gpu_ros::rtdetr::'
                'RtDetrManagedHipDecoderNode'),
            name='decoder', namespace=namespace,
            parameters=[{
                'gpu_device_id': 0,
                'confidence_threshold': 0.5,
            }],
            remappings=[('managed_tensor_input', 'managed_output')],
        ),
    ]


@pytest.mark.launch_test
def generate_test_description():
    """Launch both static-output AMD Managed HIP proof-of-life graphs."""
    TEST_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=False)
    MIGRAPHX_CACHE_PATH.mkdir()
    _write_yolov8_model()
    _write_rtdetr_model()
    container = ComposableNodeContainer(
        package='rclcpp_components',
        executable='component_container_mt',
        name='managed_hip_pol_container',
        namespace='',
        composable_node_descriptions=_yolov8_nodes() + _rtdetr_nodes(),
        output='screen',
        additional_env={
            'ORT_MIGRAPHX_MODEL_CACHE_PATH': str(MIGRAPHX_CACHE_PATH),
            'ORT_MIGRAPHX_FP16_ENABLE': '0',
            'ORT_MIGRAPHX_BF16_ENABLE': '0',
            'ORT_MIGRAPHX_FP8_ENABLE': '0',
            'ORT_MIGRAPHX_INT8_ENABLE': '0',
        },
    )
    return launch.LaunchDescription([
        container,
        launch_testing.actions.ReadyToTest(),
    ])


class TestManagedHipProofOfLife(unittest.TestCase):
    """Verify both direct strict Managed HIP graphs produce detections."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node('managed_hip_pol_test_client')

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()
        shutil.rmtree(TEST_ARTIFACT_ROOT, ignore_errors=True)

    def _wait_for_detection(self, namespace, image):
        received = []
        publisher = self.node.create_publisher(Image, f'/{namespace}/image', 10)
        subscription = self.node.create_subscription(
            Detection2DArray,
            f'/{namespace}/detections_output',
            received.append,
            10,
        )
        try:
            deadline = time.monotonic() + 180.0
            next_publish = 0.0
            while time.monotonic() < deadline and not received:
                now = time.monotonic()
                if now >= next_publish:
                    image.header.stamp = self.node.get_clock().now().to_msg()
                    publisher.publish(image)
                    next_publish = now + 0.25
                rclpy.spin_once(self.node, timeout_sec=0.1)
            self.assertTrue(received, f'No detection output from {namespace}')
            return received[-1]
        finally:
            self.node.destroy_subscription(subscription)
            self.node.destroy_publisher(publisher)

    def test_yolov8_managed_hip_graph(self):
        """Exercise HIP preprocessing, strict output binding, and D2H decoding."""
        image = Image()
        image.height = 640
        image.width = 640
        image.encoding = 'rgb8'
        image.step = image.width * 3
        image.data = array('B', [0]) * (image.height * image.step)
        image.header.frame_id = 'camera'
        output = self._wait_for_detection(YOLO_NAMESPACE, image)
        self.assertEqual(output.header.frame_id, 'camera')
        self.assertGreaterEqual(len(output.detections), 1)

    def test_rtdetr_managed_hip_graph(self):
        """Exercise the RT-DETR two-input Managed HIP graph."""
        image = Image()
        image.height = 1
        image.width = 2
        image.encoding = 'rgb8'
        image.step = 6
        image.data = [10, 20, 30, 40, 50, 60]
        image.header.frame_id = 'camera'
        output = self._wait_for_detection(RTDETR_NAMESPACE, image)
        self.assertEqual(output.header.frame_id, 'camera')
        self.assertEqual(len(output.detections), 1)
        self.assertEqual(output.detections[0].results[0].hypothesis.class_id, '7')
