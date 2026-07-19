# Copyright 2026 Maintainer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Proof-of-life test for the Phase 2a standard ROS2 + MIGraphX pipeline."""

import pathlib
import time

import launch
import launch_testing.actions
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import onnx
from onnx import helper, TensorProto
import pytest
import rclpy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray


MODEL_PATH = pathlib.Path('/tmp/rtdetr_std_migraphx_pol.onnx')
NAMESPACE = 'rtdetr_migraphx_pol'


def generate_test_model():
    """Create a small RT-DETR-shaped graph that must execute on MIGraphX."""
    inputs = [
        helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, 640, 640]),
        helper.make_tensor_value_info('orig_target_sizes', TensorProto.INT64, [1, 2]),
    ]
    outputs = [
        helper.make_tensor_value_info('labels', TensorProto.INT64, [1, 1]),
        helper.make_tensor_value_info('boxes', TensorProto.FLOAT, [1, 1, 4]),
        helper.make_tensor_value_info('scores', TensorProto.FLOAT, [1, 1]),
    ]
    initializers = [
        helper.make_tensor('labels', TensorProto.INT64, [1, 1], [7]),
        helper.make_tensor(
            'boxes_base', TensorProto.FLOAT, [1, 1, 4], [10.0, 20.0, 30.0, 50.0]),
        helper.make_tensor('scores_base', TensorProto.FLOAT, [1, 1], [0.95]),
        helper.make_tensor('zero', TensorProto.FLOAT, [], [0.0]),
    ]
    nodes = [
        helper.make_node('ReduceMean', ['images'], ['image_mean'], keepdims=0),
        helper.make_node('Mul', ['image_mean', 'zero'], ['image_zero']),
        helper.make_node('Add', ['boxes_base', 'image_zero'], ['boxes']),
        helper.make_node('Add', ['scores_base', 'image_zero'], ['scores']),
    ]
    graph = helper.make_graph(
        nodes,
        'rtdetr_std_migraphx_pol',
        inputs,
        outputs,
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid('', 17)],
        producer_name='isaac_ros_rtdetr_std_test',
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, MODEL_PATH)


@pytest.mark.launch_test
def generate_test_description():
    """Launch the complete AMD target path with a deterministic test model."""
    generate_test_model()

    image_encoder = ComposableNode(
        package='isaac_ros_rtdetr_std',
        plugin='nvidia::isaac_ros::rtdetr_std::RtDetrImageEncoderNode',
        name='image_encoder',
        namespace=NAMESPACE,
        parameters=[{
            'tensor_name': 'input_tensor',
            'output_width': 640,
            'output_height': 640,
        }],
    )
    preprocessor = ComposableNode(
        package='isaac_ros_rtdetr_std',
        plugin='nvidia::isaac_ros::rtdetr_std::RtDetrPreprocessorNode',
        name='preprocessor',
        namespace=NAMESPACE,
        parameters=[{
            'image_width': 640,
            'image_height': 640,
        }],
    )
    inference = ComposableNode(
        package='isaac_ros_onnx_inference',
        plugin='nvidia::isaac_ros::onnx_inference::OnnxInferenceNode',
        name='inference',
        namespace=NAMESPACE,
        parameters=[{
            'model_file_path': str(MODEL_PATH),
            'execution_provider': 'migraphx',
            'transport': 'std',
        }],
        remappings=[
            ('tensor_input', 'tensor_pub'),
            ('tensor_output', 'tensor_sub'),
        ],
    )
    decoder = ComposableNode(
        package='isaac_ros_rtdetr_std',
        plugin='nvidia::isaac_ros::rtdetr_std::RtDetrDecoderNode',
        name='decoder',
        namespace=NAMESPACE,
        parameters=[{'confidence_threshold': 0.5}],
    )
    container = ComposableNodeContainer(
        package='rclcpp_components',
        executable='component_container_mt',
        name='rtdetr_migraphx_pol_container',
        namespace='',
        composable_node_descriptions=[
            image_encoder,
            preprocessor,
            inference,
            decoder,
        ],
        output='screen',
    )

    return launch.LaunchDescription([
        container,
        launch_testing.actions.ReadyToTest(),
    ])


class TestRtDetrMigraphxProofOfLife:
    """Verify one image traverses the complete Phase 2a target graph."""

    @classmethod
    def setup_class(cls):
        """Create the ROS test client."""
        rclpy.init()
        cls.node = rclpy.create_node('rtdetr_migraphx_pol_test_client')

    @classmethod
    def teardown_class(cls):
        """Destroy test resources and the generated model."""
        cls.node.destroy_node()
        rclpy.shutdown()
        MODEL_PATH.unlink(missing_ok=True)

    def test_detection_output(self):
        """Publish one image and validate the decoded deterministic output."""
        received = []
        subscription = self.node.create_subscription(
            Detection2DArray,
            f'/{NAMESPACE}/detections_output',
            received.append,
            10,
        )
        publisher = self.node.create_publisher(Image, f'/{NAMESPACE}/image', 10)

        image = Image()
        image.header.frame_id = 'camera'
        image.height = 1
        image.width = 2
        image.encoding = 'rgb8'
        image.step = 6
        image.data = [10, 20, 30, 40, 50, 60]

        deadline = time.monotonic() + 120.0
        next_publish = 0.0
        while time.monotonic() < deadline and not received:
            now = time.monotonic()
            if now >= next_publish:
                image.header.stamp = self.node.get_clock().now().to_msg()
                publisher.publish(image)
                next_publish = now + 0.25
            rclpy.spin_once(self.node, timeout_sec=0.1)

        self.node.destroy_publisher(publisher)
        self.node.destroy_subscription(subscription)

        assert received, 'No Detection2DArray received from the MIGraphX pipeline'
        detections = received[-1].detections
        assert len(detections) == 1
        detection = detections[0]
        assert detection.header.frame_id == 'camera'
        assert detection.results[0].hypothesis.class_id == '7'
        assert detection.results[0].hypothesis.score == pytest.approx(0.95)
        assert detection.bbox.center.position.x == pytest.approx(20.0)
        assert detection.bbox.center.position.y == pytest.approx(35.0)
        assert detection.bbox.size_x == pytest.approx(20.0)
        assert detection.bbox.size_y == pytest.approx(30.0)
