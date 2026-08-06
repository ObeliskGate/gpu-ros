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

"""Canonical-model YOLOv8 std ROS2 + MIGraphX proof-of-life test."""

from array import array
import os
from pathlib import Path
import time

from isaac_ros_test import IsaacROSBaseTest
from launch_ros.actions.composable_node_container import ComposableNodeContainer
from launch_ros.descriptions.composable_node import ComposableNode
import pytest
import rclpy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray


MODEL_PATH = Path(
    os.environ.get("OVG_ASSETS_ROOT", "/workspaces/ovg-assets")
) / "models" / "yolov8" / "yolov8s.onnx"
POL_TIMEOUT_SEC = float(os.environ.get("YOLOV8_MIGRAPHX_POL_TIMEOUT_SEC", "900"))


def require_model() -> str:
    resolved_path = MODEL_PATH.expanduser().resolve(strict=False)
    if not resolved_path.is_file() or resolved_path.stat().st_size == 0:
        raise RuntimeError(
            "YOLOv8 ONNX asset is missing:\n"
            f"{resolved_path}\n"
            "Provide OVG_YOLOV8_ONNX_SOURCE and run phase2 assets import-yolov8."
        )
    return str(resolved_path)


@pytest.mark.rostest
def generate_test_description():
    """Build the canonical YOLOv8 graph only after checking its model asset."""
    model_path = require_model()
    namespace = IsaacROSYoloV8MigraphxPOLTest.generate_namespace()

    image_encoder_node = ComposableNode(
        name="yolov8_image_encoder",
        namespace=namespace,
        package="isaac_ros_yolov8_std",
        plugin="nvidia::isaac_ros::yolov8_std::YoloV8ImageEncoderNode",
        parameters=[{
            "tensor_name": "images",
            "output_width": 640,
            "output_height": 640,
        }],
    )

    onnx_node = ComposableNode(
        name="onnx_inference",
        namespace=namespace,
        package="isaac_ros_onnx_inference",
        plugin="nvidia::isaac_ros::onnx_inference::OnnxInferenceNode",
        parameters=[{
            "model_file_path": model_path,
            "execution_provider": "migraphx",
            "transport": "std",
        }],
        remappings=[
            ("tensor_input", "encoded_tensor"),
            ("tensor_output", "tensor_sub"),
        ],
    )

    decoder_node = ComposableNode(
        name="yolov8_decoder",
        namespace=namespace,
        package="isaac_ros_yolov8_std",
        plugin="nvidia::isaac_ros::yolov8_std::YoloV8DecoderNode",
        parameters=[{
            "tensor_name": "output0",
            "confidence_threshold": 0.25,
            "nms_threshold": 0.45,
            "num_classes": 80,
        }],
    )

    container = ComposableNodeContainer(
        name="yolov8_migraphx_pol_container",
        namespace=namespace,
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=[image_encoder_node, onnx_node, decoder_node],
        output="screen",
    )
    return IsaacROSYoloV8MigraphxPOLTest.generate_test_description([container])


class IsaacROSYoloV8MigraphxPOLTest(IsaacROSBaseTest):
    """Check message flow and header propagation with the real YOLOv8 asset."""

    def test_graph_publishes_detection_array(self):
        received_messages = {}
        self.generate_namespace_lookup(["image", "detections_output"])
        image_pub = self.node.create_publisher(
            Image, self.namespaces["image"], self.DEFAULT_QOS)
        subscriptions = self.create_logging_subscribers(
            [("detections_output", Detection2DArray)], received_messages)

        image = Image()
        image.height = 640
        image.width = 640
        image.encoding = "rgb8"
        image.step = image.width * 3
        image.data = array("B", [0]) * (image.height * image.step)
        image.header.frame_id = "camera"

        try:
            deadline = time.monotonic() + POL_TIMEOUT_SEC
            published_stamp = None
            while time.monotonic() < deadline:
                image.header.stamp = self.node.get_clock().now().to_msg()
                published_stamp = image.header.stamp
                image_pub.publish(image)
                rclpy.spin_once(self.node, timeout_sec=0.1)
                if "detections_output" in received_messages:
                    break

            self.assertIn(
                "detections_output",
                received_messages,
                "The canonical YOLOv8 graph did not publish Detection2DArray",
            )
            output = received_messages["detections_output"]
            self.assertEqual(output.header.frame_id, "camera")
            self.assertEqual(output.header.stamp, published_stamp)
        finally:
            self.node.destroy_subscription(subscriptions)
            self.node.destroy_publisher(image_pub)
