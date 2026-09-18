#!/usr/bin/env python3
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

from gpu_ros_tensor_bundle_msgs.msg import Tensor, TensorBundle
from rclpy.serialization import deserialize_message, serialize_message


def test_tensor_bundle_roundtrip_preserves_wire_fields():
    message = TensorBundle()
    message.header.frame_id = "camera"
    message.header.stamp.sec = 17
    message.header.stamp.nanosec = 23

    tensor = Tensor()
    tensor.name = "images"
    tensor.data_type = Tensor.FLOAT32
    tensor.shape = [1, 3, 2, 2]
    tensor.data = list(bytes(range(48)))
    message.tensors = [tensor]

    scalar = Tensor()
    scalar.name = "class_ids"
    scalar.data_type = Tensor.INT64
    scalar.shape = [2]
    scalar.data = list((1).to_bytes(8, "little", signed=True)) + list(
        (7).to_bytes(8, "little", signed=True)
    )
    message.tensors.append(scalar)

    restored = deserialize_message(serialize_message(message), TensorBundle)
    assert restored.header.frame_id == "camera"
    assert restored.header.stamp.sec == 17
    assert restored.header.stamp.nanosec == 23
    assert len(restored.tensors) == 2
    assert restored.tensors[0].name == "images"
    assert restored.tensors[0].data_type == Tensor.FLOAT32
    assert list(restored.tensors[0].shape) == [1, 3, 2, 2]
    assert list(restored.tensors[0].data) == list(bytes(range(48)))
    assert restored.tensors[1].name == "class_ids"
    assert restored.tensors[1].data_type == Tensor.INT64
    assert list(restored.tensors[1].shape) == [2]
    assert list(restored.tensors[1].data) == list(
        (1).to_bytes(8, "little", signed=True) + (7).to_bytes(8, "little", signed=True)
    )
