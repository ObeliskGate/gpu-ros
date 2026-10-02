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

"""Generate a small, static RT-DETR-style fixture for real ORT caller tests."""

import pathlib
import sys

import onnx
from onnx import helper, TensorProto


def write_model(path):
    """Use the POL graph's input dependency and three deterministic outputs."""
    labels = [7] + [0] * 299
    boxes = [10.0, 20.0, 30.0, 50.0] + [0.0] * 1196
    scores = [0.95] + [0.0] * 299
    graph = helper.make_graph(
        [
            helper.make_node('Identity', ['labels_base'], ['labels']),
            helper.make_node('ReduceMean', ['images'], ['image_mean'], keepdims=0),
            helper.make_node('Mul', ['image_mean', 'zero'], ['image_zero']),
            helper.make_node('Add', ['boxes_base', 'image_zero'], ['boxes']),
            helper.make_node('Add', ['scores_base', 'image_zero'], ['scores']),
        ],
        'managed_output_cancellation',
        [
            helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, 2, 2]),
            helper.make_tensor_value_info('orig_target_sizes', TensorProto.INT64, [1, 2]),
        ],
        [
            helper.make_tensor_value_info('labels', TensorProto.INT64, [1, 300]),
            helper.make_tensor_value_info('boxes', TensorProto.FLOAT, [1, 300, 4]),
            helper.make_tensor_value_info('scores', TensorProto.FLOAT, [1, 300]),
        ],
        initializer=[
            helper.make_tensor('labels_base', TensorProto.INT64, [1, 300], labels),
            helper.make_tensor('boxes_base', TensorProto.FLOAT, [1, 300, 4], boxes),
            helper.make_tensor('scores_base', TensorProto.FLOAT, [1, 300], scores),
            helper.make_tensor('zero', TensorProto.FLOAT, [], [0.0]),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid('', 17)],
        producer_name='gpu_ros_onnx_inference_cancellation_test',
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)


if __name__ == '__main__':
    write_model(pathlib.Path(sys.argv[1]))
