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
#
"""Small real ONNX computation for dynamic native-output regression, not model smoke."""

import pathlib
import sys

import onnx
from onnx import TensorProto, helper


def write_model(path):
    graph = helper.make_graph(
        [helper.make_node('Add', ['values', 'one'], ['result'])],
        'dynamic_native_output',
        [helper.make_tensor_value_info('values', TensorProto.FLOAT, [1, 'count'])],
        [helper.make_tensor_value_info('result', TensorProto.FLOAT, [1, 'count'])],
        [helper.make_tensor('one', TensorProto.FLOAT, [], [1.0])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)


if __name__ == '__main__':
    write_model(pathlib.Path(sys.argv[1]))
