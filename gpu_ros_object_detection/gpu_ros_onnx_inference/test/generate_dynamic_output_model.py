# Copyright 2026 Boshen Chen
# Licensed under the Apache License, Version 2.0.
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
