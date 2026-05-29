#!/usr/bin/env python3
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

"""Convert the RT-DETR ONNX model to FP16.

keep_io_types=True preserves the original input/output dtypes (notably the
int64 orig_target_sizes input and labels output), so only the internal float32
weights/activations become float16.

Note: numerical validation of the converted model is intentionally not done
here. The end-to-end POL tests and the A-vs-D detection comparison (Task 9)
exercise the real CUDA inference path and verify output correctness; running a
separate Python onnxruntime check here would add a redundant dependency.
"""

import argparse
import sys

import onnx
from onnxconverter_common import float16


def main() -> int:
    parser = argparse.ArgumentParser(description='Convert RT-DETR ONNX to FP16.')
    parser.add_argument('--input', required=True, help='Path to FP32 ONNX model')
    parser.add_argument('--output', required=True, help='Path to write FP16 ONNX model')
    args = parser.parse_args()

    model = onnx.load(args.input)
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, args.output)
    print(f'Wrote FP16 model: {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
