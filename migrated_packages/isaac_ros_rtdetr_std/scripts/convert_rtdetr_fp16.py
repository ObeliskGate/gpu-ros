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

"""Convert the RT-DETR ONNX model to FP16 and sanity-check it.

keep_io_types=True preserves the original input/output dtypes (notably the
int64 orig_target_sizes input and labels output), so only the internal float32
weights/activations become float16.
"""

import argparse
import sys

import numpy as np
import onnx
from onnxconverter_common import float16
import onnxruntime as ort


def convert(src_path: str, dst_path: str) -> None:
    model = onnx.load(src_path)
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, dst_path)
    print(f'Wrote FP16 model: {dst_path}')


def sanity_check(model_path: str) -> None:
    """Run one zero-input inference and assert outputs are finite."""
    sess = ort.InferenceSession(
        model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    feeds = {}
    for inp in sess.get_inputs():
        # Replace dynamic dims with 1; RT-DETR uses [B,3,640,640] + [B,2].
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        if 'int64' in inp.type:
            feeds[inp.name] = np.full(shape, 640, dtype=np.int64)
        else:
            feeds[inp.name] = np.zeros(shape, dtype=np.float32)

    outputs = sess.run(None, feeds)
    for meta, arr in zip(sess.get_outputs(), outputs):
        if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
            raise RuntimeError(f'Output {meta.name} contains NaN/Inf after FP16 conversion')
        print(f'  output {meta.name}: shape={arr.shape} dtype={arr.dtype}')
    print('Sanity check passed.')


def main() -> int:
    parser = argparse.ArgumentParser(description='Convert RT-DETR ONNX to FP16.')
    parser.add_argument('--input', required=True, help='Path to FP32 ONNX model')
    parser.add_argument('--output', required=True, help='Path to write FP16 ONNX model')
    args = parser.parse_args()

    convert(args.input, args.output)
    sanity_check(args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
