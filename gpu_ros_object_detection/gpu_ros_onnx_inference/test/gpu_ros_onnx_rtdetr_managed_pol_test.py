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

"""Proof-of-life for NITROS → Managed → ORT CUDA → Managed → NITROS."""

import importlib.util
import pathlib


_POL_MODULE_PATH = pathlib.Path(__file__).with_name('gpu_ros_onnx_rtdetr_pol_test.py')
_POL_SPEC = importlib.util.spec_from_file_location('gpu_ros_onnx_rtdetr_pol_test', _POL_MODULE_PATH)
if _POL_SPEC is None or _POL_SPEC.loader is None:
    raise ImportError(f'Unable to load shared POL module from {_POL_MODULE_PATH}')
_POL_MODULE = importlib.util.module_from_spec(_POL_SPEC)
_POL_SPEC.loader.exec_module(_POL_MODULE)

GpuRosOnnxRtDetrPOLTest = _POL_MODULE.GpuRosOnnxRtDetrPOLTest
generate_rtdetr_pol_description = _POL_MODULE.generate_rtdetr_pol_description


class GpuRosOnnxRtDetrManagedPOLTest(GpuRosOnnxRtDetrPOLTest):
    """Run the fixed-input RT-DETR POL graph through both Managed boundaries."""


def generate_test_description():
    return generate_rtdetr_pol_description(GpuRosOnnxRtDetrManagedPOLTest, 'managed')
