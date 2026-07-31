# Copyright 2026 Maintainer
# Licensed under the Apache License, Version 2.0.
"""Proof-of-life for NITROS → Managed → ORT CUDA → Managed → NITROS."""

from isaac_ros_onnx_rtdetr_pol_test import (
    IsaacROSOnnxRtDetrPOLTest,
    generate_rtdetr_pol_description,
)


class IsaacROSOnnxRtDetrManagedPOLTest(IsaacROSOnnxRtDetrPOLTest):
    """Run the fixed-input RT-DETR POL graph through both Managed boundaries."""


def generate_test_description():
    return generate_rtdetr_pol_description(IsaacROSOnnxRtDetrManagedPOLTest, 'managed')
