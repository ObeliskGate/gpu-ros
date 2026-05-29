#!/usr/bin/env bash
# Copyright 2026 Maintainer
#
# Licensed under the Apache License, Version 2.0 (the "License").
# Prepare RT-DETR FP16 artifacts for the Phase 1 benchmark matrix:
#   1. FP16 ONNX  (for ONNX Runtime, configs C/D)
#   2. FP16 TRT engine  (for TensorRT, configs A/B)
#
# Idempotent: skips steps whose outputs already exist. Re-run after switching
# GPUs, since TRT engines are not portable across architectures.
set -euo pipefail

ASSETS_ROOT="${ROS2_BENCHMARK_OVERRIDE_ASSETS_ROOT:-/workspaces/isaac_ros-dev/assets}"
MODEL_DIR="${ASSETS_ROOT}/models/sdetr"
FP32_ONNX="${MODEL_DIR}/sdetr_grasp.onnx"
FP16_ONNX="${MODEL_DIR}/sdetr_grasp_fp16.onnx"
FP16_PLAN="${MODEL_DIR}/sdetr_grasp_fp16.plan"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "${FP32_ONNX}" ]]; then
  echo "ERROR: FP32 model not found at ${FP32_ONNX}" >&2
  echo "Provide the model locally under its applicable terms; this source release does not grant model rights." >&2
  exit 1
fi

# 1. FP16 ONNX
if [[ -f "${FP16_ONNX}" ]]; then
  echo "[skip] FP16 ONNX already exists: ${FP16_ONNX}"
else
  python3 -c "import onnxconverter_common" 2>/dev/null || pip install --quiet onnxconverter-common
  echo "[run] Converting to FP16 ONNX..."
  python3 "${SCRIPT_DIR}/convert_rtdetr_fp16.py" --input "${FP32_ONNX}" --output "${FP16_ONNX}"
fi

# 2. FP16 TRT engine (GPU-architecture specific)
if [[ -f "${FP16_PLAN}" ]]; then
  echo "[skip] FP16 TRT engine already exists: ${FP16_PLAN}"
else
  echo "[run] Building FP16 TRT engine (this takes a few minutes)..."
  trtexec --onnx="${FP16_ONNX}" --saveEngine="${FP16_PLAN}" --fp16
fi

echo "Done. Artifacts in ${MODEL_DIR}:"
ls -la "${FP16_ONNX}" "${FP16_PLAN}"
