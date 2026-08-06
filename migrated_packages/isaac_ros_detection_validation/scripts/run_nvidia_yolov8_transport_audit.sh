#!/usr/bin/env bash
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

set -euo pipefail

usage() {
  echo "Usage: $0 <audit-name>"
  echo
  echo "Audit YOLOv8 Config C versus Managed with fixed input, ORT profiles and CUDA traces."
  echo "The audit output directory must not already exist."
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

AUDIT_NAME="$1"
if [[ ! ${AUDIT_NAME} =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "ERROR: audit-name may contain only letters, numbers, '.', '_' and '-'." >&2
  exit 2
fi

WORKSPACE_ROOT="${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}"
APP_ROOT="${AMD_ROS_OBJECT_DETECTION_ROOT:-${WORKSPACE_ROOT}/src/amd_ros_object_detection}"
AUDIT_PARENT="${CAPTURE_AUDIT_ROOT:-${APP_ROOT}/migrated_packages/benchmark_results/phase2b_audits}"
AUDIT_ROOT="${AUDIT_PARENT}/${AUDIT_NAME}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CAPTURE_RUNNER="${SCRIPT_DIR}/run_nvidia_fixed_input_capture.sh"
ORT_ROOT="${AUDIT_ROOT}/ort"
NSYS_ROOT="${AUDIT_ROOT}/nsys"
BAG_ROOT="${AUDIT_ROOT}/bags"
REPORT_ROOT="${AUDIT_ROOT}/reports"
LOG_ROOT="${AUDIT_ROOT}/logs"
ORT_PROFILE_FRAMES="${AUDIT_ORT_PROFILE_FRAMES:-50}"
YOLO_INPUT_BYTES=4915200
YOLO_OUTPUT_BYTES=2822400

if [[ -e ${AUDIT_ROOT} ]]; then
  echo "ERROR: refusing to overwrite existing audit path: ${AUDIT_ROOT}" >&2
  exit 1
fi

if [[ ! ${ORT_PROFILE_FRAMES} =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: AUDIT_ORT_PROFILE_FRAMES must be a positive integer." >&2
  exit 2
fi

ROS_SETUP="/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
if [[ -f ${ROS_SETUP} ]]; then
  set +u
  source "${ROS_SETUP}"
  set -u
fi

if [[ -f ${WORKSPACE_ROOT}/install/setup.bash ]]; then
  set +u
  source "${WORKSPACE_ROOT}/install/setup.bash"
  set -u
fi

for command_name in awk ctest find grep nsys ros2 tee; do
  if ! command -v "${command_name}" >/dev/null; then
    echo "ERROR: required command is unavailable: ${command_name}" >&2
    exit 1
  fi
done

if [[ ! -x ${CAPTURE_RUNNER} ]]; then
  echo "ERROR: capture runner is missing or not executable: ${CAPTURE_RUNNER}" >&2
  exit 1
fi

if [[ ! -d ${WORKSPACE_ROOT}/build/isaac_ros_onnx_inference ]]; then
  echo "ERROR: build isaac_ros_onnx_inference before running the audit." >&2
  exit 1
fi

mkdir -p "${ORT_ROOT}" "${NSYS_ROOT}" "${BAG_ROOT}" "${REPORT_ROOT}" "${LOG_ROOT}"

echo "Running pointer-identity and CUDA I/O Binding tests..."
ctest \
  --test-dir "${WORKSPACE_ROOT}/build/isaac_ros_onnx_inference" \
  --tests-regex 'test_(nitros_managed_tensor_list_adapter|onnx_inference_core)' \
  --output-on-failure \
  --verbose \
  2>&1 | tee "${LOG_ROOT}/ctest.log"

if ! grep -Eq 'tests failed out of ([2-9]|[1-9][0-9]+)' "${LOG_ROOT}/ctest.log"; then
  echo "ERROR: fewer than two required C++ transport tests were executed." >&2
  exit 1
fi

run_capture() {
  local lane="$1"
  local output_name="$2"
  local profile_prefix="$3"
  local nsys_prefix="$4"

  CAPTURE_OUTPUT_ROOT="${BAG_ROOT}" \
  CAPTURE_ORT_PROFILE_PREFIX="${profile_prefix}" \
  CAPTURE_ORT_PROFILE_FRAMES="${ORT_PROFILE_FRAMES}" \
  CAPTURE_NSYS_OUTPUT="${nsys_prefix}" \
  CAPTURE_STOP_GRACE_SECONDS=60 \
  CAPTURE_STOP_TERM_SECONDS=20 \
  "${CAPTURE_RUNNER}" \
    "${lane}" \
    "${output_name}"
}

echo "Capturing Config C (ORT CUDA + NITROS)..."
run_capture \
  yolov8-c \
  config_c \
  "${ORT_ROOT}/config_c_" \
  "${NSYS_ROOT}/config_c"

echo "Capturing Managed (ORT CUDA + Managed bridges)..."
run_capture \
  yolov8-managed \
  managed \
  "${ORT_ROOT}/managed_" \
  "${NSYS_ROOT}/managed"

find_single_profile() {
  local pattern="$1"
  local label="$2"
  local matches=()

  mapfile -t matches < <(find "${ORT_ROOT}" -maxdepth 1 -type f -name "${pattern}" -print)
  if [[ ${#matches[@]} -ne 1 ]]; then
    echo "ERROR: expected one ${label} ORT profile, found ${#matches[@]}." >&2
    return 1
  fi
  echo "${matches[0]}"
}

CONFIG_C_PROFILE="$(find_single_profile 'config_c_*.json' 'Config C')"
MANAGED_PROFILE="$(find_single_profile 'managed_*.json' 'Managed')"
CONFIG_C_REP="${NSYS_ROOT}/config_c.nsys-rep"
MANAGED_REP="${NSYS_ROOT}/managed.nsys-rep"

for trace_path in "${CONFIG_C_REP}" "${MANAGED_REP}"; do
  if [[ ! -s ${trace_path} ]]; then
    echo "ERROR: Nsight trace is missing or empty: ${trace_path}" >&2
    exit 1
  fi
done

bag_message_count() {
  ros2 bag info "$1" | awk '/^Messages:/ {print $2; exit}'
}

CONFIG_C_FRAMES="$(bag_message_count "${BAG_ROOT}/config_c")"
MANAGED_FRAMES="$(bag_message_count "${BAG_ROOT}/managed")"

if [[ ! ${CONFIG_C_FRAMES} =~ ^[1-9][0-9]*$ || ! ${MANAGED_FRAMES} =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: failed to read positive frame counts from the captured bags." >&2
  exit 1
fi

echo "Exporting CUDA trace tables..."
nsys stats \
  --quiet \
  --report cuda_gpu_trace:base \
  --format json:mem=B \
  --output - \
  "${CONFIG_C_REP}" \
  >"${NSYS_ROOT}/config_c_cuda_gpu_trace.json"

nsys stats \
  --quiet \
  --report cuda_gpu_trace:base \
  --format json:mem=B \
  --output - \
  "${MANAGED_REP}" \
  >"${NSYS_ROOT}/managed_cuda_gpu_trace.json"

echo "Comparing ONNX Runtime provider-to-node placement..."
ros2 run isaac_ros_onnx_inference summarize_ort_profile.py \
  "${CONFIG_C_PROFILE}" \
  "${MANAGED_PROFILE}" \
  --expected-provider CUDAExecutionProvider \
  --require-same-provider-layout \
  --output-json "${REPORT_ROOT}/ort_provider_layout.json" \
  2>&1 | tee "${LOG_ROOT}/ort_provider_layout.log"

echo "Comparing CUDA kernels and memory operations..."
ros2 run isaac_ros_onnx_inference compare_nsys_cuda_traces.py \
  --config-c-trace "${NSYS_ROOT}/config_c_cuda_gpu_trace.json" \
  --managed-trace "${NSYS_ROOT}/managed_cuda_gpu_trace.json" \
  --config-c-frames "${CONFIG_C_FRAMES}" \
  --managed-frames "${MANAGED_FRAMES}" \
  --payload-size "${YOLO_INPUT_BYTES}" \
  --payload-size "${YOLO_OUTPUT_BYTES}" \
  --output-json "${REPORT_ROOT}/cuda_trace_comparison.json" \
  2>&1 | tee "${LOG_ROOT}/cuda_trace_comparison.log"

{
  echo "YOLOv8 transport audit: PASS"
  echo "Config C captured frames: ${CONFIG_C_FRAMES}"
  echo "Managed captured frames: ${MANAGED_FRAMES}"
  echo "ORT profile frames per lane: ${ORT_PROFILE_FRAMES}"
  echo "Input tensor payload bytes: ${YOLO_INPUT_BYTES}"
  echo "Output tensor payload bytes: ${YOLO_OUTPUT_BYTES}"
  echo "Bridge criterion: no new CUDA kernel name, memcpy signature, or payload copy rate"
  echo "Decoder D2H copies are expected in both lanes and reported separately"
} | tee "${AUDIT_ROOT}/summary.txt"

echo "PASS: transport audit written to ${AUDIT_ROOT}"
