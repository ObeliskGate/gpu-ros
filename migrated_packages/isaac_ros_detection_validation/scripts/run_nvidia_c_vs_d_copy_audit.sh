#!/usr/bin/env bash
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

set -euo pipefail

usage() {
  echo "Usage: $0 <yolov8|rtdetr> <audit-name>"
  echo
  echo "Run a fixed-input NVIDIA Config C versus Config D copy-delta experiment."
  echo "The output directory must not already exist."
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 2 ]]; then
  usage >&2
  exit 2
fi

MODEL="$1"
AUDIT_NAME="$2"
case "${MODEL}" in
  yolov8 | rtdetr) ;;
  *)
    echo "ERROR: model must be yolov8 or rtdetr, got '${MODEL}'." >&2
    exit 2
    ;;
esac
if [[ ! ${AUDIT_NAME} =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "ERROR: audit-name may contain only letters, numbers, '.', '_' and '-'" >&2
  exit 2
fi

WORKSPACE_ROOT="${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}"
APP_ROOT="${AMD_ROS_OBJECT_DETECTION_ROOT:-${WORKSPACE_ROOT}/src/amd_ros_object_detection}"
RESULTS_ROOT="${OVG_RESULTS_ROOT:-/workspaces/ovg-results}"
AUDIT_PARENT="${CAPTURE_C_VS_D_ROOT:-${RESULTS_ROOT}/phase2b_c_vs_d}"
AUDIT_ROOT="${AUDIT_PARENT}/${AUDIT_NAME}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CAPTURE_RUNNER="${SCRIPT_DIR}/run_nvidia_fixed_input_capture.sh"
ORT_ROOT="${AUDIT_ROOT}/ort"
NSYS_ROOT="${AUDIT_ROOT}/nsys"
BAG_ROOT="${AUDIT_ROOT}/bags"
BINDING_ROOT="${AUDIT_ROOT}/binding"
REPORT_ROOT="${AUDIT_ROOT}/reports"
LOG_ROOT="${AUDIT_ROOT}/logs"
ORT_PROFILE_FRAMES="${AUDIT_ORT_PROFILE_FRAMES:-50}"

if [[ ${MODEL} == yolov8 ]]; then
  REFERENCE_LANE="yolov8-c"
  CANDIDATE_LANE="yolov8-d"
else
  REFERENCE_LANE="rtdetr-c"
  CANDIDATE_LANE="rtdetr-d"
fi

if [[ -e ${AUDIT_ROOT} ]]; then
  echo "ERROR: refusing to overwrite existing audit path: ${AUDIT_ROOT}" >&2
  exit 1
fi
if [[ ! ${ORT_PROFILE_FRAMES} =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: AUDIT_ORT_PROFILE_FRAMES must be a positive integer." >&2
  exit 2
fi
if [[ ! -x ${CAPTURE_RUNNER} ]]; then
  echo "ERROR: capture runner is missing or not executable: ${CAPTURE_RUNNER}" >&2
  exit 1
fi

ROS_SETUP="/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
if [[ -f ${ROS_SETUP} ]]; then
  set +u
  # shellcheck disable=SC1090
  source "${ROS_SETUP}"
  set -u
fi
if [[ -f ${WORKSPACE_ROOT}/install/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1090
  source "${WORKSPACE_ROOT}/install/setup.bash"
  set -u
fi

for command_name in awk ctest find grep nsys ros2 tee; do
  if ! command -v "${command_name}" >/dev/null; then
    echo "ERROR: required command is unavailable: ${command_name}" >&2
    exit 1
  fi
done
if [[ ! -d ${WORKSPACE_ROOT}/build/isaac_ros_onnx_inference ]]; then
  echo "ERROR: build isaac_ros_onnx_inference before running the audit." >&2
  exit 1
fi

mkdir -p "${ORT_ROOT}" "${NSYS_ROOT}" "${BAG_ROOT}" "${BINDING_ROOT}" \
  "${REPORT_ROOT}" "${LOG_ROOT}"

echo "Running pointer-identity, lifetime and CUDA I/O Binding tests..."
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
  local binding_path="$5"

  CAPTURE_OUTPUT_ROOT="${BAG_ROOT}" \
  CAPTURE_ORT_PROFILE_PREFIX="${profile_prefix}" \
  CAPTURE_ORT_PROFILE_FRAMES="${ORT_PROFILE_FRAMES}" \
  CAPTURE_BINDING_REPORT_PATH="${binding_path}" \
  CAPTURE_NSYS_OUTPUT="${nsys_prefix}" \
  CAPTURE_STOP_GRACE_SECONDS=60 \
  CAPTURE_STOP_TERM_SECONDS=20 \
  "${CAPTURE_RUNNER}" "${lane}" "${output_name}"
}

echo "Capturing ${MODEL} Config C (ORT CUDA + NITROS)..."
run_capture "${REFERENCE_LANE}" config_c \
  "${ORT_ROOT}/config_c_" "${NSYS_ROOT}/config_c" \
  "${BINDING_ROOT}/config_c.json"

echo "Capturing ${MODEL} Config D (ORT CUDA + standard ROS 2 TensorList)..."
run_capture "${CANDIDATE_LANE}" config_d \
  "${ORT_ROOT}/config_d_" "${NSYS_ROOT}/config_d" \
  "${BINDING_ROOT}/config_d.json"

find_single_profile() {
  local pattern="$1"
  local label="$2"
  local matches=()
  mapfile -t matches < <(
    find "${ORT_ROOT}" -maxdepth 1 -type f -name "${pattern}" -size +0c -print)
  if [[ ${#matches[@]} -ne 1 ]]; then
    echo "ERROR: expected one ${label} ORT profile, found ${#matches[@]}." >&2
    return 1
  fi
  echo "${matches[0]}"
}

CONFIG_C_PROFILE="$(find_single_profile 'config_c_*.json' 'Config C')"
CONFIG_D_PROFILE="$(find_single_profile 'config_d_*.json' 'Config D')"
CONFIG_C_REP="${NSYS_ROOT}/config_c.nsys-rep"
CONFIG_D_REP="${NSYS_ROOT}/config_d.nsys-rep"
for required_path in \
  "${CONFIG_C_REP}" "${CONFIG_D_REP}" \
  "${BINDING_ROOT}/config_c.json" "${BINDING_ROOT}/config_d.json"; do
  if [[ ! -s ${required_path} ]]; then
    echo "ERROR: required audit evidence is missing or empty: ${required_path}" >&2
    exit 1
  fi
done

bag_message_count() {
  ros2 bag info "$1" | awk '/^Messages:/ {print $2; exit}'
}

CONFIG_C_FRAMES="$(bag_message_count "${BAG_ROOT}/config_c")"
CONFIG_D_FRAMES="$(bag_message_count "${BAG_ROOT}/config_d")"
if [[ ! ${CONFIG_C_FRAMES} =~ ^[1-9][0-9]*$ || ! ${CONFIG_D_FRAMES} =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: failed to read positive frame counts from captured bags." >&2
  exit 1
fi

echo "Exporting CUDA memory-copy and kernel tables..."
nsys stats --quiet --report cuda_gpu_trace:base --format json:mem=B --output - \
  "${CONFIG_C_REP}" >"${NSYS_ROOT}/config_c_cuda_gpu_trace.json"
nsys stats --quiet --report cuda_gpu_trace:base --format json:mem=B --output - \
  "${CONFIG_D_REP}" >"${NSYS_ROOT}/config_d_cuda_gpu_trace.json"

echo "Auditing ONNX Runtime provider placement..."
ros2 run isaac_ros_onnx_inference summarize_ort_profile.py \
  "${CONFIG_C_PROFILE}" "${CONFIG_D_PROFILE}" \
  --expected-provider CUDAExecutionProvider \
  --ignore-cpu-provider-layout \
  --output-json "${REPORT_ROOT}/ort_provider_layout.json" \
  2>&1 | tee "${LOG_ROOT}/ort_provider_layout.log"

echo "Writing Config C and Config D self sections plus copy delta..."
ros2 run isaac_ros_onnx_inference compare_nvidia_copy_traces.py \
  --reference-trace "${NSYS_ROOT}/config_c_cuda_gpu_trace.json" \
  --candidate-trace "${NSYS_ROOT}/config_d_cuda_gpu_trace.json" \
  --reference-label config_c \
  --candidate-label config_d \
  --reference-frames "${CONFIG_C_FRAMES}" \
  --candidate-frames "${CONFIG_D_FRAMES}" \
  --reference-binding-report "${BINDING_ROOT}/config_c.json" \
  --candidate-binding-report "${BINDING_ROOT}/config_d.json" \
  --output-json "${REPORT_ROOT}/cuda_copy_delta.json" \
  2>&1 | tee "${LOG_ROOT}/cuda_copy_delta.log"

echo "Comparing Config C and Config D detection outputs with stamp matching..."
ros2 run isaac_ros_detection_validation compare_detection2d_bags.py \
  --reference-bag "${BAG_ROOT}/config_c" \
  --candidate-bag "${BAG_ROOT}/config_d" \
  --match-policy stamp \
  --output-json "${REPORT_ROOT}/detection_comparison.json" \
  2>&1 | tee "${LOG_ROOT}/detection_comparison.log"

COPY_STATUS="$(
  awk -F'"' '/"status":/ {print $4; exit}' "${REPORT_ROOT}/cuda_copy_delta.json"
)"
if [[ ${COPY_STATUS} != PASS ]]; then
  echo "ERROR: copy comparison evidence is incomplete (${COPY_STATUS:-missing})." >&2
  exit 1
fi

{
  echo "${MODEL} NVIDIA Config C versus Config D copy experiment: PASS"
  echo "Config C captured frames: ${CONFIG_C_FRAMES}"
  echo "Config D captured frames: ${CONFIG_D_FRAMES}"
  echo "ORT profile frames per lane: ${ORT_PROFILE_FRAMES}"
  echo "The machine-readable report contains both self sections and D-minus-C deltas."
  echo "Positive H2D/D2H deltas are direct evidence of additional explicit host/device copies in D."
  echo "D2D deltas are reported separately and are not conflated with H2D/D2H."
  echo "Kernel-name differences are diagnostic and are not counted as copies by name alone."
  echo "The capture includes graph startup and drain; interpret absolute totals with per-frame deltas."
  echo "This is a C-vs-D transfer diagnostic, not a Managed zero-copy closure verdict."
} | tee "${AUDIT_ROOT}/summary.txt"

echo "PASS: C-vs-D copy experiment written to ${AUDIT_ROOT}"
