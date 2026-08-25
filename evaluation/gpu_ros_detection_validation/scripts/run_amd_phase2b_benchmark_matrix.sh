#!/usr/bin/env bash
# Copyright 2026 Boshen Chen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

set -euo pipefail

usage() {
  echo "Usage: $0 <yolov8|rtdetr> <matrix-name>"
  echo
  echo "Run three independent processes for std, staged-control and direct Managed lanes."
  echo "The lane order rotates on every round and each graph includes fixed 10/30/60 Hz trials."
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
MATRIX_NAME="$2"
case "${MODEL}" in
  yolov8)
    GRAPH_STD="gpu_ros_yolov8_phase2a_amd_graph.py"
    GRAPH_STAGED="gpu_ros_yolov8_phase2b_amd_staged_control_graph.py"
    GRAPH_DIRECT="gpu_ros_yolov8_phase2b_amd_managed_graph.py"
    ;;
  rtdetr)
    GRAPH_STD="gpu_ros_rtdetr_phase2a_amd_graph.py"
    GRAPH_STAGED="gpu_ros_rtdetr_phase2b_amd_staged_control_graph.py"
    GRAPH_DIRECT="gpu_ros_rtdetr_phase2b_amd_managed_graph.py"
    ;;
  *)
    echo "ERROR: model must be yolov8 or rtdetr, got '${MODEL}'." >&2
    exit 2
    ;;
esac
if [[ ! ${MATRIX_NAME} =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "ERROR: matrix-name may contain only letters, numbers, '.', '_' and '-'" >&2
  exit 2
fi

WORKSPACE_ROOT="${OVG_WORKSPACE_ROOT:-/workspaces/amd_ros_object_detection}"
ASSETS_ROOT="${OVG_ASSETS_ROOT:-${ROS2_BENCHMARK_OVERRIDE_ASSETS_ROOT:-/workspaces/ovg-assets}}"
RESULT_PARENT="${OVG_RESULTS_ROOT:-/workspaces/ovg-results}/phase2b-benchmark-matrix"
OUTPUT_ROOT="${RESULT_PARENT}/${MATRIX_NAME}"
BENCHMARK_ROOT="${WORKSPACE_ROOT}/migrated_packages/benchmarks"
GPU_MANAGED_ROOT="${GPU_ROS_MANAGED_ROOT:-${WORKSPACE_ROOT}/src/gpu_ros_managed}"
if [[ ${MODEL} == yolov8 ]]; then
  MODEL_PATH="${CAPTURE_MODEL_PATH:-${ASSETS_ROOT}/models/yolov8/yolov8s.onnx}"
else
  MODEL_PATH="${CAPTURE_MODEL_PATH:-${ASSETS_ROOT}/models/rtdetrv2_r50/rtdetrv2_r50.onnx}"
fi
DATASET_PATH="${CAPTURE_INPUT_BAG:-${ASSETS_ROOT}/datasets/r2bdataset2024_v1/r2b_robotarm}"
ROS_SETUP="/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
RUN_LOG_ROOT="${OUTPUT_ROOT}/logs"
ACTIVE_PID=""

if [[ -e ${OUTPUT_ROOT} ]]; then
  echo "ERROR: refusing to overwrite existing matrix directory: ${OUTPUT_ROOT}" >&2
  exit 1
fi
if [[ ! -d ${BENCHMARK_ROOT} ]]; then
  echo "ERROR: benchmark directory is missing: ${BENCHMARK_ROOT}" >&2
  exit 1
fi
for graph in "${GRAPH_STD}" "${GRAPH_STAGED}" "${GRAPH_DIRECT}"; do
  if [[ ! -f ${BENCHMARK_ROOT}/${graph} ]]; then
    echo "ERROR: benchmark graph is missing: ${BENCHMARK_ROOT}/${graph}" >&2
    exit 1
  fi
done

if [[ -f ${ROS_SETUP} ]]; then
  set +u
  # shellcheck disable=SC1090
  source "${ROS_SETUP}"
  set -u
fi
if [[ -f /opt/ros2_benchmark/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1090
  source /opt/ros2_benchmark/setup.bash
  set -u
fi
if [[ -f ${WORKSPACE_ROOT}/install/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1090
  source "${WORKSPACE_ROOT}/install/setup.bash"
  set -u
fi

for command_name in awk date env find git launch_test mkdir sha256sum sort tee xargs; do
  if ! command -v "${command_name}" >/dev/null; then
    echo "ERROR: required command is unavailable: ${command_name}" >&2
    exit 1
  fi
done

mkdir -p "${RUN_LOG_ROOT}"

hash_path() {
  local path="$1"
  if [[ -f ${path} ]]; then
    sha256sum "${path}" | awk '{print $1}'
    return 0
  fi
  if [[ -d ${path} ]]; then
    find "${path}" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}'
    return 0
  fi
  echo "MISSING"
}

repo_revision() {
  local path="$1"
  if [[ -d ${path}/.git ]]; then
    git -C "${path}" rev-parse HEAD
  else
    echo "MISSING"
  fi
}

repo_diff_hash() {
  local path="$1"
  if [[ -d ${path}/.git ]]; then
    git -C "${path}" diff HEAD --binary | sha256sum | awk '{print $1}'
  else
    echo "MISSING"
  fi
}

repo_untracked_paths() {
  local path="$1"
  if [[ ! -d ${path}/.git ]]; then
    echo "MISSING"
    return 0
  fi
  local item first=1
  while IFS= read -r -d '' item; do
    ((first == 1)) || printf ';'
    printf '%q' "${item}"
    first=0
  done < <(git -C "${path}" ls-files --others --exclude-standard -z | sort -z)
  printf '\n'
}

repo_untracked_content_hash() {
  local path="$1"
  if [[ ! -d ${path}/.git ]]; then
    echo "MISSING"
    return 0
  fi
  (
    cd "${path}"
    while IFS= read -r -d '' item; do
      printf '%s\0' "${item}"
      sha256sum -- "${item}"
    done < <(git ls-files --others --exclude-standard -z | sort -z)
  ) | sha256sum | awk '{print $1}'
}

repo_dirty() {
  local path="$1"
  if [[ ! -d ${path}/.git ]]; then
    echo "MISSING"
  elif [[ -n $(git -C "${path}" status --porcelain) ]]; then
    echo "true"
  else
    echo "false"
  fi
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  set +e
  if [[ -n ${ACTIVE_PID} ]]; then
    kill -INT "${ACTIVE_PID}" 2>/dev/null || true
    wait "${ACTIVE_PID}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

{
  echo "schema_version=2"
  echo "model=${MODEL}"
  echo "matrix_name=${MATRIX_NAME}"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "workspace_root=${WORKSPACE_ROOT}"
  echo "assets_root=${ASSETS_ROOT}"
  echo "model_path=${MODEL_PATH}"
  echo "model_sha256=$(hash_path "${MODEL_PATH}")"
  echo "dataset_path=${DATASET_PATH}"
  echo "dataset_tree_sha256=$(hash_path "${DATASET_PATH}")"
  echo "ort_root=${ONNXRUNTIME_ROOT:-}"
  echo "ort_library_sha256=$(hash_path "${ONNXRUNTIME_LIBRARY:-${ONNXRUNTIME_ROOT:-}/lib/libonnxruntime.so}")"
  echo "application_revision=$(git -C "${WORKSPACE_ROOT}" rev-parse HEAD)"
  echo "application_diff_head_binary_sha256=$(repo_diff_hash "${WORKSPACE_ROOT}")"
  echo "application_untracked_paths=$(repo_untracked_paths "${WORKSPACE_ROOT}")"
  echo "application_untracked_content_sha256=$(repo_untracked_content_hash "${WORKSPACE_ROOT}")"
  echo "application_dirty=$(repo_dirty "${WORKSPACE_ROOT}")"
  echo "gpu_ros_managed_revision=$(repo_revision "${GPU_MANAGED_ROOT}")"
  echo "gpu_ros_managed_diff_head_binary_sha256=$(repo_diff_hash "${GPU_MANAGED_ROOT}")"
  echo "gpu_ros_managed_untracked_paths=$(repo_untracked_paths "${GPU_MANAGED_ROOT}")"
  echo "gpu_ros_managed_untracked_content_sha256=$(repo_untracked_content_hash "${GPU_MANAGED_ROOT}")"
  echo "gpu_ros_managed_dirty=$(repo_dirty "${GPU_MANAGED_ROOT}")"
  echo "fixed_rates_hz=10,30,60"
  echo "round_count=3"
  echo "lane_order_round_1=std,staged,direct"
  echo "lane_order_round_2=staged,direct,std"
  echo "lane_order_round_3=direct,std,staged"
} >"${OUTPUT_ROOT}/matrix_manifest.txt"

run_lane() {
  local round="$1"
  local lane="$2"
  local graph="$3"
  local result_file="${OUTPUT_ROOT}/round-${round}-${lane}.json"
  local log_file="${RUN_LOG_ROOT}/round-${round}-${lane}.log"
  if [[ -e ${result_file} || -e ${log_file} ]]; then
    echo "ERROR: refusing to overwrite lane output: ${lane} round ${round}" >&2
    return 1
  fi

  echo "Running round ${round}, lane ${lane}: ${graph}"
  {
    echo "round=${round}"
    echo "lane=${lane}"
    echo "graph=${graph}"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "command=R2B_RESULT_FILE=${result_file} launch_test ${BENCHMARK_ROOT}/${graph}"
  } >>"${OUTPUT_ROOT}/matrix_manifest.txt"

  set +e
  env R2B_RESULT_FILE="${result_file}" \
    launch_test "${BENCHMARK_ROOT}/${graph}" >"${log_file}" 2>&1 &
  ACTIVE_PID=$!
  wait "${ACTIVE_PID}"
  local status=$?
  ACTIVE_PID=""
  set -e
  if ((status != 0)); then
    echo "ERROR: ${lane} round ${round} failed; inspect ${log_file}." >&2
    return "${status}"
  fi
  if [[ ! -s ${result_file} ]]; then
    echo "ERROR: ${lane} round ${round} produced no result JSON: ${result_file}" >&2
    return 1
  fi
  echo "completed_at=$(date --iso-8601=seconds)" >>"${OUTPUT_ROOT}/matrix_manifest.txt"
}

run_lane 1 std "${GRAPH_STD}"
run_lane 1 staged "${GRAPH_STAGED}"
run_lane 1 direct "${GRAPH_DIRECT}"
run_lane 2 staged "${GRAPH_STAGED}"
run_lane 2 direct "${GRAPH_DIRECT}"
run_lane 2 std "${GRAPH_STD}"
run_lane 3 direct "${GRAPH_DIRECT}"
run_lane 3 std "${GRAPH_STD}"
run_lane 3 staged "${GRAPH_STAGED}"

echo "completed_at=$(date --iso-8601=seconds)" >>"${OUTPUT_ROOT}/matrix_manifest.txt"
trap - EXIT INT TERM
echo "PASS: ${MODEL} Phase 2B benchmark matrix written to ${OUTPUT_ROOT}"
