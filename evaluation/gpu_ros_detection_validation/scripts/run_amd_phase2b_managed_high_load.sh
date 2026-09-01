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
  echo "Usage: $0 <yolov8|rtdetr> <run-name>"
  echo
  echo "Run the direct Managed HIP production graph for at least 10 minutes and 10,000 inputs."
  echo "The output directory must not already exist."
  echo
  echo "Overrides:"
  echo "  PHASE2B_HIGH_LOAD_DURATION_SECONDS (default: 600)"
  echo "  PHASE2B_HIGH_LOAD_MIN_INPUT_MESSAGES (default: 10000)"
  echo "  PHASE2B_HIGH_LOAD_PLAYBACK_RATE (default: 1.0)"
  echo "  PHASE2B_HIGH_LOAD_TOPIC_WAIT_SECONDS (default: 900)"
  echo "  OVG_ASSETS_ROOT, OVG_RESULTS_ROOT, OVG_WORKSPACE_ROOT"
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
RUN_NAME="$2"
case "${MODEL}" in
  yolov8)
    LAUNCH_FILE="yolov8_ort_managed_amd.launch.py"
    NAMESPACE="yolov8_managed"
    MODEL_RELATIVE_PATH="models/yolov8/yolov8s.onnx"
    ;;
  rtdetr)
    LAUNCH_FILE="rtdetr_ort_managed_amd.launch.py"
    NAMESPACE="rtdetr_managed"
    MODEL_RELATIVE_PATH="models/rtdetrv2_r50/rtdetrv2_r50.onnx"
    ;;
  *)
    echo "ERROR: model must be yolov8 or rtdetr, got '${MODEL}'." >&2
    exit 2
    ;;
esac
if [[ ! ${RUN_NAME} =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "ERROR: run-name may contain only letters, numbers, '.', '_' and '-'" >&2
  exit 2
fi

WORKSPACE_ROOT="${OVG_WORKSPACE_ROOT:-/workspaces/amd_ros_object_detection}"
ASSETS_ROOT="${OVG_ASSETS_ROOT:-${ROS2_BENCHMARK_OVERRIDE_ASSETS_ROOT:-/workspaces/ovg-assets}}"
RESULT_ROOT="${OVG_RESULTS_ROOT:-/workspaces/ovg-results}/phase2b-high-load"
OUTPUT_ROOT="${RESULT_ROOT}/${RUN_NAME}"
COUNTER_STOP_FILE="${OUTPUT_ROOT}/input_counter.stop"
INPUT_BAG="${CAPTURE_INPUT_BAG:-${ASSETS_ROOT}/datasets/r2bdataset2024_v1/r2b_robotarm}"
IMAGE_TOPIC="${CAPTURE_IMAGE_TOPIC:-/camera_1/color/image_raw}"
MODEL_PATH="${CAPTURE_MODEL_PATH:-${ASSETS_ROOT}/${MODEL_RELATIVE_PATH}}"
DURATION_SECONDS="${PHASE2B_HIGH_LOAD_DURATION_SECONDS:-600}"
MIN_INPUT_MESSAGES="${PHASE2B_HIGH_LOAD_MIN_INPUT_MESSAGES:-10000}"
PLAYBACK_RATE="${PHASE2B_HIGH_LOAD_PLAYBACK_RATE:-1.0}"
DRAIN_SECONDS="${PHASE2B_HIGH_LOAD_DRAIN_SECONDS:-10}"
TOPIC_WAIT_TIMEOUT_SECONDS="${PHASE2B_HIGH_LOAD_TOPIC_WAIT_SECONDS:-900}"
PROFILE_PREFIX="${PHASE2B_HIGH_LOAD_ORT_PROFILE_PREFIX:-${OUTPUT_ROOT}/ort/profile}"
BINDING_REPORT="${PHASE2B_HIGH_LOAD_BINDING_REPORT:-${OUTPUT_ROOT}/binding.json}"
DETECTION_TOPIC=""
DETECTION_CANDIDATES=(
  /detections_output
  "/${NAMESPACE}/detections_output"
)
LAUNCH_PID=""
COUNTER_PID=""
RECORDER_PID=""
PLAYBACK_PID=""

if [[ -e ${OUTPUT_ROOT} ]]; then
  echo "ERROR: refusing to overwrite high-load output: ${OUTPUT_ROOT}" >&2
  exit 1
fi
if [[ ! -s ${MODEL_PATH} ]]; then
  echo "ERROR: model is missing: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -d ${INPUT_BAG} ]]; then
  echo "ERROR: input bag is missing: ${INPUT_BAG}" >&2
  exit 1
fi
if [[ ! ${DURATION_SECONDS} =~ ^[0-9]+$ ]] || ((DURATION_SECONDS < 600)); then
  echo "ERROR: PHASE2B_HIGH_LOAD_DURATION_SECONDS must be at least 600." >&2
  exit 2
fi
if [[ ! ${MIN_INPUT_MESSAGES} =~ ^[0-9]+$ ]] || ((MIN_INPUT_MESSAGES < 10000)); then
  echo "ERROR: PHASE2B_HIGH_LOAD_MIN_INPUT_MESSAGES must be at least 10000." >&2
  exit 2
fi
if [[ ! ${DRAIN_SECONDS} =~ ^[0-9]+$ ]]; then
  echo "ERROR: PHASE2B_HIGH_LOAD_DRAIN_SECONDS must be a non-negative integer." >&2
  exit 2
fi
if [[ ! ${TOPIC_WAIT_TIMEOUT_SECONDS} =~ ^[0-9]+$ ]] ||
  ((TOPIC_WAIT_TIMEOUT_SECONDS < 1)); then
  echo "ERROR: PHASE2B_HIGH_LOAD_TOPIC_WAIT_SECONDS must be a positive integer." >&2
  exit 2
fi
if [[ ! ${PLAYBACK_RATE} =~ ^[0-9]+([.][0-9]+)?$ ]] ||
  [[ ${PLAYBACK_RATE} == 0 || ${PLAYBACK_RATE} == 0.0 ]]; then
  echo "ERROR: PHASE2B_HIGH_LOAD_PLAYBACK_RATE must be positive." >&2
  exit 2
fi

ROS_SETUP="/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
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

for command_name in awk date find git grep mkdir ros2 sha256sum sleep sort tail timeout xargs; do
  if ! command -v "${command_name}" >/dev/null; then
    echo "ERROR: required command is unavailable: ${command_name}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/ort" \
  "$(dirname "${PROFILE_PREFIX}")" "$(dirname "${BINDING_REPORT}")"
STARTUP_LOG="${OUTPUT_ROOT}/logs/startup.log"
CURRENT_PHASE="initializing"

startup_log() {
  local message="$*"
  printf '%s %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "${message}" \
    >>"${STARTUP_LOG}"
  echo "${message}"
}

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
    git -C "${path}" diff --binary | sha256sum | awk '{print $1}'
  else
    echo "MISSING"
  fi
}

ORT_LIBRARY_PATH="${ONNXRUNTIME_LIBRARY:-${ONNXRUNTIME_ROOT:-}/lib/libonnxruntime.so}"
GPU_MANAGED_ROOT="${GPU_ROS_MANAGED_ROOT:-${WORKSPACE_ROOT}/src/gpu_ros_managed}"
COUNTER_EXECUTABLE="${WORKSPACE_ROOT}/install/lib/gpu_ros_detection_validation/count_ros_messages.py"
for target in \
  "${OUTPUT_ROOT}/input_counter.json" \
  "${COUNTER_STOP_FILE}" \
  "${OUTPUT_ROOT}/detections" \
  "${BINDING_REPORT}" \
  "${OUTPUT_ROOT}/run_config.txt"; do
  if [[ -e ${target} ]]; then
    echo "ERROR: refusing to overwrite: ${target}" >&2
    exit 1
  fi
done
if [[ ! -x ${COUNTER_EXECUTABLE} ]]; then
  echo "ERROR: input counter executable is missing or not executable: ${COUNTER_EXECUTABLE}" >&2
  exit 1
fi

stop_process() {
  local pid="$1"
  local label="$2"
  [[ -n ${pid} ]] || return 0
  if ! kill -0 "${pid}" 2>/dev/null; then
    wait "${pid}" 2>/dev/null || true
    return 0
  fi
  echo "Stopping ${label} (PID ${pid})..."
  kill -INT "${pid}" 2>/dev/null || true
  for _ in {1..20}; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" 2>/dev/null || true
      return 0
    fi
    sleep 0.5
  done
  kill -TERM "${pid}" 2>/dev/null || true
  for _ in {1..20}; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" 2>/dev/null || true
      return 0
    fi
    sleep 0.5
  done
  kill -KILL "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
}

cleanup() {
  local status=$?
  if [[ -n ${STARTUP_LOG:-} ]]; then
    startup_log "cleanup status=${status} phase=${CURRENT_PHASE:-unknown}"
  fi
  trap - EXIT INT TERM
  set +e
  stop_process "${PLAYBACK_PID}" "playback"
  stop_process "${RECORDER_PID}" "detection recorder"
  if [[ -n ${COUNTER_PID:-} && -d ${OUTPUT_ROOT:-} ]]; then
    : >"${COUNTER_STOP_FILE}"
  fi
  stop_process "${COUNTER_PID}" "input counter"
  stop_process "${LAUNCH_PID}" "Managed HIP graph"
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

LAUNCH_COMMAND=(
  ros2 launch gpu_ros_onnx_inference "${LAUNCH_FILE}"
  "model_file_path:=${MODEL_PATH}"
  "image_topic:=${IMAGE_TOPIC}"
  "namespace:=${NAMESPACE}"
  "ort_profile_prefix:=${PROFILE_PREFIX}"
  "binding_report_path:=${BINDING_REPORT}"
)
if [[ ${MODEL} == rtdetr ]]; then
  LAUNCH_COMMAND+=(
    input_image_width:=1280
    input_image_height:=720
    use_max_dim_for_orig_size:=true
  )
fi

{
  echo "schema_version=1"
  echo "model=${MODEL}"
  echo "model_path=${MODEL_PATH}"
  echo "model_sha256=$(hash_path "${MODEL_PATH}")"
  echo "input_bag=${INPUT_BAG}"
  echo "dataset_tree_sha256=$(hash_path "${INPUT_BAG}")"
  echo "input_topic=${IMAGE_TOPIC}"
  echo "detection_topic=auto"
  printf 'detection_topic_candidates='
  printf '%s,' "${DETECTION_CANDIDATES[@]}"
  echo
  echo "duration_seconds=${DURATION_SECONDS}"
  echo "minimum_input_messages=${MIN_INPUT_MESSAGES}"
  echo "playback_rate=${PLAYBACK_RATE}"
  echo "managed_io_contract=hip_managed_strict"
  echo "managed_pool_capacity=16"
  echo "managed_pool_wait_timeout_ms=100"
  echo "topic_wait_timeout_seconds=${TOPIC_WAIT_TIMEOUT_SECONDS}"
  echo "ort_root=${ONNXRUNTIME_ROOT:-}"
  echo "ort_library_sha256=$(hash_path "${ORT_LIBRARY_PATH}")"
  echo "application_revision=$(repo_revision "${WORKSPACE_ROOT}")"
  echo "application_worktree_diff_sha256=$(repo_diff_hash "${WORKSPACE_ROOT}")"
  echo "gpu_ros_managed_revision=$(repo_revision "${GPU_MANAGED_ROOT}")"
  echo "gpu_ros_managed_worktree_diff_sha256=$(repo_diff_hash "${GPU_MANAGED_ROOT}")"
  printf 'command='
  printf '%q ' "${LAUNCH_COMMAND[@]}"
  echo
} >"${OUTPUT_ROOT}/run_config.txt"

CURRENT_PHASE="starting graph"
startup_log "Starting direct Managed HIP graph..."
setsid bash -c 'trap - INT TERM; exec "$@"' high-load-graph \
  "${LAUNCH_COMMAND[@]}" >"${OUTPUT_ROOT}/logs/graph.log" 2>&1 &
LAUNCH_PID=$!

topic_count() {
  local topic="$1"
  local kind="$2"
  local count
  count="$(ros2 topic info "${topic}" 2>/dev/null |
    awk -v kind="${kind}" '$1 == kind && $2 == "count:" {print $3; exit}' || true)"
  if [[ ${count:-} =~ ^[0-9]+$ ]]; then
    printf '%s\n' "${count}"
  else
    printf '0\n'
  fi
}

write_startup_diagnostics() {
  local reason="$1"
  local topic="$2"
  {
    echo "reason=${reason}"
    echo "phase=${CURRENT_PHASE:-unknown}"
    echo "topic=${topic}"
    echo "topic_wait_timeout_seconds=${TOPIC_WAIT_TIMEOUT_SECONDS}"
    echo "--- topic info ---"
    ros2 topic info "${topic}" 2>&1 || true
    echo "--- graph log tail ---"
    tail -n 100 "${OUTPUT_ROOT}/logs/graph.log" 2>&1 || true
  } >>"${STARTUP_LOG}"
  cat "${STARTUP_LOG}" >&2
}

discover_detection_topic() {
  local attempts=$((TOPIC_WAIT_TIMEOUT_SECONDS * 2))
  local attempt
  local candidate
  local count
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    for candidate in "${DETECTION_CANDIDATES[@]}"; do
      count="$(topic_count "${candidate}" Publisher)"
      if ((count >= 1)); then
        DETECTION_TOPIC="${candidate}"
        startup_log "Ready: graph detection publisher on ${DETECTION_TOPIC} (Publisher count=${count})."
        printf 'resolved_detection_topic=%s\n' "${DETECTION_TOPIC}" \
          >>"${OUTPUT_ROOT}/run_config.txt"
        return 0
      fi
    done
    if ! kill -0 "${LAUNCH_PID}" 2>/dev/null; then
      echo "ERROR: process exited while waiting for a graph detection publisher." >&2
      write_startup_diagnostics \
        "process exited while waiting for a graph detection publisher" \
        "${DETECTION_CANDIDATES[0]}"
      return 1
    fi
    if ((attempt == 1 || attempt % 20 == 0)); then
      startup_log "Waiting for graph detection publisher (candidates: ${DETECTION_CANDIDATES[*]}; ${attempt}/${attempts})..."
    fi
    sleep 0.5
  done
  echo "ERROR: timed out waiting for a graph detection publisher after ${TOPIC_WAIT_TIMEOUT_SECONDS}s." >&2
  write_startup_diagnostics \
    "timeout waiting for a graph detection publisher" "${DETECTION_CANDIDATES[0]}"
  return 1
}

wait_for_topic() {
  local topic="$1"
  local kind="$2"
  local pid="$3"
  local label="${4:-${kind,,} on ${topic}}"
  local attempts=$((TOPIC_WAIT_TIMEOUT_SECONDS * 2))
  local attempt
  local count
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    count="$(topic_count "${topic}" "${kind}")"
    if ((count >= 1)); then
      startup_log "Ready: ${label} (${kind} count=${count})."
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "ERROR: process exited while waiting for ${label}." >&2
      write_startup_diagnostics "process exited while waiting for ${label}" "${topic}"
      return 1
    fi
    if ((attempt == 1 || attempt % 20 == 0)); then
      startup_log "Waiting for ${label} (${attempt}/${attempts}, ${count} found)..."
    fi
    sleep 0.5
  done
  echo "ERROR: timed out waiting for ${label} after ${TOPIC_WAIT_TIMEOUT_SECONDS}s." >&2
  write_startup_diagnostics "timeout waiting for ${label}" "${topic}"
  return 1
}

CURRENT_PHASE="waiting for graph input subscription"
wait_for_topic "${IMAGE_TOPIC}" Subscription "${LAUNCH_PID}" \
  "graph input subscription on ${IMAGE_TOPIC}"
CURRENT_PHASE="waiting for graph detection publisher"
discover_detection_topic

CURRENT_PHASE="starting input counter"
"${COUNTER_EXECUTABLE}" \
  --topic "${IMAGE_TOPIC}" \
  --output-json "${OUTPUT_ROOT}/input_counter.json" \
  --stop-file "${COUNTER_STOP_FILE}" \
  >"${OUTPUT_ROOT}/logs/input_counter.log" 2>&1 &
COUNTER_PID=$!
CURRENT_PHASE="waiting for input counter subscription"
wait_for_topic "${IMAGE_TOPIC}" Subscription "${COUNTER_PID}" \
  "input counter subscription on ${IMAGE_TOPIC}"

CURRENT_PHASE="starting detection recorder"
ros2 bag record --disable-keyboard-controls \
  --output "${OUTPUT_ROOT}/detections" \
  --topics "${DETECTION_TOPIC}" \
  >"${OUTPUT_ROOT}/logs/recorder.log" 2>&1 &
RECORDER_PID=$!
CURRENT_PHASE="waiting for detection recorder subscription"
wait_for_topic "${DETECTION_TOPIC}" Subscription "${RECORDER_PID}" \
  "detection recorder subscription on ${DETECTION_TOPIC}"

CURRENT_PHASE="running bounded high-load playback"
echo "Running bounded high-load playback for ${DURATION_SECONDS}s..."
set +e
timeout --signal=INT --kill-after=30 "${DURATION_SECONDS}s" \
  ros2 bag play "${INPUT_BAG}" --loop --rate "${PLAYBACK_RATE}" \
  --topics "${IMAGE_TOPIC}" >"${OUTPUT_ROOT}/logs/playback.log" 2>&1 &
PLAYBACK_PID=$!
wait "${PLAYBACK_PID}"
PLAYBACK_STATUS=$?
PLAYBACK_PID=""
set -e
if ((PLAYBACK_STATUS != 124 && PLAYBACK_STATUS != 130 && PLAYBACK_STATUS != 143)); then
  echo "ERROR: bounded playback exited unexpectedly with status ${PLAYBACK_STATUS}." >&2
  exit 1
fi

echo "Playback duration reached; draining for ${DRAIN_SECONDS}s..."
sleep "${DRAIN_SECONDS}"
stop_process "${RECORDER_PID}" "detection recorder"
RECORDER_PID=""
: >"${COUNTER_STOP_FILE}"
stop_process "${COUNTER_PID}" "input counter"
COUNTER_PID=""
stop_process "${LAUNCH_PID}" "Managed HIP graph"
LAUNCH_PID=""

if [[ ! -s ${OUTPUT_ROOT}/input_counter.json ]]; then
  echo "ERROR: input counter did not write a report." >&2
  exit 1
fi
INPUT_COUNT="$(awk -F: '/"message_count"/ {gsub(/[^0-9]/, "", $2); print $2; exit}' \
  "${OUTPUT_ROOT}/input_counter.json")"
if [[ ! ${INPUT_COUNT:-} =~ ^[0-9]+$ ]] || ((INPUT_COUNT < MIN_INPUT_MESSAGES)); then
  echo "ERROR: counted ${INPUT_COUNT:-0} input messages; required at least ${MIN_INPUT_MESSAGES}." >&2
  exit 1
fi

if [[ ! -d ${OUTPUT_ROOT}/detections ]]; then
  echo "ERROR: detection output bag was not created." >&2
  exit 1
fi
DETECTION_INFO="$(ros2 bag info "${OUTPUT_ROOT}/detections")"
echo "${DETECTION_INFO}" >"${OUTPUT_ROOT}/logs/detection_bag_info.txt"
DETECTION_COUNT="$(awk '/^Messages:/ {print $2; exit}' <<<"${DETECTION_INFO}")"
if [[ ! ${DETECTION_COUNT:-} =~ ^[0-9]+$ ]] ||
  ((DETECTION_COUNT != INPUT_COUNT)); then
  echo "ERROR: output messages=${DETECTION_COUNT:-0} do not equal input messages=${INPUT_COUNT}." >&2
  exit 1
fi
if [[ ! -s ${BINDING_REPORT} ]]; then
  echo "ERROR: strict Managed binding report was not produced: ${BINDING_REPORT}" >&2
  exit 1
fi
PROFILE_DIR="$(dirname "${PROFILE_PREFIX}")"
PROFILE_BASENAME="$(basename "${PROFILE_PREFIX}")"
PROFILE_JSON="$(find "${PROFILE_DIR}" -maxdepth 1 -type f \
  -name "${PROFILE_BASENAME}*.json" -size +0c -print -quit)"
if [[ -z ${PROFILE_JSON} ]]; then
  echo "ERROR: strict Managed ORT profile was not produced under ${PROFILE_DIR}." >&2
  exit 1
fi

if grep -Eiq \
  'hip.*(error|failed)|pool exhausted|did not drain|pending release|unhealthy|deadlock|unknown asynchronous' \
  "${OUTPUT_ROOT}/logs/graph.log"; then
  echo "ERROR: high-load graph log contains a HIP/lifecycle failure." >&2
  exit 1
fi

{
  echo "status=PASS"
  echo "input_messages=${INPUT_COUNT}"
  echo "output_messages=${DETECTION_COUNT}"
  echo "minimum_input_messages=${MIN_INPUT_MESSAGES}"
  echo "duration_seconds=${DURATION_SECONDS}"
  echo "detection_bag=${OUTPUT_ROOT}/detections"
  echo "binding_report=${BINDING_REPORT}"
  echo "ort_profile_prefix=${PROFILE_PREFIX}"
  echo "ort_profile=${PROFILE_JSON}"
} >"${OUTPUT_ROOT}/summary.txt"

trap - EXIT INT TERM
echo "PASS: direct Managed HIP high-load run written to ${OUTPUT_ROOT}"
