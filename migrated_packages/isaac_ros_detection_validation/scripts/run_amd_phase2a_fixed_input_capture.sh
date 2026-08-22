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
  echo "Usage: $0 <output-name>"
  echo "       $0 yolov8 <output-name>"
  echo
  echo "Launch, warm up and record AMD Phase 2A in one terminal (RT-DETR or YOLOv8)."
  echo "Set CAPTURE_TRANSPORT=managed for the Phase 2B Managed HIP lane."
  echo "By default detection output is recorded; CAPTURE_RECORD=0 enables profile-only playback."
  echo "The output name and its log files must not already exist."
  echo
  echo "Environment overrides:"
  echo "  CAPTURE_EXECUTION_PROVIDER=migraphx|cpu  (default: migraphx)"
  echo "  CAPTURE_TRANSPORT=std|managed            (default: std)"
  echo "  CAPTURE_WARMUP_ONLY=0|1                 (default: 0)"
  echo "  CAPTURE_PLAYBACK_RATE=<positive number> (default: 0.25)"
  echo "  CAPTURE_MIN_MESSAGES=<integer>          (default: 20)"
  echo "  CAPTURE_FIRST_OUTPUT_TIMEOUT_SECONDS=<integer> (default: 900)"
  echo "  CAPTURE_GRAPH_READY_TIMEOUT_SECONDS=<integer>  (default: 900)"
  echo "  CAPTURE_ORT_PROFILE_PREFIX=<absolute path prefix> (default: disabled)"
  echo "  CAPTURE_BINDING_REPORT_PATH=<absolute path>  (default: disabled)"
  echo "  CAPTURE_TRACE_ATTACH_READY_FILE=<path>       (default: disabled)"
  echo "  CAPTURE_TRACE_ATTACH_RELEASE_FILE=<path>     (default: disabled)"
  echo "  CAPTURE_TRACE_ATTACH_PLAYBACK_DONE_FILE=<path> (default: disabled)"
  echo "  CAPTURE_TRACE_ATTACH_DETACH_COMPLETE_FILE=<path> (default: disabled)"
  echo "  CAPTURE_RECORD=0|1                        (default: 1)"
  echo "  CAPTURE_OUTPUT_ROOT=<absolute path>"
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

PIPELINE="rtdetr"
if [[ $# -eq 1 ]]; then
  OUTPUT_NAME="$1"
elif [[ $# -eq 2 && $1 == "yolov8" ]]; then
  PIPELINE="yolov8"
  OUTPUT_NAME="$2"
else
  usage >&2
  exit 2
fi

if [[ ! ${OUTPUT_NAME} =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "ERROR: output-name may contain only letters, numbers, '.', '_' and '-'." >&2
  exit 2
fi

WORKSPACE_ROOT="${OVG_WORKSPACE_ROOT:-/workspaces/amd_ros_object_detection}"
ASSETS_ROOT="${OVG_ASSETS_ROOT:-${ROS2_BENCHMARK_OVERRIDE_ASSETS_ROOT:-/workspaces/ovg-assets}}"
RESULTS_ROOT="${OVG_RESULTS_ROOT:-/workspaces/ovg-results}"
INPUT_BAG="${CAPTURE_INPUT_BAG:-${ASSETS_ROOT}/datasets/r2bdataset2024_v1/r2b_robotarm}"
OUTPUT_ROOT="${CAPTURE_OUTPUT_ROOT:-${RESULTS_ROOT}/phase2a-bags}"
OUTPUT_PATH="${OUTPUT_ROOT}/${OUTPUT_NAME}"
LOG_ROOT="${OUTPUT_ROOT}/logs"
LAUNCH_LOG="${LOG_ROOT}/${OUTPUT_NAME}.launch.log"
RECORD_LOG="${LOG_ROOT}/${OUTPUT_NAME}.record.log"
PLAYBACK_LOG="${LOG_ROOT}/${OUTPUT_NAME}.playback.log"
WARMUP_LOG="${LOG_ROOT}/${OUTPUT_NAME}.warmup-playback.log"
WARMUP_OUTPUT="${LOG_ROOT}/${OUTPUT_NAME}.warmup-detection.yaml"
COMMAND_LOG="${LOG_ROOT}/${OUTPUT_NAME}.command.txt"
EXECUTION_PROVIDER="${CAPTURE_EXECUTION_PROVIDER:-migraphx}"
TRANSPORT="${CAPTURE_TRANSPORT:-std}"
WARMUP_ONLY="${CAPTURE_WARMUP_ONLY:-0}"
PLAYBACK_RATE="${CAPTURE_PLAYBACK_RATE:-0.25}"
MIN_MESSAGES="${CAPTURE_MIN_MESSAGES:-20}"
DRAIN_SECONDS="${CAPTURE_DRAIN_SECONDS:-10}"
INPUT_READY_TIMEOUT_SECONDS="${CAPTURE_INPUT_READY_TIMEOUT_SECONDS:-120}"
GRAPH_READY_TIMEOUT_SECONDS="${CAPTURE_GRAPH_READY_TIMEOUT_SECONDS:-900}"
FIRST_OUTPUT_TIMEOUT_SECONDS="${CAPTURE_FIRST_OUTPUT_TIMEOUT_SECONDS:-900}"
STOP_GRACE_SECONDS="${CAPTURE_STOP_GRACE_SECONDS:-10}"
STOP_TERM_SECONDS="${CAPTURE_STOP_TERM_SECONDS:-5}"
IMAGE_TOPIC="${CAPTURE_IMAGE_TOPIC:-/camera_1/color/image_raw}"
ORT_PROFILE_PREFIX="${CAPTURE_ORT_PROFILE_PREFIX:-}"
BINDING_REPORT_PATH="${CAPTURE_BINDING_REPORT_PATH:-}"
TRACE_ATTACH_READY_FILE="${CAPTURE_TRACE_ATTACH_READY_FILE:-}"
TRACE_ATTACH_RELEASE_FILE="${CAPTURE_TRACE_ATTACH_RELEASE_FILE:-}"
TRACE_ATTACH_PLAYBACK_DONE_FILE="${CAPTURE_TRACE_ATTACH_PLAYBACK_DONE_FILE:-}"
TRACE_ATTACH_DETACH_COMPLETE_FILE="${CAPTURE_TRACE_ATTACH_DETACH_COMPLETE_FILE:-}"
TRACE_ATTACH_TIMEOUT_SECONDS="${CAPTURE_TRACE_ATTACH_TIMEOUT_SECONDS:-900}"
RECORD_OUTPUT="${CAPTURE_RECORD:-1}"
DETECTION_TOPIC=""

case "${TRANSPORT}" in
  std)
    GRAPH_PACKAGE=""
    GRAPH_LAUNCH_FILE=""
    DEFAULT_NAMESPACE=""
    ;;
  managed)
    GRAPH_PACKAGE="isaac_ros_onnx_inference"
    GRAPH_LAUNCH_FILE=""
    DEFAULT_NAMESPACE=""
    ;;
  *)
    echo "ERROR: CAPTURE_TRANSPORT must be std or managed." >&2
    exit 2
    ;;
esac

if [[ ${PIPELINE} == "yolov8" ]]; then
  MODEL_PATH="${CAPTURE_MODEL_PATH:-${ASSETS_ROOT}/models/yolov8/yolov8s.onnx}"
  if [[ ${TRANSPORT} == "managed" ]]; then
    GRAPH_LAUNCH_FILE="yolov8_ort_managed_amd.launch.py"
    DEFAULT_NAMESPACE="yolov8_managed"
  else
    GRAPH_PACKAGE="isaac_ros_yolov8_std"
    GRAPH_LAUNCH_FILE="yolov8_ort_std_image.launch.py"
    DEFAULT_NAMESPACE="yolov8"
  fi
else
  MODEL_PATH="${CAPTURE_MODEL_PATH:-${ASSETS_ROOT}/models/synthetica_detr_v1.0.0_onnx/sdetr_grasp.onnx}"
  if [[ ${TRANSPORT} == "managed" ]]; then
    GRAPH_LAUNCH_FILE="rtdetr_ort_managed_amd.launch.py"
    DEFAULT_NAMESPACE="rtdetr_managed"
  else
    GRAPH_PACKAGE="isaac_ros_rtdetr_std"
    GRAPH_LAUNCH_FILE="rtdetr_ort_std_image.launch.py"
    DEFAULT_NAMESPACE="rtdetr"
  fi
fi
GRAPH_NAMESPACE="${CAPTURE_NAMESPACE:-${DEFAULT_NAMESPACE}}"
DETECTION_CANDIDATES=(
  /detections_output
  "/${GRAPH_NAMESPACE}/detections_output"
)

case "${EXECUTION_PROVIDER}" in
  migraphx | cpu) ;;
  *)
    echo "ERROR: CAPTURE_EXECUTION_PROVIDER must be migraphx or cpu." >&2
    exit 2
    ;;
esac

if [[ ${WARMUP_ONLY} != 0 && ${WARMUP_ONLY} != 1 ]]; then
  echo "ERROR: CAPTURE_WARMUP_ONLY must be 0 or 1." >&2
  exit 2
fi

if [[ ${RECORD_OUTPUT} != 0 && ${RECORD_OUTPUT} != 1 ]]; then
  echo "ERROR: CAPTURE_RECORD must be 0 or 1." >&2
  exit 2
fi

if [[ ! ${PLAYBACK_RATE} =~ ^[0-9]+([.][0-9]+)?$ ]] ||
  [[ ${PLAYBACK_RATE} == 0 || ${PLAYBACK_RATE} == 0.0 ]]; then
  echo "ERROR: CAPTURE_PLAYBACK_RATE must be a positive number." >&2
  exit 2
fi

for value_name in \
  MIN_MESSAGES \
  DRAIN_SECONDS \
  INPUT_READY_TIMEOUT_SECONDS \
  GRAPH_READY_TIMEOUT_SECONDS \
  FIRST_OUTPUT_TIMEOUT_SECONDS \
  STOP_GRACE_SECONDS \
  STOP_TERM_SECONDS; do
  value="${!value_name}"
  if [[ ! ${value} =~ ^[0-9]+$ ]]; then
    echo "ERROR: ${value_name} must be a non-negative integer." >&2
    exit 2
  fi
done

if [[ ! ${TRACE_ATTACH_TIMEOUT_SECONDS} =~ ^[0-9]+$ ]]; then
  echo "ERROR: CAPTURE_TRACE_ATTACH_TIMEOUT_SECONDS must be a non-negative integer." >&2
  exit 2
fi
if [[ -n ${TRACE_ATTACH_READY_FILE} && -z ${TRACE_ATTACH_RELEASE_FILE} ]]; then
  echo "ERROR: CAPTURE_TRACE_ATTACH_READY_FILE requires CAPTURE_TRACE_ATTACH_RELEASE_FILE." >&2
  exit 2
fi
if [[ -n ${TRACE_ATTACH_PLAYBACK_DONE_FILE} &&
  -z ${TRACE_ATTACH_DETACH_COMPLETE_FILE} ]]; then
  echo "ERROR: CAPTURE_TRACE_ATTACH_PLAYBACK_DONE_FILE requires " \
    "CAPTURE_TRACE_ATTACH_DETACH_COMPLETE_FILE." >&2
  exit 2
fi
if [[ -n ${TRACE_ATTACH_DETACH_COMPLETE_FILE} &&
  -z ${TRACE_ATTACH_PLAYBACK_DONE_FILE} ]]; then
  echo "ERROR: CAPTURE_TRACE_ATTACH_DETACH_COMPLETE_FILE requires " \
    "CAPTURE_TRACE_ATTACH_PLAYBACK_DONE_FILE." >&2
  exit 2
fi

if [[ ! -s ${MODEL_PATH} ]]; then
  if [[ ${PIPELINE} == "yolov8" ]]; then
    echo "YOLOv8 ONNX asset is missing:" >&2
    echo "${MODEL_PATH}" >&2
    echo "Provide OVG_YOLOV8_ONNX_SOURCE and run phase2 assets import-yolov8." >&2
  else
    echo "ERROR: model is missing or empty: ${MODEL_PATH}" >&2
  fi
  exit 1
fi

if [[ ! -f ${INPUT_BAG}/metadata.yaml ]]; then
  echo "ERROR: input bag is missing metadata.yaml: ${INPUT_BAG}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
if [[ -n ${ORT_PROFILE_PREFIX} ]]; then
  mkdir -p "$(dirname "${ORT_PROFILE_PREFIX}")"
fi
if [[ -n ${BINDING_REPORT_PATH} ]]; then
  mkdir -p "$(dirname "${BINDING_REPORT_PATH}")"
fi
for handshake_file in \
  "${TRACE_ATTACH_READY_FILE}" \
  "${TRACE_ATTACH_RELEASE_FILE}" \
  "${TRACE_ATTACH_PLAYBACK_DONE_FILE}" \
  "${TRACE_ATTACH_DETACH_COMPLETE_FILE}"; do
  if [[ -z ${handshake_file} ]]; then
    continue
  fi
  mkdir -p "$(dirname "${handshake_file}")"
  if [[ -e ${handshake_file} ]]; then
    echo "ERROR: refusing to overwrite trace attach handshake file: ${handshake_file}" >&2
    exit 1
  fi
done

TARGET_PATHS=(
  "${LAUNCH_LOG}" \
  "${RECORD_LOG}" \
  "${PLAYBACK_LOG}" \
  "${WARMUP_LOG}" \
  "${WARMUP_OUTPUT}" \
  "${COMMAND_LOG}"
)
if [[ -n ${BINDING_REPORT_PATH} ]]; then
  TARGET_PATHS+=("${BINDING_REPORT_PATH}")
fi
if [[ ${RECORD_OUTPUT} == 1 ]]; then
  TARGET_PATHS+=("${OUTPUT_PATH}")
fi

for target in "${TARGET_PATHS[@]}"; do
  if [[ -e ${target} ]]; then
    echo "ERROR: refusing to overwrite existing path: ${target}" >&2
    exit 1
  fi
done
if [[ -n ${ORT_PROFILE_PREFIX} ]]; then
  existing_profiles=()
  mapfile -t existing_profiles < <(
    find "$(dirname -- "${ORT_PROFILE_PREFIX}")" -maxdepth 1 -type f \
      -name "$(basename -- "${ORT_PROFILE_PREFIX}")*.json" -print)
  if [[ ${#existing_profiles[@]} -ne 0 ]]; then
    echo "ERROR: refusing to overwrite existing ORT profile output for prefix: ${ORT_PROFILE_PREFIX}" >&2
    exit 1
  fi
fi

source_setup() {
  local setup_file="$1"
  if [[ -f ${setup_file} ]]; then
    set +u
    # shellcheck disable=SC1090
    source "${setup_file}"
    set -u
  fi
}

source_setup "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
source_setup "/opt/ros2_benchmark/setup.bash"
source_setup "${WORKSPACE_ROOT}/install/setup.bash"

for command_name in awk find git ps ros2 setsid sha256sum sleep sort tail timeout xargs; do
  if ! command -v "${command_name}" >/dev/null; then
    echo "ERROR: required command is unavailable: ${command_name}" >&2
    exit 1
  fi
done

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

print_command() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

LAUNCH_PID=""
WARMUP_PID=""
RECORD_PID=""

child_pids() {
  local parent_pid="$1"
  ps -eo pid=,ppid= \
    | awk -v parent="${parent_pid}" '$2 == parent {print $1}'
}

collect_descendants() {
  local parent_pid="$1"
  local child_pid
  while IFS= read -r child_pid; do
    [[ -n ${child_pid} ]] || continue
    collect_descendants "${child_pid}"
    printf '%s\n' "${child_pid}"
  done < <(child_pids "${parent_pid}")
}

find_component_container_pid() {
  local candidate
  local command_line
  for candidate in "${LAUNCH_PID}" $(collect_descendants "${LAUNCH_PID}"); do
    [[ -n ${candidate} ]] || continue
    command_line="$(ps -o args= -p "${candidate}" 2>/dev/null || true)"
    if [[ ${command_line} == *component_container_mt* ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

process_alive() {
  local pid="$1"
  local state
  state="$(ps -o stat= -p "${pid}" 2>/dev/null || true)"
  state="${state//[[:space:]]/}"
  [[ -n ${state} && ${state:0:1} != Z ]]
}

descendant_tree_alive() {
  local descendant
  for descendant in "${descendants[@]}"; do
    if process_alive "${descendant}"; then
      return 0
    fi
  done
  return 1
}

stop_process() {
  local pid="$1"
  local label="$2"
  local attempt
  local process_group=""
  local descendants=()

  if [[ -z ${pid} ]]; then
    return
  fi

  process_alive "${pid}" || return
  process_group="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ' || true)"
  mapfile -t descendants < <(collect_descendants "${pid}")

  echo "Stopping ${label} (PID ${pid}, PGID ${process_group:-unknown})..."

  # setsid normally gives each capture child its own process group, but
  # ros2 launch may create additional process groups for the component
  # container. Signal the original group and the complete descendant tree.
  if [[ ${process_group:-} =~ ^[0-9]+$ && ${process_group} != 1 ]]; then
    kill -INT -- "-${process_group}" 2>/dev/null || true
  fi
  for descendant in "${descendants[@]}"; do
    kill -INT "${descendant}" 2>/dev/null || true
  done
  kill -INT "${pid}" 2>/dev/null || true

  for ((attempt = 1; attempt <= STOP_GRACE_SECONDS * 4; attempt++)); do
    if ! process_alive "${pid}" && ! descendant_tree_alive; then
      break
    fi
    sleep 0.25
  done

  if process_alive "${pid}" || descendant_tree_alive; then
    echo "${label} did not stop after SIGINT; sending SIGTERM..."
    if [[ ${process_group:-} =~ ^[0-9]+$ && ${process_group} != 1 ]]; then
      kill -TERM -- "-${process_group}" 2>/dev/null || true
    fi
    for descendant in "${descendants[@]}"; do
      kill -TERM "${descendant}" 2>/dev/null || true
    done
    kill -TERM "${pid}" 2>/dev/null || true
  fi

  for ((attempt = 1; attempt <= STOP_TERM_SECONDS * 4; attempt++)); do
    if ! process_alive "${pid}" && ! descendant_tree_alive; then
      break
    fi
    sleep 0.25
  done

  if process_alive "${pid}" || descendant_tree_alive; then
    echo "WARNING: ${label} ignored SIGTERM; sending SIGKILL." >&2
    if [[ ${process_group:-} =~ ^[0-9]+$ && ${process_group} != 1 ]]; then
      kill -KILL -- "-${process_group}" 2>/dev/null || true
    fi
    for descendant in "${descendants[@]}"; do
      kill -KILL "${descendant}" 2>/dev/null || true
    done
    kill -KILL "${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
}

stop_recorder() {
  local pid="$1"
  local attempt

  if [[ -z ${pid} ]] || ! process_alive "${pid}"; then
    return
  fi

  # rosbag2's stop service finalizes the storage plugin and writes
  # metadata.yaml. Signals remain a fallback for older or incomplete CLI
  # installations, but SIGKILL must not be the normal bag shutdown path.
  echo "Stopping recorder via /rosbag2_recorder/stop service..."
  if timeout --signal=TERM --kill-after=5s 60s \
    ros2 service call \
    /rosbag2_recorder/stop \
    rosbag2_interfaces/srv/Stop \
    "{}" >>"${RECORD_LOG}" 2>&1; then
    for ((attempt = 1; attempt <= STOP_GRACE_SECONDS * 4; attempt++)); do
      if ! process_alive "${pid}"; then
        return 0
      fi
      sleep 0.25
    done
    echo "Recorder stop service returned, but recorder is still alive; " \
      "falling back to signals." >&2
  else
    echo "WARNING: recorder stop service failed; falling back to signals." >&2
  fi

  stop_process "${pid}" "recorder"
}

cleanup() {
  local status=$?
  trap - EXIT
  set +e
  stop_recorder "${RECORD_PID}"
  stop_process "${WARMUP_PID}" "warm-up player"
  stop_process "${LAUNCH_PID}" "graph"
  wait_for_no_publishers "${DETECTION_TOPIC:-/detections_output}" 10 || true
  if [[ ${status} -ne 0 ]]; then
    echo "Capture failed. Inspect:"
    echo "  ${LAUNCH_LOG}"
    echo "  ${WARMUP_LOG}"
    echo "  ${RECORD_LOG}"
    echo "  ${PLAYBACK_LOG}"
  fi
  exit "${status}"
}

topic_count() {
  local topic="$1"
  local count_kind="$2"
  ros2 topic info "${topic}" 2>/dev/null |
    awk -v kind="${count_kind}" '$1 == kind && $2 == "count:" {print $3; exit}'
}

wait_for_no_publishers() {
  local topic="$1"
  local timeout_seconds="$2"
  local attempts=$((timeout_seconds * 2))
  local attempt
  local count

  for ((attempt = 1; attempt <= attempts; attempt++)); do
    count="$(topic_count "${topic}" "Publisher" || true)"
    if [[ ! ${count:-0} =~ ^[0-9]+$ ]] || ((count == 0)); then
      return 0
    fi
    sleep 0.5
  done

  echo "WARNING: publisher remains on ${topic} after capture cleanup." >&2
  ros2 topic info "${topic}" >&2 || true
  ps -eo pid=,ppid=,pgid=,stat=,args= \
    | awk '/ros2 launch isaac_ros_rtdetr_std|ros2 launch isaac_ros_yolov8_std|rtdetr_ort_managed_amd|yolov8_ort_managed_amd|component_container_mt|ros2 bag play/ {print}' \
    >&2 || true
  return 1
}

wait_for_profiler_detach() {
  local detach_complete_file="$1"
  local timeout_seconds="$2"
  local attempt
  local attempts=$((timeout_seconds * 2))

  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if [[ -e ${detach_complete_file} ]]; then
      return 0
    fi
    if ! process_alive "${LAUNCH_PID}"; then
      echo "ERROR: graph exited while waiting for profiler detach." >&2
      return 1
    fi
    sleep 0.5
  done

  echo "ERROR: profiler detach did not complete within ${timeout_seconds}s." >&2
  return 1
}

wait_for_topic_count() {
  local topic="$1"
  local count_kind="$2"
  local process_pid="$3"
  local process_label="$4"
  local timeout_seconds="$5"
  local attempts=$((timeout_seconds * 2))
  local attempt
  local count

  for ((attempt = 1; attempt <= attempts; attempt++)); do
    count="$(topic_count "${topic}" "${count_kind}" || true)"
    if [[ ${count:-0} =~ ^[0-9]+$ ]] && ((count >= 1)); then
      echo "${topic}: ${count_kind} count=${count}"
      return 0
    fi

    if ! kill -0 "${process_pid}" 2>/dev/null; then
      echo "ERROR: ${process_label} exited while waiting for ${topic}." >&2
      return 1
    fi

    if ((attempt % 20 == 0)); then
      echo "Waiting for ${count_kind,,} on ${topic}..."
    fi
    sleep 0.5
  done

  echo "ERROR: timed out waiting for ${count_kind,,} on ${topic}." >&2
  return 1
}

discover_detection_topic() {
  local attempts=$((GRAPH_READY_TIMEOUT_SECONDS * 2))
  local attempt
  local candidate
  local count

  for ((attempt = 1; attempt <= attempts; attempt++)); do
    for candidate in "${DETECTION_CANDIDATES[@]}"; do
      count="$(topic_count "${candidate}" "Publisher" || true)"
      if [[ ${count:-0} =~ ^[0-9]+$ ]] && ((count >= 1)); then
        DETECTION_TOPIC="${candidate}"
        echo "Discovered detection topic: ${DETECTION_TOPIC}"
        return 0
      fi
    done

    if ! kill -0 "${LAUNCH_PID}" 2>/dev/null; then
      echo "ERROR: graph exited before the detection publisher was ready." >&2
      return 1
    fi

    if ((attempt % 20 == 0)); then
      echo "Waiting for the detection publisher (MIGraphX may still be compiling)..."
    fi
    sleep 0.5
  done

  echo "ERROR: timed out waiting for a detection publisher." >&2
  return 1
}

for candidate in "${DETECTION_CANDIDATES[@]}"; do
  existing_publishers="$(topic_count "${candidate}" "Publisher" || true)"
  if [[ ${existing_publishers:-0} =~ ^[0-9]+$ ]] && ((existing_publishers >= 1)); then
    echo "ERROR: another detection graph is already publishing on ${candidate}." >&2
    echo "Stop it before starting an automated capture." >&2
    exit 1
  fi
done

trap cleanup EXIT
trap 'exit 130' INT TERM

if [[ ${PIPELINE} == "yolov8" ]]; then
  LAUNCH_COMMAND=(
    ros2 launch
    "${GRAPH_PACKAGE}"
    "${GRAPH_LAUNCH_FILE}"
    "model_file_path:=${MODEL_PATH}"
    "image_topic:=${IMAGE_TOPIC}"
    "namespace:=${GRAPH_NAMESPACE}"
    "confidence_threshold:=0.25"
    "nms_threshold:=0.45"
  )
else
  LAUNCH_COMMAND=(
    ros2 launch
    "${GRAPH_PACKAGE}"
    "${GRAPH_LAUNCH_FILE}"
    "model_file_path:=${MODEL_PATH}"
    "image_topic:=${IMAGE_TOPIC}"
    input_image_width:=1280
    input_image_height:=720
    # Match the NVIDIA Config C reference capture. With a 1280x720 source this
    # makes orig_target_sizes [1280, 1280], as in the upstream RT-DETR node.
    use_max_dim_for_orig_size:=true
    confidence_threshold:=0.6
  )
fi
if [[ ${TRANSPORT} == std ]]; then
  LAUNCH_COMMAND+=("execution_provider:=${EXECUTION_PROVIDER}")
fi
if [[ -n ${ORT_PROFILE_PREFIX} ]]; then
  LAUNCH_COMMAND+=("ort_profile_prefix:=${ORT_PROFILE_PREFIX}")
fi
if [[ -n ${BINDING_REPORT_PATH} ]]; then
  LAUNCH_COMMAND+=("binding_report_path:=${BINDING_REPORT_PATH}")
fi

{
  echo "pipeline=${PIPELINE}"
  echo "execution_provider=${EXECUTION_PROVIDER}"
  echo "transport=${TRANSPORT}"
  echo "input_bag=${INPUT_BAG}"
  echo "dataset_tree_sha256=$(hash_path "${INPUT_BAG}")"
  echo "model_path=${MODEL_PATH}"
  echo "model_sha256=$(hash_path "${MODEL_PATH}")"
  echo "ort_root=${ONNXRUNTIME_ROOT:-}"
  echo "ort_library_sha256=$(hash_path "${ORT_LIBRARY_PATH}")"
  echo "application_revision=$(repo_revision "${WORKSPACE_ROOT}")"
  echo "application_worktree_diff_sha256=$(repo_diff_hash "${WORKSPACE_ROOT}")"
  echo "gpu_ros_managed_revision=$(repo_revision "${GPU_MANAGED_ROOT}")"
  echo "gpu_ros_managed_worktree_diff_sha256=$(repo_diff_hash "${GPU_MANAGED_ROOT}")"
  echo "graph_package=${GRAPH_PACKAGE}"
  echo "graph_launch_file=${GRAPH_LAUNCH_FILE}"
  echo "graph_namespace=${GRAPH_NAMESPACE}"
  echo "output_path=${OUTPUT_PATH}"
  echo "record_output=${RECORD_OUTPUT}"
  echo "ort_profile_prefix=${ORT_PROFILE_PREFIX}"
  echo "binding_report_path=${BINDING_REPORT_PATH}"
  echo "trace_attach_ready_file=${TRACE_ATTACH_READY_FILE}"
  print_command "${LAUNCH_COMMAND[@]}"
} >"${COMMAND_LOG}"

echo "Starting AMD Phase 2A (${EXECUTION_PROVIDER}) graph..."
print_command "${LAUNCH_COMMAND[@]}"
setsid bash -c \
  'trap - INT TERM; exec "$@"' \
  capture-child \
  "${LAUNCH_COMMAND[@]}" >"${LAUNCH_LOG}" 2>&1 &
LAUNCH_PID=$!

if ! wait_for_topic_count \
  "${IMAGE_TOPIC}" \
  "Subscription" \
  "${LAUNCH_PID}" \
  "graph" \
  "${INPUT_READY_TIMEOUT_SECONDS}"; then
  tail -n 100 "${LAUNCH_LOG}" >&2 || true
  exit 1
fi

WARMUP_COMMAND=(
  ros2 bag play
  "${INPUT_BAG}"
  --loop
  --rate "${PLAYBACK_RATE}"
  --topics "${IMAGE_TOPIC}"
)

echo "Starting looped warm-up input; this covers first-run MIGraphX compilation."
print_command "${WARMUP_COMMAND[@]}"
setsid bash -c \
  'trap - INT TERM; exec "$@"' \
  capture-child \
  "${WARMUP_COMMAND[@]}" >"${WARMUP_LOG}" 2>&1 &
WARMUP_PID=$!

if ! discover_detection_topic; then
  tail -n 100 "${LAUNCH_LOG}" >&2 || true
  exit 1
fi

echo "Waiting for one warm-up detection..."
print_command timeout \
  --signal=INT \
  --kill-after=5s \
  "${FIRST_OUTPUT_TIMEOUT_SECONDS}s" \
  ros2 topic echo \
  --once \
  "${DETECTION_TOPIC}"
if ! timeout \
  --signal=INT \
  --kill-after=5s \
  "${FIRST_OUTPUT_TIMEOUT_SECONDS}s" \
  ros2 topic echo \
  --once \
  "${DETECTION_TOPIC}" >"${WARMUP_OUTPUT}" 2>&1; then
  echo "ERROR: no detection arrived during the warm-up timeout." >&2
  tail -n 100 "${LAUNCH_LOG}" >&2 || true
  exit 1
fi

stop_process "${WARMUP_PID}" "warm-up player"
WARMUP_PID=""
echo "Warm-up complete. First detection: ${WARMUP_OUTPUT}"

if [[ -n ${TRACE_ATTACH_READY_FILE} ]]; then
  COMPONENT_CONTAINER_PID="$(find_component_container_pid || true)"
  if [[ ! ${COMPONENT_CONTAINER_PID} =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: could not locate component_container_mt after warm-up." >&2
    exit 1
  fi
  READY_TMP="${TRACE_ATTACH_READY_FILE}.tmp.$$.$RANDOM"
  {
    echo "launch_pid=${LAUNCH_PID}"
    echo "component_container_pid=${COMPONENT_CONTAINER_PID}"
    echo "fixed_input_playback_pending=true"
  } >"${READY_TMP}"
  mv -- "${READY_TMP}" "${TRACE_ATTACH_READY_FILE}"
  echo "Trace attach ready: component_container_mt PID ${COMPONENT_CONTAINER_PID}"

  ATTACH_WAIT_ATTEMPTS=$((TRACE_ATTACH_TIMEOUT_SECONDS * 2))
  for ((attach_wait_attempt = 1; attach_wait_attempt <= ATTACH_WAIT_ATTEMPTS; attach_wait_attempt++)); do
    if [[ -e ${TRACE_ATTACH_RELEASE_FILE} ]]; then
      break
    fi
    if ! process_alive "${LAUNCH_PID}"; then
      echo "ERROR: graph exited while waiting for the profiler attach." >&2
      exit 1
    fi
    if ((attach_wait_attempt % 20 == 0)); then
      echo "Waiting for profiler attach release file..."
    fi
    sleep 0.5
  done
  if [[ ! -e ${TRACE_ATTACH_RELEASE_FILE} ]]; then
    echo "ERROR: profiler attach release timed out after ${TRACE_ATTACH_TIMEOUT_SECONDS}s." >&2
    exit 1
  fi
  echo "Profiler attach acknowledged; starting fixed-input playback."
fi

if [[ ${WARMUP_ONLY} == 1 ]]; then
  stop_process "${LAUNCH_PID}" "graph"
  LAUNCH_PID=""
  trap - EXIT INT TERM
  echo "PASS: warm-up-only probe completed for ${EXECUTION_PROVIDER}."
  exit 0
fi

PLAYBACK_COMMAND=(
  ros2 bag play
  "${INPUT_BAG}"
  --rate "${PLAYBACK_RATE}"
  --topics "${IMAGE_TOPIC}"
)

if [[ ${RECORD_OUTPUT} == 0 ]]; then
  echo "Profile-only mode: playing the complete fixed input without recording output..."
  print_command "${PLAYBACK_COMMAND[@]}"
  print_command "${PLAYBACK_COMMAND[@]}" >>"${COMMAND_LOG}"
  "${PLAYBACK_COMMAND[@]}" >"${PLAYBACK_LOG}" 2>&1

  echo "Playback completed. Draining the graph for ${DRAIN_SECONDS} seconds..."
  sleep "${DRAIN_SECONDS}"

  if [[ -n ${TRACE_ATTACH_PLAYBACK_DONE_FILE} ]]; then
    touch "${TRACE_ATTACH_PLAYBACK_DONE_FILE}"
    echo "Fixed-input playback complete; waiting for profiler detach..."
    if ! wait_for_profiler_detach \
      "${TRACE_ATTACH_DETACH_COMPLETE_FILE}" \
      "${TRACE_ATTACH_TIMEOUT_SECONDS}"; then
      exit 1
    fi
    echo "Profiler detach complete; stopping graph."
  fi

  stop_process "${LAUNCH_PID}" "graph"
  LAUNCH_PID=""

  if [[ -n ${ORT_PROFILE_PREFIX} ]]; then
    PROFILE_DIR="$(dirname "${ORT_PROFILE_PREFIX}")"
    PROFILE_BASENAME="$(basename "${ORT_PROFILE_PREFIX}")"
    PROFILE_JSON="$(
      find "${PROFILE_DIR}" \
        -maxdepth 1 \
        -type f \
        -name "${PROFILE_BASENAME}*.json" \
        -size +0c \
        -print \
        | head -n 1
    )"
    if [[ -z ${PROFILE_JSON} ]]; then
      echo "ERROR: ORT profile was requested but no non-empty profile was produced." >&2
      exit 1
    fi
    echo "ORT profile: ${PROFILE_JSON}"
  fi

  trap - EXIT INT TERM
  echo "PASS: profile-only playback completed for ${EXECUTION_PROVIDER}."
  exit 0
fi

RECORD_COMMAND=(
  ros2 bag record
  --disable-keyboard-controls
  --output "${OUTPUT_PATH}"
  --topics "${DETECTION_TOPIC}"
)
echo "Recording ${DETECTION_TOPIC} to ${OUTPUT_PATH}..."
print_command "${RECORD_COMMAND[@]}"
setsid bash -c \
  'trap - INT TERM; exec "$@"' \
  capture-child \
  "${RECORD_COMMAND[@]}" >"${RECORD_LOG}" 2>&1 &
RECORD_PID=$!

if ! wait_for_topic_count \
  "${DETECTION_TOPIC}" \
  "Subscription" \
  "${RECORD_PID}" \
  "recorder" \
  60; then
  tail -n 100 "${RECORD_LOG}" >&2 || true
  exit 1
fi

echo "Playing the complete fixed input once at rate ${PLAYBACK_RATE}..."
print_command "${PLAYBACK_COMMAND[@]}"
print_command "${PLAYBACK_COMMAND[@]}" >>"${COMMAND_LOG}"
"${PLAYBACK_COMMAND[@]}" >"${PLAYBACK_LOG}" 2>&1

echo "Playback completed. Draining the graph for ${DRAIN_SECONDS} seconds..."
sleep "${DRAIN_SECONDS}"

if [[ -n ${TRACE_ATTACH_PLAYBACK_DONE_FILE} ]]; then
  touch "${TRACE_ATTACH_PLAYBACK_DONE_FILE}"
  echo "Fixed-input playback complete; waiting for profiler detach..."
  if ! wait_for_profiler_detach \
    "${TRACE_ATTACH_DETACH_COMPLETE_FILE}" \
    "${TRACE_ATTACH_TIMEOUT_SECONDS}"; then
    exit 1
  fi
  echo "Profiler detach complete; stopping graph."
fi

stop_recorder "${RECORD_PID}"
RECORD_PID=""
stop_process "${LAUNCH_PID}" "graph"
LAUNCH_PID=""

BAG_INFO="$(ros2 bag info "${OUTPUT_PATH}")"
echo "${BAG_INFO}"
MESSAGE_COUNT="$(awk '/^Messages:/ {print $2; exit}' <<<"${BAG_INFO}")"

if [[ ! ${MESSAGE_COUNT:-} =~ ^[0-9]+$ ]]; then
  echo "ERROR: could not read the message count from ros2 bag info." >&2
  exit 1
fi

if ((MESSAGE_COUNT < MIN_MESSAGES)); then
  echo "ERROR: captured ${MESSAGE_COUNT} messages; required at least ${MIN_MESSAGES}." >&2
  exit 1
fi

trap - EXIT INT TERM
echo "PASS: captured ${MESSAGE_COUNT} messages in ${OUTPUT_PATH}"
