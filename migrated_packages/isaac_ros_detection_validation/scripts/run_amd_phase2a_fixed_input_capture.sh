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
  echo "Usage: $0 <output-name>"
  echo
  echo "Launch, warm up, play the fixed r2b input and record AMD Phase 2A in one terminal."
  echo "The output name and its log files must not already exist."
  echo
  echo "Environment overrides:"
  echo "  CAPTURE_EXECUTION_PROVIDER=migraphx|cpu  (default: migraphx)"
  echo "  CAPTURE_WARMUP_ONLY=0|1                 (default: 0)"
  echo "  CAPTURE_PLAYBACK_RATE=<positive number> (default: 0.25)"
  echo "  CAPTURE_MIN_MESSAGES=<integer>          (default: 20)"
  echo "  CAPTURE_FIRST_OUTPUT_TIMEOUT_SECONDS=<integer> (default: 900)"
  echo "  CAPTURE_GRAPH_READY_TIMEOUT_SECONDS=<integer>  (default: 900)"
  echo "  CAPTURE_OUTPUT_ROOT=<absolute path>"
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

OUTPUT_NAME="$1"
if [[ ! ${OUTPUT_NAME} =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "ERROR: output-name may contain only letters, numbers, '.', '_' and '-'." >&2
  exit 2
fi

WORKSPACE_ROOT="${OVG_WORKSPACE_ROOT:-/workspaces/amd_ros_object_detection}"
ASSETS_ROOT="${OVG_ASSETS_ROOT:-${ROS2_BENCHMARK_OVERRIDE_ASSETS_ROOT:-/workspaces/ovg-assets}}"
RESULTS_ROOT="${OVG_RESULTS_ROOT:-/workspaces/ovg-results}"
INPUT_BAG="${CAPTURE_INPUT_BAG:-${ASSETS_ROOT}/datasets/r2bdataset2024_v1/r2b_robotarm}"
MODEL_PATH="${CAPTURE_MODEL_PATH:-${ASSETS_ROOT}/models/synthetica_detr_v1.0.0_onnx/sdetr_grasp.onnx}"
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
DETECTION_TOPIC=""
DETECTION_CANDIDATES=(
  /detections_output
  /rtdetr/detections_output
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

if [[ ! -s ${MODEL_PATH} ]]; then
  echo "ERROR: model is missing or empty: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f ${INPUT_BAG}/metadata.yaml ]]; then
  echo "ERROR: input bag is missing metadata.yaml: ${INPUT_BAG}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

for target in \
  "${OUTPUT_PATH}" \
  "${LAUNCH_LOG}" \
  "${RECORD_LOG}" \
  "${PLAYBACK_LOG}" \
  "${WARMUP_LOG}" \
  "${WARMUP_OUTPUT}" \
  "${COMMAND_LOG}"; do
  if [[ -e ${target} ]]; then
    echo "ERROR: refusing to overwrite existing path: ${target}" >&2
    exit 1
  fi
done

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

for command_name in awk ps ros2 setsid sleep tail timeout; do
  if ! command -v "${command_name}" >/dev/null; then
    echo "ERROR: required command is unavailable: ${command_name}" >&2
    exit 1
  fi
done

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

cleanup() {
  local status=$?
  trap - EXIT
  set +e
  stop_process "${RECORD_PID}" "recorder"
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
    | awk '/ros2 launch isaac_ros_rtdetr_std|component_container_mt|ros2 bag play/ {print}' \
    >&2 || true
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

LAUNCH_COMMAND=(
  ros2 launch
  isaac_ros_rtdetr_std
  rtdetr_ort_std_image.launch.py
  "model_file_path:=${MODEL_PATH}"
  "image_topic:=${IMAGE_TOPIC}"
  input_image_width:=1280
  input_image_height:=720
  # Match the NVIDIA Config C reference capture. With a 1280x720 source this
  # makes orig_target_sizes [1280, 1280], as in the upstream RT-DETR node.
  use_max_dim_for_orig_size:=true
  "execution_provider:=${EXECUTION_PROVIDER}"
  confidence_threshold:=0.6
)

{
  echo "execution_provider=${EXECUTION_PROVIDER}"
  echo "input_bag=${INPUT_BAG}"
  echo "model_path=${MODEL_PATH}"
  echo "output_path=${OUTPUT_PATH}"
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

if [[ ${WARMUP_ONLY} == 1 ]]; then
  stop_process "${LAUNCH_PID}" "graph"
  LAUNCH_PID=""
  trap - EXIT INT TERM
  echo "PASS: warm-up-only probe completed for ${EXECUTION_PROVIDER}."
  exit 0
fi

RECORD_COMMAND=(
  ros2 bag record
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

PLAYBACK_COMMAND=(
  ros2 bag play
  "${INPUT_BAG}"
  --rate "${PLAYBACK_RATE}"
  --topics "${IMAGE_TOPIC}"
)
echo "Playing the complete fixed input once at rate ${PLAYBACK_RATE}..."
print_command "${PLAYBACK_COMMAND[@]}"
print_command "${PLAYBACK_COMMAND[@]}" >>"${COMMAND_LOG}"
"${PLAYBACK_COMMAND[@]}" >"${PLAYBACK_LOG}" 2>&1

echo "Playback completed. Draining the graph for ${DRAIN_SECONDS} seconds..."
sleep "${DRAIN_SECONDS}"

stop_process "${RECORD_PID}" "recorder"
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
