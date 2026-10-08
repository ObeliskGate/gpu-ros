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
  echo "Usage: $0 <rtdetr-rosidl-buffer|rtdetr-d|rtdetr-managed|yolov8-rosidl-buffer|yolov8-d|yolov8-managed> <output-name>"
  echo
  echo "Record one fixed-input NVIDIA detection bag in a single terminal."
  echo "The output name must not already exist."
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 2 ]]; then
  usage >&2
  exit 2
fi

LANE="$1"
OUTPUT_NAME="$2"

if [[ ! ${OUTPUT_NAME} =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "ERROR: output-name may contain only letters, numbers, '.', '_' and '-'." >&2
  exit 2
fi

WORKSPACE_ROOT="${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}"
ASSETS_ROOT="${ROS2_BENCHMARK_OVERRIDE_ASSETS_ROOT:-${WORKSPACE_ROOT}/assets}"
INPUT_BAG="${ASSETS_ROOT}/datasets/r2bdataset2024_v1/r2b_robotarm"
RESULTS_ROOT="${OVG_RESULTS_ROOT:-/workspaces/ovg-results}"
OUTPUT_ROOT="${CAPTURE_OUTPUT_ROOT:-${RESULTS_ROOT}/phase2b_bags}"
OUTPUT_PATH="${OUTPUT_ROOT}/${OUTPUT_NAME}"
LOG_ROOT="${OUTPUT_ROOT}/logs"
LAUNCH_LOG="${LOG_ROOT}/${OUTPUT_NAME}.launch.log"
GRAPH_SIGNAL_PID_FILE="${LOG_ROOT}/${OUTPUT_NAME}.launch.pid"
RECORD_LOG="${LOG_ROOT}/${OUTPUT_NAME}.record.log"
PLAYBACK_RATE="${CAPTURE_PLAYBACK_RATE:-0.25}"
DRAIN_SECONDS="${CAPTURE_DRAIN_SECONDS:-10}"
MIN_MESSAGES="${CAPTURE_MIN_MESSAGES:-20}"
ORT_PROFILE_PREFIX="${CAPTURE_ORT_PROFILE_PREFIX:-}"
ORT_PROFILE_FRAMES="${CAPTURE_ORT_PROFILE_FRAMES:-0}"
BINDING_REPORT_PATH="${CAPTURE_BINDING_REPORT_PATH:-}"
NSYS_OUTPUT="${CAPTURE_NSYS_OUTPUT:-}"
STOP_GRACE_SECONDS="${CAPTURE_STOP_GRACE_SECONDS:-10}"
STOP_TERM_SECONDS="${CAPTURE_STOP_TERM_SECONDS:-5}"
DETECTION_TOPIC=""
CONFIDENCE_THRESHOLD="0.6"
EXTRA_LAUNCH_ARGS=()
LAUNCH_INPUT_ARGS=()
REQUIRES_CAMERA_INFO=1

case "${LANE}" in
  rtdetr-rosidl-buffer)
    MODEL_PATH="${ASSETS_ROOT}/models/synthetica_detr_v1.0.0_onnx/sdetr_grasp.onnx"
    LAUNCH_PACKAGE="gpu_ros_nvidia_reference"
    LAUNCH_FILE="rtdetr_ort_tensor_list.launch.py"
    EXTRA_LAUNCH_ARGS+=(execution_provider:=cuda)
    LAUNCH_INPUT_ARGS+=(input_image_width:=1280 input_image_height:=720)
    DETECTION_CANDIDATES=(
      /detections_output
      /rtdetr_container/detections_output
    )
    ;;
  rtdetr-d)
    MODEL_PATH="${ASSETS_ROOT}/models/synthetica_detr_v1.0.0_onnx/sdetr_grasp.onnx"
    LAUNCH_PACKAGE="gpu_ros_nvidia_reference"
    LAUNCH_FILE="rtdetr_ort_std.launch.py"
    EXTRA_LAUNCH_ARGS+=(execution_provider:=cuda)
    LAUNCH_INPUT_ARGS+=(input_image_width:=1280 input_image_height:=720)
    DETECTION_CANDIDATES=(
      /detections_output
      /rtdetr_container/detections_output
    )
    ;;
  rtdetr-managed)
    MODEL_PATH="${ASSETS_ROOT}/models/synthetica_detr_v1.0.0_onnx/sdetr_grasp.onnx"
    LAUNCH_PACKAGE="gpu_ros_nvidia_reference"
    LAUNCH_FILE="rtdetr_ort_managed.launch.py"
    LAUNCH_INPUT_ARGS+=(input_image_width:=1280 input_image_height:=720)
    DETECTION_CANDIDATES=(
      /detections_output
      /rtdetr_managed_container/detections_output
    )
    ;;
  yolov8-rosidl-buffer)
    MODEL_PATH="${ASSETS_ROOT}/models/yolov8/yolov8s.onnx"
    LAUNCH_PACKAGE="gpu_ros_nvidia_reference"
    LAUNCH_FILE="yolov8_ort_transport.launch.py"
    CONFIDENCE_THRESHOLD="0.25"
    EXTRA_LAUNCH_ARGS+=(transport:=rosidl_buffer execution_provider:=cuda)
    LAUNCH_INPUT_ARGS+=(input_image_width:=1280 input_image_height:=720)
    DETECTION_CANDIDATES=(
      /detections_output
      /yolov8_container/detections_output
    )
    ;;
  yolov8-d)
    MODEL_PATH="${ASSETS_ROOT}/models/yolov8/yolov8s.onnx"
    LAUNCH_PACKAGE="gpu_ros_yolov8"
    LAUNCH_FILE="yolov8_ort_std_image.launch.py"
    CONFIDENCE_THRESHOLD="0.25"
    REQUIRES_CAMERA_INFO=0
    EXTRA_LAUNCH_ARGS+=(
      execution_provider:=cuda
      image_topic:=image_rect
      namespace:=yolov8_d
    )
    DETECTION_CANDIDATES=(
      /detections_output
      /yolov8_d/detections_output
      /yolov8_phase2a_container/detections_output
    )
    ;;
  yolov8-managed)
    MODEL_PATH="${ASSETS_ROOT}/models/yolov8/yolov8s.onnx"
    LAUNCH_PACKAGE="gpu_ros_nvidia_reference"
    LAUNCH_FILE="yolov8_ort_transport.launch.py"
    CONFIDENCE_THRESHOLD="0.25"
    LAUNCH_INPUT_ARGS+=(input_image_width:=1280 input_image_height:=720)
    EXTRA_LAUNCH_ARGS+=(transport:=managed execution_provider:=cuda)
    DETECTION_CANDIDATES=(
      /detections_output
      /yolov8_container/detections_output
    )
    ;;
  *)
    echo "ERROR: unsupported lane '${LANE}'." >&2
    usage >&2
    exit 2
    ;;
esac

if [[ ! ${PLAYBACK_RATE} =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "ERROR: CAPTURE_PLAYBACK_RATE must be a positive number." >&2
  exit 2
fi

if [[ ! ${DRAIN_SECONDS} =~ ^[0-9]+$ || ! ${MIN_MESSAGES} =~ ^[0-9]+$ ]]; then
  echo "ERROR: CAPTURE_DRAIN_SECONDS and CAPTURE_MIN_MESSAGES must be integers." >&2
  exit 2
fi

if [[ ! ${ORT_PROFILE_FRAMES} =~ ^[0-9]+$ ]]; then
  echo "ERROR: CAPTURE_ORT_PROFILE_FRAMES must be a non-negative integer." >&2
  exit 2
fi

if [[ ! ${STOP_GRACE_SECONDS} =~ ^[0-9]+$ || ! ${STOP_TERM_SECONDS} =~ ^[0-9]+$ ]]; then
  echo "ERROR: CAPTURE_STOP_GRACE_SECONDS and CAPTURE_STOP_TERM_SECONDS must be integers." >&2
  exit 2
fi

if ((ORT_PROFILE_FRAMES > 0)) && [[ -z ${ORT_PROFILE_PREFIX} ]]; then
  echo "ERROR: CAPTURE_ORT_PROFILE_FRAMES requires CAPTURE_ORT_PROFILE_PREFIX." >&2
  exit 2
fi

if [[ -n ${NSYS_OUTPUT} && ${NSYS_OUTPUT} == *.nsys-rep ]]; then
  echo "ERROR: CAPTURE_NSYS_OUTPUT must be a prefix without the .nsys-rep suffix." >&2
  exit 2
fi

if [[ -n ${ORT_PROFILE_PREFIX} ]]; then
  EXTRA_LAUNCH_ARGS+=(
    "ort_profile_prefix:=${ORT_PROFILE_PREFIX}"
    "ort_profile_frames:=${ORT_PROFILE_FRAMES}"
  )
fi

if [[ -n ${BINDING_REPORT_PATH} ]]; then
  EXTRA_LAUNCH_ARGS+=("binding_report_path:=${BINDING_REPORT_PATH}")
fi

if [[ ! -s ${MODEL_PATH} ]]; then
  echo "ERROR: model is missing or empty: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f ${INPUT_BAG}/metadata.yaml ]]; then
  echo "ERROR: input bag is missing metadata.yaml: ${INPUT_BAG}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

if [[ -n ${ORT_PROFILE_PREFIX} ]]; then
  mkdir -p "$(dirname -- "${ORT_PROFILE_PREFIX}")"
fi

if [[ -n ${NSYS_OUTPUT} ]]; then
  if ! command -v nsys >/dev/null; then
    echo "ERROR: CAPTURE_NSYS_OUTPUT requires NVIDIA Nsight Systems (nsys)." >&2
    exit 1
  fi
  mkdir -p "$(dirname -- "${NSYS_OUTPUT}")"
  for target in "${NSYS_OUTPUT}.nsys-rep" "${NSYS_OUTPUT}.qdstrm"; do
    if [[ -e ${target} ]]; then
      echo "ERROR: refusing to overwrite existing Nsight output: ${target}" >&2
      exit 1
    fi
  done
fi

for target in "${OUTPUT_PATH}" "${LAUNCH_LOG}" "${RECORD_LOG}" "${GRAPH_SIGNAL_PID_FILE}"; do
  if [[ -e ${target} ]]; then
    echo "ERROR: refusing to overwrite existing path: ${target}" >&2
    exit 1
  fi
done
if [[ -n ${BINDING_REPORT_PATH} && -e ${BINDING_REPORT_PATH} ]]; then
  echo "ERROR: refusing to overwrite binding report: ${BINDING_REPORT_PATH}" >&2
  exit 1
fi
if [[ -n ${ORT_PROFILE_PREFIX} ]]; then
  existing_profiles=()
  mapfile -t existing_profiles < <(
    find "$(dirname -- "${ORT_PROFILE_PREFIX}")" -maxdepth 1 -type f \
      -name "$(basename -- "${ORT_PROFILE_PREFIX}")*.json" -print
  )
  if [[ ${#existing_profiles[@]} -ne 0 ]]; then
    echo "ERROR: refusing to overwrite existing ORT profile output for prefix: ${ORT_PROFILE_PREFIX}" >&2
    exit 1
  fi
fi

ROS_SETUP="/opt/ros/${ROS_DISTRO:-lyrical}/setup.bash"
if [[ -f ${ROS_SETUP} ]]; then
  set +u
  # The selected ROS distribution supplies this setup file outside the repository.
  # shellcheck disable=SC1090
  source "${ROS_SETUP}"
  set -u
fi

if [[ -f ${WORKSPACE_ROOT}/install/setup.bash ]]; then
  set +u
  # Colcon generates this setup file in the external runtime workspace.
  # shellcheck disable=SC1090
  source "${WORKSPACE_ROOT}/install/setup.bash"
  set -u
fi

for command_name in ros2 awk ps setsid env sleep tail; do
  if ! command -v "${command_name}" >/dev/null; then
    echo "ERROR: required command is unavailable: ${command_name}" >&2
    exit 1
  fi
done

LAUNCH_PID=""
RECORD_PID=""
GRAPH_WAIT_RC=""
GRAPH_SIGINT_SENT=0
GRAPH_TERM_SENT=0
GRAPH_KILL_SENT=0
RECORDER_ESCALATED=0

stop_process() {
  local pid="$1"
  local label="$2"
  local attempt
  local state=""
  local sigint_sent=0
  local term_sent=0
  local kill_sent=0
  local wait_status
  local signal_pid="${pid}"
  local signal_ancestor=""

  if [[ -z ${pid} ]]; then
    return
  fi

  if kill -0 "${pid}" 2>/dev/null; then
    echo "Stopping ${label} (PID ${pid})..."
    if [[ ${label} == graph ]]; then
      # Launch forwards SIGINT to its components. Signalling the whole group
      # also delivers a second SIGINT during component/static teardown.
      signal_pid=""
      if [[ -r ${GRAPH_SIGNAL_PID_FILE} ]]; then
        read -r signal_pid <"${GRAPH_SIGNAL_PID_FILE}" || true
      fi
      if [[ ${signal_pid} =~ ^[1-9][0-9]*$ ]]; then
        # Nsight starts its target in a separate session. Verify ancestry,
        # rather than process-group identity, before trusting the PID file.
        signal_ancestor="${signal_pid}"
        while [[ ${signal_ancestor} =~ ^[1-9][0-9]*$ &&
          ${signal_ancestor} != "${pid}" && ${signal_ancestor} != 1 ]]; do
          signal_ancestor="$(ps -o ppid= -p "${signal_ancestor}" 2>/dev/null || true)"
          signal_ancestor="${signal_ancestor//[[:space:]]/}"
        done
        [[ ${signal_ancestor} == "${pid}" ]] || signal_pid=""
      else
        signal_pid=""
      fi
    fi
    if [[ -n ${signal_pid} ]] && kill -INT "${signal_pid}" 2>/dev/null; then
      sigint_sent=1
    fi
  fi

  for ((attempt = 0; attempt <= STOP_GRACE_SECONDS * 4; attempt++)); do
    state="$(ps -o stat= -p "${pid}" 2>/dev/null || true)"
    state="${state//[[:space:]]/}"
    if [[ -z ${state} || ${state:0:1} == "Z" ]]; then
      break
    fi
    if ((attempt == STOP_GRACE_SECONDS * 4)); then
      break
    fi
    sleep 0.25
  done

  if [[ -n ${state} && ${state:0:1} != "Z" ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "${label} did not stop after SIGINT; sending SIGTERM..."
    if kill -TERM -- "-${pid}" 2>/dev/null; then
      term_sent=1
    elif kill -TERM "${pid}" 2>/dev/null; then
      term_sent=1
    fi
  fi

  for ((attempt = 0; attempt <= STOP_TERM_SECONDS * 4; attempt++)); do
    state="$(ps -o stat= -p "${pid}" 2>/dev/null || true)"
    state="${state//[[:space:]]/}"
    if [[ -z ${state} || ${state:0:1} == "Z" ]]; then
      break
    fi
    if ((attempt == STOP_TERM_SECONDS * 4)); then
      break
    fi
    sleep 0.25
  done

  if [[ -n ${state} && ${state:0:1} != "Z" ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "WARNING: ${label} ignored SIGTERM; sending SIGKILL." >&2
    if kill -KILL -- "-${pid}" 2>/dev/null; then
      kill_sent=1
    elif kill -KILL "${pid}" 2>/dev/null; then
      kill_sent=1
    fi
  fi

  if wait "${pid}" 2>/dev/null; then
    wait_status=0
  else
    wait_status=$?
  fi
  if [[ ${label} == "graph" ]]; then
    GRAPH_WAIT_RC=${wait_status}
    GRAPH_SIGINT_SENT=${sigint_sent}
    GRAPH_TERM_SENT=${term_sent}
    GRAPH_KILL_SENT=${kill_sent}
  elif [[ ${term_sent} -eq 1 || ${kill_sent} -eq 1 ]]; then
    RECORDER_ESCALATED=1
  fi
}

check_component_exit() {
  local result

  if [[ ! -r ${LAUNCH_LOG} ]]; then
    echo "ERROR: component exit is unconfirmed: launch log is missing or unreadable (${LAUNCH_LOG})." >&2
    return 1
  fi

  # ROS 2 launch's ExecuteLocal emits these process lifecycle records.
  # ComposableNodeContainer uses component_container_mt as its executable;
  # launch_ros labels this process with the executable basename by default.
  if result="$(
    awk '
      BEGIN {
        quote = sprintf("%c", 39)
        death_pattern = "^\\[ERROR\\] \\[component_container_mt-[0-9]+\\]: process has died \\[pid [0-9]+, exit code -?[0-9]+, cmd " quote ".*" quote "\\]\\.$"
      }
      function logger_name(line) {
        sub(/^\[[A-Z]+\] \[/, "", line)
        sub(/\]:.*/, "", line)
        return line
      }
      function process_pid(line) {
        sub(/^.*pid \[?/, "", line)
        sub(/[].,].*$/, "", line)
        return line
      }
      {
        if ($0 ~ /^\[INFO\] \[component_container_mt-[0-9]+\]: process started with pid \[[0-9]+\]$/) {
          start_count++
          if (start_count == 1) {
            start_logger = logger_name($0)
            start_pid = process_pid($0)
          }
          next
        }
        if ($0 ~ /^\[INFO\] \[component_container_mt-[0-9]+\]: process has finished cleanly \[pid [0-9]+\]$/) {
          end_count++
          if (end_count == 1) {
            end_kind = "clean"
            end_logger = logger_name($0)
            end_pid = process_pid($0)
          }
          next
        }
        if ($0 ~ death_pattern) {
          end_count++
          if (end_count == 1) {
            end_kind = "death"
            end_logger = logger_name($0)
            end_pid = process_pid($0)
            end_exit_code = $0
            sub(/^.*exit code /, "", end_exit_code)
            sub(/, cmd .*$/, "", end_exit_code)
          }
          next
        }
        if ($0 ~ /^\[[A-Z]+\] \[component_container_mt-[0-9]+\]: process (started with pid|has finished cleanly|has died)/) {
          malformed = 1
        }
      }
      END {
        if (malformed) {
          print "component exit is unconfirmed: a component_container_mt lifecycle record does not match the expected ROS 2 launch format."
          exit 1
        }
        if (start_count == 0) {
          print "component exit is unconfirmed: no component_container_mt start record was found."
          exit 1
        }
        if (start_count != 1) {
          print "component exit is contradictory: found " start_count " component_container_mt start records; expected exactly one."
          exit 1
        }
        if (end_count == 0) {
          print "component exit is unconfirmed: no component_container_mt clean/death record was found."
          exit 1
        }
        if (end_count != 1) {
          print "component exit is contradictory: found " end_count " component_container_mt termination records; expected exactly one."
          exit 1
        }
        if (end_logger != start_logger || end_pid != start_pid) {
          print "component exit is contradictory: termination logger/PID " end_logger "/" end_pid \
            " does not match start logger/PID " start_logger "/" start_pid "."
          exit 1
        }
        if (end_kind == "death") {
          if (end_exit_code == 0) {
            print "component container " start_logger " (PID " start_pid \
              ") has a ROS 2 launch death record with exit code 0."
            exit 1
          }
          print "component container " start_logger " (PID " start_pid \
            ") exited nonzero (exit code " end_exit_code ")."
          exit 1
        }
        print "component container " start_logger " (PID " start_pid ") exited cleanly."
      }
    ' "${LAUNCH_LOG}"
  )"; then
    echo "INFO: ${result}"
    return 0
  fi

  echo "ERROR: ${result}" >&2
  return 1
}

check_graph_lifecycle() {
  local failed=0
  local component_clean=0

  if check_component_exit; then
    component_clean=1
  else
    failed=1
  fi
  if [[ ${RECORDER_ESCALATED} -eq 1 ]]; then
    echo "ERROR: capture lifecycle gate failed: recorder required SIGTERM/SIGKILL escalation." >&2
    failed=1
  fi

  if [[ ${GRAPH_TERM_SENT} -eq 1 ]]; then
    echo "ERROR: capture lifecycle gate failed: graph required SIGTERM escalation." >&2
    failed=1
  fi
  if [[ ${GRAPH_KILL_SENT} -eq 1 ]]; then
    echo "ERROR: capture lifecycle gate failed: graph required SIGKILL escalation." >&2
    failed=1
  fi

  if [[ -z ${GRAPH_WAIT_RC} ]]; then
    echo "ERROR: capture lifecycle gate failed: graph wrapper wait status was not recorded." >&2
    failed=1
  elif [[ ${GRAPH_WAIT_RC} -eq 0 ]]; then
    :
  elif [[ ${GRAPH_WAIT_RC} -eq 130 ]]; then
    if [[ ${GRAPH_SIGINT_SENT} -ne 1 ]]; then
      echo "ERROR: capture lifecycle gate failed: graph wrapper exited 130 without an intentional SIGINT." >&2
      failed=1
    fi
    if [[ ${component_clean} -ne 1 ]]; then
      echo "ERROR: capture lifecycle gate failed: graph wrapper exited 130 without confirmed clean component exit." >&2
      failed=1
    fi
  else
    echo "ERROR: capture lifecycle gate failed: graph wrapper exited with status ${GRAPH_WAIT_RC}." >&2
    failed=1
  fi

  if [[ ${failed} -ne 0 ]]; then
    return 1
  fi
  echo "INFO: graph wrapper exited with status ${GRAPH_WAIT_RC}; intentional SIGINT=${GRAPH_SIGINT_SENT}."
}

cleanup() {
  local status=$?
  trap - EXIT
  set +e
  stop_process "${RECORD_PID}" "recorder"
  stop_process "${LAUNCH_PID}" "graph"
  if [[ -n ${LAUNCH_PID} ]]; then
    check_graph_lifecycle || true
  fi
  if [[ ${status} -ne 0 ]]; then
    echo "Capture failed. Logs:"
    echo "  ${LAUNCH_LOG}"
    echo "  ${RECORD_LOG}"
  fi
  exit "${status}"
}

topic_count() {
  local topic="$1"
  local count_kind="$2"
  ros2 topic info "${topic}" 2>/dev/null |
    awk -v kind="${count_kind}" '$1 == kind && $2 == "count:" {print $3; exit}'
}

wait_for_topic_count() {
  local topic="$1"
  local count_kind="$2"
  local process_pid="$3"
  local process_label="$4"
  local attempt
  local count

  for ((attempt = 1; attempt <= 120; attempt++)); do
    count="$(topic_count "${topic}" "${count_kind}" || true)"
    if [[ ${count:-0} =~ ^[0-9]+$ ]] && ((count >= 1)); then
      echo "${topic}: ${count_kind} count=${count}"
      return 0
    fi

    if ! kill -0 "${process_pid}" 2>/dev/null; then
      echo "ERROR: ${process_label} exited before topic discovery completed." >&2
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
  local attempt
  local candidate
  local count

  for ((attempt = 1; attempt <= 120; attempt++)); do
    for candidate in "${DETECTION_CANDIDATES[@]}"; do
      count="$(topic_count "${candidate}" "Publisher" || true)"
      if [[ ${count:-0} =~ ^[0-9]+$ ]] && ((count >= 1)); then
        DETECTION_TOPIC="${candidate}"
        echo "Discovered detection topic: ${DETECTION_TOPIC}"
        return 0
      fi
    done

    if ! kill -0 "${LAUNCH_PID}" 2>/dev/null; then
      echo "ERROR: graph exited before topic discovery completed." >&2
      return 1
    fi

    if ((attempt % 20 == 0)); then
      echo "Waiting for a detection publisher..."
    fi
    sleep 0.5
  done

  echo "ERROR: timed out waiting for a detection publisher." >&2
  return 1
}

for candidate in "${DETECTION_CANDIDATES[@]}"; do
  EXISTING_PUBLISHERS="$(topic_count "${candidate}" "Publisher" || true)"
  if [[ ${EXISTING_PUBLISHERS:-0} =~ ^[0-9]+$ ]] && ((EXISTING_PUBLISHERS >= 1)); then
    echo "ERROR: a detection graph is already publishing on ${candidate}." >&2
    echo "Stop the existing graph before starting a capture." >&2
    exit 1
  fi
done

trap cleanup EXIT
trap 'exit 130' INT TERM

LAUNCH_COMMAND=(
  ros2 launch
  "${LAUNCH_PACKAGE}"
  "${LAUNCH_FILE}"
  "model_file_path:=${MODEL_PATH}"
  "${LAUNCH_INPUT_ARGS[@]}"
  "confidence_threshold:=${CONFIDENCE_THRESHOLD}"
  "${EXTRA_LAUNCH_ARGS[@]}"
)

# Record ros2 launch's PID even when Nsight is the outer process. Keep group
# signalling only for TERM/KILL escalation, which remains a lifecycle failure.
LAUNCH_COMMAND=(
  bash -c 'printf "%s\n" "$$" >"$1"; shift; exec "$@"'
  capture-launch "${GRAPH_SIGNAL_PID_FILE}" "${LAUNCH_COMMAND[@]}"
)

if [[ -n ${NSYS_OUTPUT} ]]; then
  LAUNCH_COMMAND=(
    nsys profile
    "--trace=cuda,nvtx"
    --sample=none
    "--output=${NSYS_OUTPUT}"
    "${LAUNCH_COMMAND[@]}"
  )
fi

echo "Starting ${LANE} graph..."
# Bash cannot reset a signal inherited as ignored by an asynchronous command.
# Restore dispositions before exec so non-interactive recorders receive SIGINT.
setsid env --default-signal=INT,TERM \
  "${LAUNCH_COMMAND[@]}" >"${LAUNCH_LOG}" 2>&1 &
LAUNCH_PID=$!

if ! discover_detection_topic; then
  tail -n 100 "${LAUNCH_LOG}" >&2 || true
  exit 1
fi

TOPIC_PREFIX="${DETECTION_TOPIC%/detections_output}"
IMAGE_TOPIC="${TOPIC_PREFIX}/image_rect"
CAMERA_INFO_TOPIC="${TOPIC_PREFIX}/camera_info_rect"

if ! wait_for_topic_count "${IMAGE_TOPIC}" "Subscription" "${LAUNCH_PID}" "graph"; then
  tail -n 100 "${LAUNCH_LOG}" >&2 || true
  exit 1
fi

if ((REQUIRES_CAMERA_INFO)); then
  if ! wait_for_topic_count \
    "${CAMERA_INFO_TOPIC}" "Subscription" "${LAUNCH_PID}" "graph"; then
    tail -n 100 "${LAUNCH_LOG}" >&2 || true
    exit 1
  fi
fi

echo "Recording ${DETECTION_TOPIC} to ${OUTPUT_PATH}..."
setsid env --default-signal=INT,TERM \
  ros2 bag record \
  --output "${OUTPUT_PATH}" \
  --topics "${DETECTION_TOPIC}" >"${RECORD_LOG}" 2>&1 &
RECORD_PID=$!

if ! wait_for_topic_count \
  "${DETECTION_TOPIC}" "Subscription" "${RECORD_PID}" "recorder"; then
  tail -n 100 "${RECORD_LOG}" >&2 || true
  exit 1
fi

echo "Playing the complete fixed input bag at rate ${PLAYBACK_RATE}..."
ros2 bag play \
  "${INPUT_BAG}" \
  --rate "${PLAYBACK_RATE}" \
  --topics \
  /camera_1/color/image_raw \
  /camera_1/color/camera_info \
  --remap \
  "/camera_1/color/image_raw:=${IMAGE_TOPIC}" \
  "/camera_1/color/camera_info:=${CAMERA_INFO_TOPIC}"

echo "Playback completed. Draining the graph for ${DRAIN_SECONDS} seconds..."
sleep "${DRAIN_SECONDS}"

stop_process "${RECORD_PID}" "recorder"
RECORD_PID=""
stop_process "${LAUNCH_PID}" "graph"
LAUNCH_PID=""
if ! check_graph_lifecycle; then
  exit 1
fi

BAG_INFO="$(ros2 bag info "${OUTPUT_PATH}")"
echo "${BAG_INFO}"

MESSAGE_COUNT="$(
  awk '/^Messages:/ {print $2; exit}' <<<"${BAG_INFO}"
)"

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
