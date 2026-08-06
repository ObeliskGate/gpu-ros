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
  echo "Usage: $0 <rtdetr-c|rtdetr-managed|yolov8-c|yolov8-managed> <output-name>"
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
APP_ROOT="${AMD_ROS_OBJECT_DETECTION_ROOT:-${WORKSPACE_ROOT}/src/amd_ros_object_detection}"
ASSETS_ROOT="${ROS2_BENCHMARK_OVERRIDE_ASSETS_ROOT:-${WORKSPACE_ROOT}/assets}"
INPUT_BAG="${ASSETS_ROOT}/datasets/r2bdataset2024_v1/r2b_robotarm"
OUTPUT_ROOT="${CAPTURE_OUTPUT_ROOT:-${APP_ROOT}/migrated_packages/benchmark_results/phase2b_bags}"
OUTPUT_PATH="${OUTPUT_ROOT}/${OUTPUT_NAME}"
LOG_ROOT="${OUTPUT_ROOT}/logs"
LAUNCH_LOG="${LOG_ROOT}/${OUTPUT_NAME}.launch.log"
RECORD_LOG="${LOG_ROOT}/${OUTPUT_NAME}.record.log"
PLAYBACK_RATE="${CAPTURE_PLAYBACK_RATE:-0.25}"
DRAIN_SECONDS="${CAPTURE_DRAIN_SECONDS:-10}"
MIN_MESSAGES="${CAPTURE_MIN_MESSAGES:-20}"
ORT_PROFILE_PREFIX="${CAPTURE_ORT_PROFILE_PREFIX:-}"
ORT_PROFILE_FRAMES="${CAPTURE_ORT_PROFILE_FRAMES:-0}"
NSYS_OUTPUT="${CAPTURE_NSYS_OUTPUT:-}"
STOP_GRACE_SECONDS="${CAPTURE_STOP_GRACE_SECONDS:-10}"
STOP_TERM_SECONDS="${CAPTURE_STOP_TERM_SECONDS:-5}"
DETECTION_TOPIC=""
CONFIDENCE_THRESHOLD="0.6"
EXTRA_LAUNCH_ARGS=()

case "${LANE}" in
  rtdetr-c)
    MODEL_PATH="${ASSETS_ROOT}/models/synthetica_detr_v1.0.0_onnx/sdetr_grasp.onnx"
    LAUNCH_PACKAGE="isaac_ros_rtdetr_std"
    LAUNCH_FILE="rtdetr_ort_nitros.launch.py"
    EXTRA_LAUNCH_ARGS+=(execution_provider:=cuda)
    DETECTION_CANDIDATES=(
      /detections_output
      /rtdetr_container/detections_output
    )
    ;;
  rtdetr-managed)
    MODEL_PATH="${ASSETS_ROOT}/models/synthetica_detr_v1.0.0_onnx/sdetr_grasp.onnx"
    LAUNCH_PACKAGE="isaac_ros_onnx_inference"
    LAUNCH_FILE="rtdetr_ort_managed.launch.py"
    DETECTION_CANDIDATES=(
      /detections_output
      /rtdetr_managed_container/detections_output
    )
    ;;
  yolov8-c)
    MODEL_PATH="${ASSETS_ROOT}/models/yolov8/yolov8s.onnx"
    LAUNCH_PACKAGE="isaac_ros_onnx_inference"
    LAUNCH_FILE="yolov8_ort_transport.launch.py"
    CONFIDENCE_THRESHOLD="0.25"
    EXTRA_LAUNCH_ARGS+=(transport:=nitros execution_provider:=cuda)
    DETECTION_CANDIDATES=(
      /detections_output
      /yolov8_container/detections_output
    )
    ;;
  yolov8-managed)
    MODEL_PATH="${ASSETS_ROOT}/models/yolov8/yolov8s.onnx"
    LAUNCH_PACKAGE="isaac_ros_onnx_inference"
    LAUNCH_FILE="yolov8_ort_transport.launch.py"
    CONFIDENCE_THRESHOLD="0.25"
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

if [[ -n ${ORT_PROFILE_PREFIX} && ${LANE} != yolov8-c && ${LANE} != yolov8-managed ]]; then
  echo "ERROR: bounded ORT profiling is currently supported only for YOLOv8 lanes." >&2
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

for target in "${OUTPUT_PATH}" "${LAUNCH_LOG}" "${RECORD_LOG}"; do
  if [[ -e ${target} ]]; then
    echo "ERROR: refusing to overwrite existing path: ${target}" >&2
    exit 1
  fi
done

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

for command_name in ros2 awk ps setsid sleep tail; do
  if ! command -v "${command_name}" >/dev/null; then
    echo "ERROR: required command is unavailable: ${command_name}" >&2
    exit 1
  fi
done

LAUNCH_PID=""
RECORD_PID=""

stop_process() {
  local pid="$1"
  local label="$2"
  local attempt
  local state

  if [[ -z ${pid} ]]; then
    return
  fi

  if kill -0 "${pid}" 2>/dev/null; then
    echo "Stopping ${label} (PID ${pid})..."
    kill -INT -- "-${pid}" 2>/dev/null || kill -INT "${pid}" 2>/dev/null || true
  fi

  for ((attempt = 1; attempt <= STOP_GRACE_SECONDS * 4; attempt++)); do
    state="$(ps -o stat= -p "${pid}" 2>/dev/null || true)"
    state="${state//[[:space:]]/}"
    if [[ -z ${state} || ${state:0:1} == "Z" ]]; then
      break
    fi
    sleep 0.25
  done

  if [[ -n ${state} && ${state:0:1} != "Z" ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "${label} did not stop after SIGINT; sending SIGTERM..."
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  fi

  for ((attempt = 1; attempt <= STOP_TERM_SECONDS * 4; attempt++)); do
    state="$(ps -o stat= -p "${pid}" 2>/dev/null || true)"
    state="${state//[[:space:]]/}"
    if [[ -z ${state} || ${state:0:1} == "Z" ]]; then
      break
    fi
    sleep 0.25
  done

  if [[ -n ${state} && ${state:0:1} != "Z" ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "WARNING: ${label} ignored SIGTERM; sending SIGKILL." >&2
    kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
}

cleanup() {
  local status=$?
  trap - EXIT
  set +e
  stop_process "${RECORD_PID}" "recorder"
  stop_process "${LAUNCH_PID}" "graph"
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
  input_image_width:=1280
  input_image_height:=720
  "confidence_threshold:=${CONFIDENCE_THRESHOLD}"
  "${EXTRA_LAUNCH_ARGS[@]}"
)

if [[ -n ${NSYS_OUTPUT} ]]; then
  LAUNCH_COMMAND=(
    nsys profile
    --trace=cuda,nvtx
    --sample=none
    "--output=${NSYS_OUTPUT}"
    "${LAUNCH_COMMAND[@]}"
  )
fi

echo "Starting ${LANE} graph..."
setsid bash -c \
  'trap - INT TERM; exec "$@"' \
  capture-child \
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

if ! wait_for_topic_count \
  "${CAMERA_INFO_TOPIC}" "Subscription" "${LAUNCH_PID}" "graph"; then
  tail -n 100 "${LAUNCH_LOG}" >&2 || true
  exit 1
fi

echo "Recording ${DETECTION_TOPIC} to ${OUTPUT_PATH}..."
setsid bash -c \
  'trap - INT TERM; exec "$@"' \
  capture-child \
  ros2 bag record \
  --output "${OUTPUT_PATH}" \
  "${DETECTION_TOPIC}" >"${RECORD_LOG}" 2>&1 &
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
