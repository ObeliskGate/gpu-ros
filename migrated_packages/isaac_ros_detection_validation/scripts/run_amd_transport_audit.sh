#!/usr/bin/env bash
# Copyright 2026 Maintainer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

set -euo pipefail

usage() {
  echo "Usage: $0 <yolov8|rtdetr> <audit-name>"
  echo
  echo "Run std and Managed AMD fixed-input audits with rocprofv3 attach tracing."
  echo "Warm-up is completed before rocprofv3 attaches to component_container_mt."
  echo "The audit output directory must not already exist."
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

WORKSPACE_ROOT="${OVG_WORKSPACE_ROOT:-/workspaces/amd_ros_object_detection}"
ASSETS_ROOT="${OVG_ASSETS_ROOT:-${ROS2_BENCHMARK_OVERRIDE_ASSETS_ROOT:-/workspaces/ovg-assets}}"
AUDIT_PARENT="${CAPTURE_AUDIT_ROOT:-${OVG_RESULTS_ROOT:-/workspaces/ovg-results}/phase2b_amd_audits}"
AUDIT_ROOT="${AUDIT_PARENT}/${AUDIT_NAME}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CAPTURE_RUNNER="${SCRIPT_DIR}/run_amd_phase2a_fixed_input_capture.sh"
ORT_ROOT="${AUDIT_ROOT}/ort"
TRACE_ROOT="${AUDIT_ROOT}/rocprof"
BAG_ROOT="${AUDIT_ROOT}/bags"
BINDING_ROOT="${AUDIT_ROOT}/binding"
REPORT_ROOT="${AUDIT_ROOT}/reports"
LOG_ROOT="${AUDIT_ROOT}/logs"
PLAYBACK_RATE="${AUDIT_PLAYBACK_RATE:-0.25}"
TRACE_ATTACH_TIMEOUT_SECONDS="${AUDIT_TRACE_ATTACH_TIMEOUT_SECONDS:-900}"

if [[ ${MODEL} == yolov8 ]]; then
  PAYLOAD_SIZES=(4915200 2822400)
else
  PAYLOAD_SIZES=(4915200 800 1600 400)
fi

if [[ -e ${AUDIT_ROOT} ]]; then
  echo "ERROR: refusing to overwrite existing audit path: ${AUDIT_ROOT}" >&2
  exit 1
fi
if [[ ! -x ${CAPTURE_RUNNER} ]]; then
  echo "ERROR: capture runner is missing or not executable: ${CAPTURE_RUNNER}" >&2
  exit 1
fi
if [[ ! ${TRACE_ATTACH_TIMEOUT_SECONDS} =~ ^[0-9]+$ ]]; then
  echo "ERROR: AUDIT_TRACE_ATTACH_TIMEOUT_SECONDS must be a non-negative integer." >&2
  exit 2
fi
if [[ ! -s ${ASSETS_ROOT}/models/synthetica_detr_v1.0.0_onnx/sdetr_grasp.onnx &&
  ${MODEL} == rtdetr ]]; then
  echo "ERROR: RT-DETR model is missing under ${ASSETS_ROOT}." >&2
  exit 1
fi
if [[ ${MODEL} == yolov8 && ! -s ${ASSETS_ROOT}/models/yolov8/yolov8s.onnx ]]; then
  echo "YOLOv8 ONNX asset is missing under ${ASSETS_ROOT}." >&2
  echo "Provide the user-supplied model through the documented local import flow." >&2
  exit 1
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

for command_name in awk find grep ros2 rocprofv3 sed sleep sort tee; do
  if ! command -v "${command_name}" >/dev/null; then
    echo "ERROR: required command is unavailable: ${command_name}" >&2
    exit 1
  fi
done

ROCPROF_ATTACH_ARGS=()
ROCPROF_ATTACH_MODE=target-tool-attach
if rocprofv3 --help 2>&1 | grep -q -- '--attach-sync-output'; then
  ROCPROF_ATTACH_ARGS+=(--attach-sync-output)
  ROCPROF_ATTACH_MODE="target-tool-attach+sync-output"
fi
if rocprofv3 --help 2>&1 | grep -q -- '--attach-duration-msec'; then
  ROCPROF_ATTACH_ARGS+=(
    --attach-duration-msec "$((TRACE_ATTACH_TIMEOUT_SECONDS * 1000))")
  ROCPROF_ATTACH_MODE="${ROCPROF_ATTACH_MODE}+duration"
else
  echo "WARNING: installed rocprofv3 has no non-interactive attach duration; " \
    "SIGINT detach will be used." >&2
fi

mkdir -p "${ORT_ROOT}" "${TRACE_ROOT}" "${BAG_ROOT}" "${BINDING_ROOT}" \
  "${REPORT_ROOT}" "${LOG_ROOT}"

CAPTURE_PID=""
PROFILER_PID=""
ACTIVE_RELEASE_FILE=""
cleanup() {
  local status=$?
  trap - EXIT
  set +e
  if [[ -n ${CAPTURE_PID} ]]; then
    if [[ -n ${ACTIVE_RELEASE_FILE:-} ]]; then
      touch "${ACTIVE_RELEASE_FILE}" 2>/dev/null || true
    fi
    kill -INT "${CAPTURE_PID}" 2>/dev/null || true
    wait "${CAPTURE_PID}" 2>/dev/null || true
  fi
  if [[ -n ${PROFILER_PID} ]]; then
    kill -TERM "${PROFILER_PID}" 2>/dev/null || true
    wait "${PROFILER_PID}" 2>/dev/null || true
  fi
  if ((status != 0)); then
    echo "AMD transport audit failed; inspect ${LOG_ROOT}." >&2
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

wait_for_ready_file() {
  local ready_file="$1"
  local capture_pid="$2"
  local attempt
  local attempts=$((TRACE_ATTACH_TIMEOUT_SECONDS * 2))
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if [[ -s ${ready_file} ]]; then
      return 0
    fi
    if ! kill -0 "${capture_pid}" 2>/dev/null; then
      echo "ERROR: capture exited before trace attach became ready: ${ready_file}" >&2
      return 1
    fi
    sleep 0.5
  done
  echo "ERROR: timed out waiting for trace attach readiness: ${ready_file}" >&2
  return 1
}

wait_for_playback_done_file() {
  local done_file="$1"
  local capture_pid="$2"
  local profiler_pid="$3"
  local attempt
  local attempts=$((TRACE_ATTACH_TIMEOUT_SECONDS * 2))

  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if [[ -e ${done_file} ]]; then
      return 0
    fi
    if ! kill -0 "${capture_pid}" 2>/dev/null; then
      echo "ERROR: capture exited before fixed-input playback completed: ${done_file}" >&2
      return 1
    fi
    if ! kill -0 "${profiler_pid}" 2>/dev/null; then
      echo "ERROR: rocprofv3 exited before fixed-input playback completed." >&2
      return 1
    fi
    sleep 0.5
  done

  echo "ERROR: timed out waiting for fixed-input playback completion: ${done_file}" >&2
  return 1
}

read_ready_value() {
  local key="$1"
  local path="$2"
  awk -F= -v wanted="${key}" '$1 == wanted {print $2; exit}' "${path}"
}

wait_for_trace_outputs() {
  local trace_dir="$1"
  local attempts=$((TRACE_ATTACH_TIMEOUT_SECONDS * 2))
  local attempt
  local previous=""
  local current=""
  local stable_count=0

  # Older ROCprofiler SDKs do not provide --attach-sync-output. Wait until
  # the JSON output exists and its file set/sizes are stable before parsing.
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    current="$(find "${trace_dir}" -type f -name '*.json' -size +0c \
      -printf '%p %s\n' | sort)"
    if [[ -n ${current} && ${current} == "${previous}" ]]; then
      ((stable_count += 1))
      if ((stable_count >= 2)); then
        return 0
      fi
    else
      stable_count=0
      previous="${current}"
    fi
    sleep 0.5
  done

  echo "ERROR: rocprofv3 produced no stable non-empty JSON output in ${trace_dir}." >&2
  return 1
}

write_capture_manifest() {
  local path="$1"
  local lane="$2"
  if [[ -e ${path} ]]; then
    echo "ERROR: refusing to overwrite capture manifest: ${path}" >&2
    return 1
  fi
  {
    echo '{'
    echo '  "schema_version": 1,'
    echo "  \"lane\": \"${lane}\","
    echo '  "tracing_domains": ["memory_copy_trace", "kernel_trace"],'
    echo '  "output_formats": ["json"],'
    echo '  "attach_mode": "target-tool-attach",'
    echo '  "warmup_excluded": true'
    echo '}'
  } >"${path}"
}

start_profiler_attach() {
  local container_pid="$1"
  local trace_dir="$2"
  local profiler_log="$3"
  mkdir -p "${trace_dir}"
  echo "Attaching rocprofv3 to component_container_mt PID ${container_pid}..."
  # Keep this attach form explicit.  The warm-up player has already stopped,
  # so no warm-up GPU activity is included in the copy statistics.
  (
    cd "${trace_dir}"
    # ROCprofiler-SDK 1.0 uses these values when it injects the profiler
    # configuration into the attached target. Keep them scoped to this
    # profiler process; do not modify the container or shell environment.
    ROCPROF_KERNEL_TRACE=1 \
    ROCPROF_MEMORY_COPY_TRACE=1 \
    ROCPROF_OUTPUT_PATH="${trace_dir}" \
    ROCPROF_OUTPUT_FILE_NAME="${MODEL}_managed_attach" \
    ROCPROF_OUTPUT_FORMAT=json \
    rocprofv3 --attach "${container_pid}" \
      --memory-copy-trace \
      --kernel-trace \
      --output-format json \
      "${ROCPROF_ATTACH_ARGS[@]}" \
      --output-directory "${trace_dir}" \
      --output-file "${MODEL}_managed_attach"
  ) >"${profiler_log}" 2>&1 &
  PROFILER_PID=$!
  sleep 1
  if ! kill -0 "${PROFILER_PID}" 2>/dev/null; then
    wait "${PROFILER_PID}" 2>/dev/null || true
    echo "ERROR: rocprofv3 attach exited before fixed-input playback." >&2
    echo "rocprofv3 log: ${profiler_log}" >&2
    sed -n '1,160p' "${profiler_log}" >&2 || true
    return 1
  fi
}

run_lane() {
  local transport="$1"
  local lane_name="$2"
  local output_name="${transport}"
  local ready_file="${LOG_ROOT}/${lane_name}.attach-ready"
  local release_file="${LOG_ROOT}/${lane_name}.attach-release"
  local trace_dir="${TRACE_ROOT}/${lane_name}"
  local profiler_log="${LOG_ROOT}/${lane_name}.rocprof.log"
  local playback_done_file="${LOG_ROOT}/${lane_name}.playback-done"
  local detach_complete_file="${LOG_ROOT}/${lane_name}.detach-complete"
  local binding_path="${BINDING_ROOT}/${lane_name}.json"
  local profile_prefix="${ORT_ROOT}/${lane_name}_"
  local command_log="${LOG_ROOT}/${lane_name}.capture.log"
  local container_pid
  local capture_status
  local profiler_status
  local wait_for_playback_status
  local trace_files
  local capture_args=("${output_name}")
  local manifest_path="${TRACE_ROOT}/${lane_name}.capture-manifest.json"

  # The RT-DETR capture runner uses its one-argument form, while the
  # YOLOv8 runner is selected with an explicit leading "yolov8" argument.
  if [[ ${MODEL} == yolov8 ]]; then
    capture_args=(yolov8 "${output_name}")
  fi

  echo "Starting AMD ${MODEL} ${transport} lane..."
  # ROCprofiler-SDK process attachment requires the target process to opt in
  # before it is attached.  Keep this compatibility setting scoped to the
  # capture lane; do not export it from the container or modify the user's
  # shell environment.
  ROCP_TOOL_ATTACH=1 \
  CAPTURE_TRANSPORT="${transport}" \
  CAPTURE_EXECUTION_PROVIDER=migraphx \
  CAPTURE_OUTPUT_ROOT="${BAG_ROOT}" \
  CAPTURE_PLAYBACK_RATE="${PLAYBACK_RATE}" \
  CAPTURE_ORT_PROFILE_PREFIX="${profile_prefix}" \
  CAPTURE_BINDING_REPORT_PATH="${binding_path}" \
  CAPTURE_TRACE_ATTACH_READY_FILE="${ready_file}" \
  CAPTURE_TRACE_ATTACH_RELEASE_FILE="${release_file}" \
  CAPTURE_TRACE_ATTACH_PLAYBACK_DONE_FILE="${playback_done_file}" \
  CAPTURE_TRACE_ATTACH_DETACH_COMPLETE_FILE="${detach_complete_file}" \
  CAPTURE_TRACE_ATTACH_TIMEOUT_SECONDS="${TRACE_ATTACH_TIMEOUT_SECONDS}" \
  CAPTURE_STOP_GRACE_SECONDS=60 \
  CAPTURE_STOP_TERM_SECONDS=20 \
  "${CAPTURE_RUNNER}" "${capture_args[@]}" >"${command_log}" 2>&1 &
  CAPTURE_PID=$!
  ACTIVE_RELEASE_FILE="${release_file}"

  if ! wait_for_ready_file "${ready_file}" "${CAPTURE_PID}"; then
    return 1
  fi
  container_pid="$(read_ready_value component_container_pid "${ready_file}")"
  if [[ ! ${container_pid} =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: attach-ready file has no valid component_container_pid: ${ready_file}" >&2
    return 1
  fi
  if ! start_profiler_attach "${container_pid}" "${trace_dir}" "${profiler_log}"; then
    return 1
  fi

  touch "${release_file}"
  set +e
  if ! wait_for_playback_done_file \
    "${playback_done_file}" \
    "${CAPTURE_PID}" \
    "${PROFILER_PID}"; then
    wait_for_playback_status=1
  else
    wait_for_playback_status=0
  fi
  if [[ -n ${PROFILER_PID} ]] && kill -0 "${PROFILER_PID}" 2>/dev/null; then
    echo "Detaching rocprofv3 from ${container_pid}..."
    kill -INT "${PROFILER_PID}" 2>/dev/null || true
  fi
  if [[ -n ${PROFILER_PID} ]]; then
    wait "${PROFILER_PID}"
    profiler_status=$?
  else
    profiler_status=1
  fi
  touch "${detach_complete_file}"
  set -e
  PROFILER_PID=""
  wait "${CAPTURE_PID}"
  capture_status=$?
  CAPTURE_PID=""
  if ((wait_for_playback_status != 0)); then
    return 1
  fi
  if ((capture_status != 0)); then
    echo "ERROR: ${transport} fixed-input capture failed." >&2
    return 1
  fi
  if ((profiler_status != 0)); then
    echo "ERROR: rocprofv3 attach/detach failed for ${transport}." >&2
    return 1
  fi

  if ! wait_for_trace_outputs "${trace_dir}"; then
    return 1
  fi
  mapfile -t trace_files < <(find "${trace_dir}" -type f -name '*.json' -size +0c -print | sort)
  if [[ ${#trace_files[@]} -eq 0 ]]; then
    echo "ERROR: rocprofv3 produced no non-empty JSON files for ${transport}." >&2
    return 1
  fi
  printf '%s\n' "${trace_files[@]}" >"${trace_dir}/trace-files.txt"
  write_capture_manifest "${manifest_path}" "${lane_name}"
}

run_lane std std
run_lane managed managed

find_single_profile() {
  local pattern="$1"
  local label="$2"
  local matches=()
  mapfile -t matches < <(find "${ORT_ROOT}" -maxdepth 1 -type f -name "${pattern}" -size +0c -print)
  if [[ ${#matches[@]} -ne 1 ]]; then
    echo "ERROR: expected one ${label} ORT profile, found ${#matches[@]}." >&2
    return 1
  fi
  echo "${matches[0]}"
}

STD_PROFILE="$(find_single_profile 'std_*.json' 'std')"
MANAGED_PROFILE="$(find_single_profile 'managed_*.json' 'Managed')"
STD_BINDING="${BINDING_ROOT}/std.json"
MANAGED_BINDING="${BINDING_ROOT}/managed.json"
for required_path in "${STD_BINDING}" "${MANAGED_BINDING}"; do
  if [[ ! -s ${required_path} ]]; then
    echo "ERROR: required pointer/lifetime evidence is missing: ${required_path}" >&2
    exit 1
  fi
done

bag_message_count() {
  ros2 bag info "$1" | awk '/^Messages:/ {print $2; exit}'
}
STD_FRAMES="$(bag_message_count "${BAG_ROOT}/std")"
MANAGED_FRAMES="$(bag_message_count "${BAG_ROOT}/managed")"
if [[ ! ${STD_FRAMES} =~ ^[1-9][0-9]*$ || ! ${MANAGED_FRAMES} =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: failed to read positive AMD frame counts." >&2
  exit 1
fi

mapfile -t STD_TRACE_FILES <"${TRACE_ROOT}/std/trace-files.txt"
mapfile -t MANAGED_TRACE_FILES <"${TRACE_ROOT}/managed/trace-files.txt"
STD_MANIFEST="${TRACE_ROOT}/std.capture-manifest.json"
MANAGED_MANIFEST="${TRACE_ROOT}/managed.capture-manifest.json"
for manifest_path in "${STD_MANIFEST}" "${MANAGED_MANIFEST}"; do
  if [[ ! -s ${manifest_path} ]]; then
    echo "ERROR: required ROCprofiler capture manifest is missing: ${manifest_path}" >&2
    exit 1
  fi
done
STD_TRACE_ARGS=()
MANAGED_TRACE_ARGS=()
for trace_file in "${STD_TRACE_FILES[@]}"; do
  STD_TRACE_ARGS+=(--std-trace "${trace_file}")
done
for trace_file in "${MANAGED_TRACE_FILES[@]}"; do
  MANAGED_TRACE_ARGS+=(--managed-trace "${trace_file}")
done

echo "Auditing AMD MIGraphX provider placement..."
ros2 run isaac_ros_onnx_inference summarize_ort_profile.py \
  "${STD_PROFILE}" "${MANAGED_PROFILE}" \
  --expected-provider MIGraphXExecutionProvider \
  --ignore-cpu-provider-layout \
  --output-json "${REPORT_ROOT}/ort_provider_layout.json" \
  2>&1 | tee "${LOG_ROOT}/ort_provider_layout.log"

echo "Writing std and Managed rocprof self reports..."
STD_SELF_TRACE_ARGS=()
MANAGED_SELF_TRACE_ARGS=()
for trace_file in "${STD_TRACE_FILES[@]}"; do
  STD_SELF_TRACE_ARGS+=(--trace "${trace_file}")
done
for trace_file in "${MANAGED_TRACE_FILES[@]}"; do
  MANAGED_SELF_TRACE_ARGS+=(--trace "${trace_file}")
done
ros2 run isaac_ros_onnx_inference analyze_rocprof_traces.py \
  --self-report --lane std --frame-count "${STD_FRAMES}" \
  --manifest "${STD_MANIFEST}" \
  "${STD_SELF_TRACE_ARGS[@]}" \
  --output-json "${REPORT_ROOT}/std_self.json" \
  2>&1 | tee "${LOG_ROOT}/std_self.log"
ros2 run isaac_ros_onnx_inference analyze_rocprof_traces.py \
  --self-report --lane managed --frame-count "${MANAGED_FRAMES}" \
  --manifest "${MANAGED_MANIFEST}" \
  "${MANAGED_SELF_TRACE_ARGS[@]}" \
  --output-json "${REPORT_ROOT}/managed_self.json" \
  2>&1 | tee "${LOG_ROOT}/managed_self.log"

PAYLOAD_ARGS=()
for payload_size in "${PAYLOAD_SIZES[@]}"; do
  PAYLOAD_ARGS+=(--payload-size "${payload_size}")
done

echo "Writing paired AMD memory-copy audit..."
set +e
ros2 run isaac_ros_onnx_inference analyze_rocprof_traces.py \
  "${STD_TRACE_ARGS[@]}" "${MANAGED_TRACE_ARGS[@]}" \
  --std-frames "${STD_FRAMES}" --managed-frames "${MANAGED_FRAMES}" \
  --std-manifest "${STD_MANIFEST}" --managed-manifest "${MANAGED_MANIFEST}" \
  --std-binding-report "${STD_BINDING}" \
  --managed-binding-report "${MANAGED_BINDING}" \
  "${PAYLOAD_ARGS[@]}" \
  --output-json "${REPORT_ROOT}/rocprof_comparison.json" \
  2>&1 | tee "${LOG_ROOT}/rocprof_comparison.log"
TRACE_STATUS=${PIPESTATUS[0]}
set -e

echo "Comparing std and Managed detection outputs with stamp matching..."
set +e
ros2 run isaac_ros_detection_validation compare_detection2d_bags.py \
  --reference-bag "${BAG_ROOT}/std" \
  --candidate-bag "${BAG_ROOT}/managed" \
  --match-policy stamp \
  --report-only \
  --output-json "${REPORT_ROOT}/detection_comparison.json" \
  2>&1 | tee "${LOG_ROOT}/detection_comparison.log"
DETECTION_STATUS=${PIPESTATUS[0]}
set -e

FINAL_STATUS="PASS"
if ((TRACE_STATUS != 0)); then
  FINAL_STATUS="INCONCLUSIVE"
  if ((TRACE_STATUS == 2)); then
    FINAL_STATUS="TOOLING_ERROR"
  fi
  if [[ ${FINAL_STATUS} != TOOLING_ERROR &&
    -s ${REPORT_ROOT}/rocprof_comparison.json ]] &&
    grep -q '"status": "FAIL"' "${REPORT_ROOT}/rocprof_comparison.json"; then
    FINAL_STATUS="FAIL"
  fi
fi
if ((DETECTION_STATUS != 0)) && [[ ${FINAL_STATUS} != TOOLING_ERROR ]]; then
  FINAL_STATUS="FAIL"
fi

{
  echo "${MODEL} AMD transport audit: ${FINAL_STATUS}"
  echo "std captured frames: ${STD_FRAMES}"
  echo "Managed captured frames: ${MANAGED_FRAMES}"
  echo "rocprof command: rocprofv3 --attach <component_container_mt PID> --memory-copy-trace --kernel-trace --output-format json"
  echo "capture manifests: ${STD_MANIFEST}, ${MANAGED_MANIFEST}"
  echo "rocprof attach mode: ${ROCPROF_ATTACH_MODE}"
  echo "rocprof target opt-in: ROCP_TOOL_ATTACH=1 (scoped to each capture lane)"
  echo "Trace output is required to be stable before parsing."
  echo "Warm-up is excluded: rocprofv3 attaches only after the warm-up handshake."
  echo "Direct Managed HIP production lane expects no application TensorList staging copies."
  echo "Use --require-adapter-directions only for the staged-control lane."
  echo "Managed-only tensor-sized inference-boundary copies fail; ambiguous payload kernels are INCONCLUSIVE."
  echo "Kernel names containing copy/memcpy/blit are diagnostic unless trace evidence resolves their role."
  echo "CPU fallback is recorded in the ORT provider report and is not a closure failure."
  echo "No claim is made for copies outside the Managed TensorList inference boundary."
} | tee "${AUDIT_ROOT}/summary.txt"

if [[ ${FINAL_STATUS} == TOOLING_ERROR ]]; then
  echo "TOOLING_ERROR: AMD transport audit parser rejected capture input." >&2
  exit 2
fi
if [[ ${FINAL_STATUS} != PASS ]]; then
  echo "${FINAL_STATUS}: AMD transport audit written to ${AUDIT_ROOT}" >&2
  exit 1
fi
echo "PASS: AMD transport audit written to ${AUDIT_ROOT}"
