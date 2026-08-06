#!/usr/bin/env bash
set -euo pipefail

source_setup() {
  set +u
  # shellcheck disable=SC1090
  source "$1"
  set -u
}

validate_external_ort() {
  local root="${OVG_ORT_ROOT%/}"
  [[ -n "${root}" ]] || return 0
  [[ -d "${root}" ]] || {
    echo "ERROR: OVG_ORT_ROOT does not exist: ${root}" >&2
    return 1
  }
  [[ -f "${root}/include/onnxruntime_cxx_api.h" ]] || {
    echo "ERROR: external ORT install is incomplete (missing C++ headers): ${root}" >&2
    return 1
  }
  local required_library
  for required_library in \
    libonnxruntime.so \
    libonnxruntime_providers_shared.so \
    libonnxruntime_providers_migraphx.so; do
    [[ -e "${root}/lib/${required_library}" ]] || {
      echo "ERROR: external ORT install is incomplete (missing ${required_library}): ${root}" >&2
      return 1
    }
  done
  [[ -f "${root}/.ovg-ort-fingerprint" ]] || {
    echo "ERROR: external ORT install is missing .ovg-ort-fingerprint: ${root}" >&2
    return 1
  }
  local expected_fingerprint="${root##*/}"
  local recorded_fingerprint
  recorded_fingerprint="$(tr -d '\r\n' < "${root}/.ovg-ort-fingerprint")"
  [[ "${recorded_fingerprint}" == "${expected_fingerprint}" ]] || {
    echo "ERROR: external ORT fingerprint marker does not match install path: ${root}" >&2
    return 1
  }

  export ONNXRUNTIME_ROOT="${root}"
  export ONNXRUNTIME_INCLUDE_DIR="${root}/include"
  export ONNXRUNTIME_LIBRARY="${root}/lib/libonnxruntime.so"
  export LD_LIBRARY_PATH="${root}/lib:${LD_LIBRARY_PATH:-}"
}

if [[ -n "${OVG_ORT_ROOT:-}" ]]; then
  validate_external_ort
fi

if [[ -n "${ROS_DISTRO:-}" && -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  source_setup "/opt/ros/${ROS_DISTRO}/setup.bash"
fi

if [[ -f /opt/ros2_benchmark/setup.bash ]]; then
  source_setup /opt/ros2_benchmark/setup.bash
fi
export PATH="/workspaces/amd_ros_object_detection/tools:${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"

mkdir -p "${OVG_ASSETS_ROOT:-/workspaces/ovg-assets}" \
  "${OVG_CACHE_ROOT:-/workspaces/ovg-cache}" \
  "${OVG_RESULTS_ROOT:-/workspaces/ovg-results}"

if [[ -n "${ORT_MIGRAPHX_MODEL_CACHE_PATH:-}" ]]; then
  mkdir -p "${ORT_MIGRAPHX_MODEL_CACHE_PATH}"
elif [[ -n "${OVG_CACHE_ROOT:-}" ]]; then
  export ORT_MIGRAPHX_MODEL_CACHE_PATH="${OVG_CACHE_ROOT}/migraphx/${OVG_IMAGE_FINGERPRINT:-unresolved}"
  mkdir -p "${ORT_MIGRAPHX_MODEL_CACHE_PATH}"
fi

if [[ -f "/workspaces/amd_ros_object_detection/install/setup.bash" ]]; then
  source_setup "/workspaces/amd_ros_object_detection/install/setup.bash"
fi

exec "$@"
