#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${OVG_WORKSPACE_ROOT:-/workspaces/amd_ros_object_detection}"
COLCON_DEFAULTS_FILE="${WORKSPACE_ROOT}/docker/colcon-defaults-phase2a-amd.yaml"
export OVG_WORKSPACE_ROOT="${WORKSPACE_ROOT}"
export COLCON_DEFAULTS_FILE

if [[ ! -f "${COLCON_DEFAULTS_FILE}" ]]; then
  if [[ "${BASH_SOURCE[0]}" == "${WORKSPACE_ROOT}/docker/phase2a-amd-entrypoint.sh" ]]; then
    echo "ERROR: canonical AMD colcon defaults file is missing: ${COLCON_DEFAULTS_FILE}" >&2
    exit 1
  fi
  echo "WARNING: canonical AMD colcon defaults file is unavailable: ${COLCON_DEFAULTS_FILE}" >&2
fi

source_setup() {
  set +u
  # shellcheck disable=SC1090
  source "$1"
  set -u
}

resolve_hip_root() {
  local candidate=""
  unset OVG_ROCM_ROOT hip_ROOT

  command -v hipconfig >/dev/null 2>&1 || return 0
  candidate="$(hipconfig --path 2>/dev/null | awk 'NF {print $1; exit}')"
  [[ -n "${candidate}" ]] || return 0
  candidate="$(readlink -f -- "${candidate}" 2>/dev/null || true)"
  [[ -n "${candidate}" ]] || return 0
  [[ -f "${candidate}/lib/cmake/hip/hip-config.cmake" ]] || return 0

  export OVG_ROCM_ROOT="${candidate}"
  export hip_ROOT="${candidate}"
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

resolve_hip_root

if [[ -z "${OVG_ORT_ROOT:-}" && "${OVG_ORT_BUILD_MODE:-0}" != 1 ]]; then
  echo "ERROR: external ORT is required for every AMD build/validation run; /opt/onnxruntime in the SIF is legacy build-only content" >&2
  exit 1
fi

if [[ -n "${OVG_ORT_ROOT:-}" ]]; then
  validate_external_ort
fi

if [[ -n "${ROS_DISTRO:-}" && -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  source_setup "/opt/ros/${ROS_DISTRO}/setup.bash"
fi

if [[ -f /opt/ros2_benchmark/setup.bash ]]; then
  source_setup /opt/ros2_benchmark/setup.bash
fi
export PATH="${WORKSPACE_ROOT}/tools:${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"

mkdir -p "${OVG_ASSETS_ROOT:-/workspaces/ovg-assets}" \
  "${OVG_CACHE_ROOT:-/workspaces/ovg-cache}" \
  "${OVG_RESULTS_ROOT:-/workspaces/ovg-results}"

if [[ -n "${ORT_MIGRAPHX_MODEL_CACHE_PATH:-}" ]]; then
  mkdir -p "${ORT_MIGRAPHX_MODEL_CACHE_PATH}"
elif [[ -n "${OVG_CACHE_ROOT:-}" ]]; then
  export ORT_MIGRAPHX_MODEL_CACHE_PATH="${OVG_CACHE_ROOT}/migraphx/${OVG_IMAGE_FINGERPRINT:-unresolved}"
  mkdir -p "${ORT_MIGRAPHX_MODEL_CACHE_PATH}"
fi

if [[ -f "${WORKSPACE_ROOT}/install/setup.bash" ]]; then
  source_setup "${WORKSPACE_ROOT}/install/setup.bash"
fi

exec "$@"
