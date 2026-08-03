#!/usr/bin/env bash
set -euo pipefail

source_setup() {
  set +u
  # shellcheck disable=SC1090
  source "$1"
  set -u
}

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
