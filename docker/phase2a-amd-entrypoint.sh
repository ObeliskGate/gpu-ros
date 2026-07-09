#!/usr/bin/env bash
set -e

if [[ -n "${ROS_DISTRO:-}" && -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
fi

if [[ -f "/workspaces/amd_ros_object_detection/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "/workspaces/amd_ros_object_detection/install/setup.bash"
fi

exec "$@"
