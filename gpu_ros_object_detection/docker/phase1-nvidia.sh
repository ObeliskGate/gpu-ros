#!/usr/bin/env bash
set -euo pipefail
trap 'echo "ERROR: Phase 1 NVIDIA setup failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/gpu_ros_object_detection/docker/docker-compose.yaml"
ORT_LOCK="${ROOT_DIR}/gpu_ros_object_detection/config/onnxruntime.lock"
# shellcheck source=/dev/null
source "${ORT_LOCK}"
GPU_ROS_REPO_ROOT="${GPU_ROS_REPO_ROOT:-${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}/src/gpu-ros}"
export GPU_ROS_REPO_ROOT

cd "${ROOT_DIR}" || exit
COMPOSE=(docker compose -f "${COMPOSE_FILE}")

check_pins() {
  grep -Fqx "ARG ISAAC_ROS_BASE_IMAGE" gpu_ros_object_detection/docker/Dockerfile || {
    echo "ERROR: NVIDIA Dockerfile must consume ISAAC_ROS_BASE_IMAGE." >&2
    exit 1
  }
  grep -Fq "ISAAC_ROS_BASE_IMAGE: \${ISAAC_ROS_BASE_IMAGE:-}" \
    gpu_ros_object_detection/docker/docker-compose.yaml || {
    echo "ERROR: NVIDIA Compose must pass the optional image build argument without blocking no-build commands." >&2
    exit 1
  }
  grep -Fq "COPY gpu_ros_object_detection/config/onnxruntime.lock" gpu_ros_object_detection/docker/Dockerfile || {
    echo "ERROR: Dockerfile no longer reads the ONNX Runtime lock." >&2
    exit 1
  }
}

require_base_image_for_build() {
  local image="${ISAAC_ROS_BASE_IMAGE:-}"
  [[ "${image}" =~ ^nvcr\.io/nvidia/isaac/ros(:[^@[:space:]]+)?@sha256:[0-9a-f]{64}$ ]] || {
    echo "BLOCKED: set ISAAC_ROS_BASE_IMAGE to the official Isaac ROS 5.0 reference including its verified sha256 digest before building." >&2
    return 1
  }
}

report_repo_state() {
  local label="$1"
  local repository="$2"
  local untracked_paths
  echo "${label} HEAD: $(git -C "${repository}" rev-parse HEAD)"
  echo "${label} diff HEAD --binary sha256: $(
    git -C "${repository}" diff HEAD --binary | sha256sum | awk '{print $1}'
  )"
  untracked_paths="$(git -C "${repository}" ls-files --others --exclude-standard)"
  if [[ -n "${untracked_paths}" ]]; then
    echo "${label} untracked paths:"
    printf '%s\n' "${untracked_paths}"
    echo "${label} untracked content sha256: $(
      git -C "${repository}" ls-files --others --exclude-standard -z |
        while IFS= read -r -d '' path; do
          sha256sum "${repository}/${path}"
        done | sha256sum | awk '{print $1}'
    )"
  else
    echo "${label} untracked paths: none"
    echo "${label} untracked content sha256: none"
  fi
}

check_host() {
  command -v docker >/dev/null
  docker compose version >/dev/null
  command -v nvidia-smi >/dev/null
  nvidia-smi >/dev/null
  check_pins
  report_repo_state "gpu-ros" "${ROOT_DIR}"
  echo "Monorepo commit is recorded; compatibility is decided by build/tests."
  echo "NVIDIA Phase 1 runtime contract: Lyrical, native TensorList, ORT ${ORT_VERSION}"
}

verify_container() {
  "${COMPOSE[@]}" exec -T dev bash -lc '
    set -euo pipefail
    source /opt/ros/lyrical/setup.bash
    test "${ROS_DISTRO}" = lyrical
    test "${GPU_ROS_NVIDIA_PROFILE}" = 1
    test "${ROS2_BENCHMARK_OVERRIDE_ASSETS_ROOT}" = \
      /workspaces/isaac_ros-dev/assets
    test -d "${ROS2_BENCHMARK_OVERRIDE_ASSETS_ROOT}"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    test -f /opt/onnxruntime/include/onnxruntime_cxx_api.h
    test -e /opt/tritonserver/backends/onnxruntime/libonnxruntime.so
    for package in \
      rclcpp rosidl_buffer cuda_buffer cuda_buffer_backend \
      isaac_ros_tensor_list_interfaces isaac_ros_benchmark ros2_benchmark \
      isaac_ros_image_proc isaac_ros_tensor_proc isaac_ros_rtdetr \
      isaac_ros_yolov8 isaac_ros_tensor_rt; do
      ros2 pkg prefix "${package}"
    done
  '
}

verify_release_caches() {
  "${COMPOSE[@]}" exec -T dev bash -lc '
    set -e
    for package in \
      gpu_ros_onnx_inference \
      gpu_ros_rtdetr \
      gpu_ros_yolov8; do
      cache="/workspaces/isaac_ros-dev/build/${package}/CMakeCache.txt"
      test -f "${cache}"
      grep -Fqx "CMAKE_BUILD_TYPE:STRING=Release" "${cache}" || {
        echo "ERROR: ${package} was not built with CMAKE_BUILD_TYPE=Release" >&2
        exit 1
      }
    done
    grep -Fqx "BUILD_NATIVE_TENSOR_LIST_TRANSPORT:BOOL=ON" \
      /workspaces/isaac_ros-dev/build/gpu_ros_onnx_inference/CMakeCache.txt || {
        echo "ERROR: native TensorList transport was not enabled for the NVIDIA profile." >&2
        exit 1
      }
    test "${ROS_DISTRO}" = lyrical
    test "${GPU_ROS_NVIDIA_PROFILE}" = 1
    source /opt/ros/lyrical/setup.bash
    source install/setup.bash
    ros2 pkg prefix gpu_ros_managed_core
    ros2 pkg prefix gpu_ros_managed_cuda
    ros2 pkg prefix gpu_ros_managed_tensor_bundle
    ros2 pkg prefix gpu_ros_tensor_bundle_msgs
    ros2 pkg prefix gpu_ros_nvidia_tensor_bundle_compat
    ros2 pkg prefix gpu_ros_onnx_inference
  '
}

build_workspace() {
  "${COMPOSE[@]}" exec -T dev bash -lc '
    test "${ROS_DISTRO}" = lyrical
    test "${GPU_ROS_NVIDIA_PROFILE}" = 1
    source /opt/ros/lyrical/setup.bash
    colcon build
  '
  verify_release_caches
}

case "${1:-bootstrap}" in
  bootstrap)
    check_host
    require_base_image_for_build
    "${COMPOSE[@]}" build dev
    "${COMPOSE[@]}" up -d dev
    verify_container
    ;;
  build)
    check_host
    require_base_image_for_build
    "${COMPOSE[@]}" build dev
    ;;
  up)
    check_host
    "${COMPOSE[@]}" up -d dev
    verify_container
    ;;
  colcon)
    build_workspace
    ;;
  verify)
    verify_container
    verify_release_caches
    ;;
  shell)
    "${COMPOSE[@]}" exec dev bash -lc '
      source /opt/ros/lyrical/setup.bash
      if [[ -f install/setup.bash ]]; then source install/setup.bash; fi
      exec bash
    '
    ;;
  stop)
    "${COMPOSE[@]}" stop dev
    ;;
  down)
    "${COMPOSE[@]}" down
    ;;
  *)
    echo "Usage: $0 {bootstrap|build|up|colcon|verify|shell|stop|down}" >&2
    exit 2
    ;;
esac
