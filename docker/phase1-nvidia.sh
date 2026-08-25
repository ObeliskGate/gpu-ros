#!/usr/bin/env bash
set -euo pipefail
trap 'echo "ERROR: Phase 1 NVIDIA setup failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker-compose.yaml"
EXPECTED_BASE="nvcr.io/nvidia/isaac/ros:isaac_ros_89df02a734965ed64c227ef531c09d65-amd64"
ORT_LOCK="${ROOT_DIR}/config/onnxruntime.lock"
# shellcheck source=/dev/null
source "${ORT_LOCK}"
MANAGED_DIR="${GPU_ROS_MANAGED_DIR:-${ROOT_DIR}/../gpu_ros_managed}"
EXPECTED_OBJECT_DETECTION_COMMIT="060ced887bd8a3a0be60b1fa454365942eefd128"
EXPECTED_BENCHMARK_COMMIT="f46699e124262c5bfb6f00099061f6718f026b3f"

cd "${ROOT_DIR}"
COMPOSE=(docker compose -f "${COMPOSE_FILE}")

check_pins() {
  grep -Fqx "FROM ${EXPECTED_BASE}" Dockerfile || {
    echo "ERROR: Dockerfile no longer uses the Phase 1 Isaac ROS image." >&2
    exit 1
  }
  grep -Fq "COPY config/onnxruntime.lock" Dockerfile || {
    echo "ERROR: Dockerfile no longer reads the ONNX Runtime lock." >&2
    exit 1
  }
}

report_repo_state() {
  local label="$1"
  local repository="$2"
  local untracked_paths
  echo "${label} HEAD: $(git -C "${repository}" rev-parse HEAD)"
  echo "${label} diff HEAD --binary sha256: $(
    git -C "${repository}" diff HEAD --binary | sha256sum | awk '{print $1}')"
  untracked_paths="$(git -C "${repository}" ls-files --others --exclude-standard)"
  if [[ -n "${untracked_paths}" ]]; then
    echo "${label} untracked paths:"
    printf '%s\n' "${untracked_paths}"
    echo "${label} untracked content sha256: $(
      git -C "${repository}" ls-files --others --exclude-standard -z |
      while IFS= read -r -d '' path; do
        sha256sum "${repository}/${path}"
      done | sha256sum | awk '{print $1}')"
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
  [[ -n "${OVG_NVIDIA_EXTERNAL_SOURCE_ROOT:-}" ]] || {
    echo "ERROR: set OVG_NVIDIA_EXTERNAL_SOURCE_ROOT to an explicit external checkout." >&2
    exit 1
  }
  [[ -d "${OVG_NVIDIA_EXTERNAL_SOURCE_ROOT}/isaac_ros_object_detection/.git" ]] || {
    echo "ERROR: external isaac_ros_object_detection checkout is missing." >&2
    exit 1
  }
  [[ -d "${OVG_NVIDIA_EXTERNAL_SOURCE_ROOT}/isaac_ros_benchmark/.git" ]] || {
    echo "ERROR: external isaac_ros_benchmark checkout is missing." >&2
    exit 1
  }
  local object_detection_commit benchmark_commit
  object_detection_commit="$(
    git -C "${OVG_NVIDIA_EXTERNAL_SOURCE_ROOT}/isaac_ros_object_detection" rev-parse HEAD)"
  benchmark_commit="$(
    git -C "${OVG_NVIDIA_EXTERNAL_SOURCE_ROOT}/isaac_ros_benchmark" rev-parse HEAD)"
  [[ "${object_detection_commit}" == "${EXPECTED_OBJECT_DETECTION_COMMIT}" ]] || {
    echo "ERROR: external isaac_ros_object_detection is not at the manifest commit." >&2
    exit 1
  }
  [[ "${benchmark_commit}" == "${EXPECTED_BENCHMARK_COMMIT}" ]] || {
    echo "ERROR: external isaac_ros_benchmark is not at the manifest commit." >&2
    exit 1
  }
  [[ -f "${MANAGED_DIR}/gpu_ros_managed_core/package.xml" ]] || {
    echo "ERROR: gpu_ros_managed sibling checkout is missing: ${MANAGED_DIR}" >&2
    exit 1
  }
  report_repo_state "amd_ros_object_detection" "${ROOT_DIR}"
  report_repo_state "gpu_ros_managed" "${MANAGED_DIR}"
  echo "Sibling commits are recorded, not allowlisted; compatibility is decided by build/tests."
  echo "NVIDIA Phase 1 environment: Isaac ROS pinned image, ORT ${ORT_VERSION}"
}

verify_container() {
  "${COMPOSE[@]}" exec -T dev bash -lc '
    source /opt/ros/jazzy/setup.bash
    test "${ROS2_BENCHMARK_OVERRIDE_ASSETS_ROOT}" = \
      /workspaces/isaac_ros-dev/assets
    test -d "${ROS2_BENCHMARK_OVERRIDE_ASSETS_ROOT}"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    test -f /opt/onnxruntime/include/onnxruntime_cxx_api.h
    test -e /opt/tritonserver/backends/onnxruntime/libonnxruntime.so
    ros2 pkg prefix isaac_ros_rtdetr_benchmark
    ros2 pkg prefix isaac_ros_grounding_dino_benchmark
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
    source /opt/ros/jazzy/setup.bash
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
    source /opt/ros/jazzy/setup.bash
    colcon build
  '
  verify_release_caches
}

case "${1:-bootstrap}" in
  bootstrap)
    check_host
    "${COMPOSE[@]}" build dev
    "${COMPOSE[@]}" up -d dev
    verify_container
    ;;
  build)
    check_host
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
      source /opt/ros/jazzy/setup.bash
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
