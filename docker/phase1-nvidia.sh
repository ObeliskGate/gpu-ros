#!/usr/bin/env bash
set -euo pipefail
trap 'echo "ERROR: Phase 1 NVIDIA setup failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker-compose.yaml"
ENV_FILE="${ROOT_DIR}/.env"
EXPECTED_BASE="nvcr.io/nvidia/isaac/ros:isaac_ros_89df02a734965ed64c227ef531c09d65-amd64"
EXPECTED_ORT="1.23.1"
MANAGED_DIR="${GPU_ROS_MANAGED_DIR:-${ROOT_DIR}/../gpu_ros_managed}"
EXPECTED_MANAGED_COMMIT="96c4bcf93dca522830efe7c5b6573daf3be482db"

cd "${ROOT_DIR}"
COMPOSE=(docker compose -f "${COMPOSE_FILE}")

check_pins() {
  grep -Fqx "FROM ${EXPECTED_BASE}" Dockerfile || {
    echo "ERROR: Dockerfile no longer uses the Phase 1 Isaac ROS image." >&2
    exit 1
  }
  grep -Fqx "ARG ORT_VERSION=${EXPECTED_ORT}" Dockerfile || {
    echo "ERROR: Dockerfile no longer pins ORT ${EXPECTED_ORT}." >&2
    exit 1
  }
}

check_host() {
  command -v docker >/dev/null
  docker compose version >/dev/null
  command -v nvidia-smi >/dev/null
  nvidia-smi >/dev/null
  check_pins
  [[ -f "${MANAGED_DIR}/gpu_ros_managed_core/package.xml" ]] || {
    echo "ERROR: gpu_ros_managed sibling checkout is missing: ${MANAGED_DIR}" >&2
    exit 1
  }
  local managed_commit
  managed_commit="$(git -C "${MANAGED_DIR}" rev-parse HEAD)"
  echo "gpu_ros_managed commit: ${managed_commit}"
  [[ "${managed_commit}" = "${EXPECTED_MANAGED_COMMIT}" ]] || {
    echo "ERROR: gpu_ros_managed is not at the pinned commit." >&2
    exit 1
  }
  echo "NVIDIA Phase 1 environment: Isaac ROS pinned image, ORT ${EXPECTED_ORT}"
}

prepare_env() {
  if [[ ! -e "${ENV_FILE}" ]]; then
    touch "${ENV_FILE}"
  fi
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
      isaac_ros_onnx_inference \
      isaac_ros_rtdetr_std \
      isaac_ros_yolov8_std; do
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
    ros2 pkg prefix gpu_ros_managed_tensor_list
    ros2 pkg prefix isaac_ros_onnx_inference
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
    prepare_env
    "${COMPOSE[@]}" build dev
    "${COMPOSE[@]}" up -d dev
    verify_container
    ;;
  build)
    check_host
    prepare_env
    "${COMPOSE[@]}" build dev
    ;;
  up)
    check_host
    prepare_env
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
