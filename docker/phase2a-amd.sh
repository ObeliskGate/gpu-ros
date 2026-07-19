#!/usr/bin/env bash
set -euo pipefail
trap 'echo "ERROR: phase2a setup failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker-compose.phase2a-amd.yaml"
ENV_FILE="${ROOT_DIR}/.env.phase2a-amd"
COMMON_DIR="${ROOT_DIR}/third_party/isaac_ros_common"
COMMON_URL="https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git"
COMMON_REF="v4.4-0"

cd "${ROOT_DIR}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
  COMPOSE=(docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}")
else
  COMPOSE=(docker compose -f "${COMPOSE_FILE}")
fi

prepare_interfaces() {
  mkdir -p "${ROOT_DIR}/third_party"
  if [[ ! -d "${COMMON_DIR}/.git" ]]; then
    git clone --branch "${COMMON_REF}" --depth 1 \
      "${COMMON_URL}" "${COMMON_DIR}"
  fi
  [[ -f "${COMMON_DIR}/isaac_ros_tensor_list_interfaces/package.xml" ]] || {
    echo "ERROR: official isaac_ros_tensor_list_interfaces was not found." >&2
    exit 1
  }
}

check_host() {
  command -v docker >/dev/null
  docker compose version >/dev/null
  [[ -e /dev/kfd ]] || {
    echo "ERROR: /dev/kfd is not available on the host." >&2
    exit 1
  }
  [[ -d /dev/dri ]] || {
    echo "ERROR: /dev/dri is not available on the host." >&2
    exit 1
  }

  if [[ -z "${AMD_GPU_TARGETS:-}" ]] && command -v rocminfo >/dev/null; then
    AMD_GPU_TARGETS="$(
      rocminfo | awk '
        /^[[:space:]]*Name:[[:space:]]+gfx[0-9]/ && target == "" {
          target = $2
        }
        END {print target}
      '
    )"
  fi
  AMD_GPU_TARGETS="${AMD_GPU_TARGETS:-gfx942}"
  export AMD_GPU_TARGETS
  echo "AMD GPU build target: ${AMD_GPU_TARGETS}"
}

build_workspace() {
  "${COMPOSE[@]}" exec -T amd bash -lc '
    colcon build \
      --base-paths \
        migrated_packages \
        third_party/isaac_ros_common \
      --packages-select \
        isaac_ros_common \
        isaac_ros_tensor_list_interfaces \
        isaac_ros_onnx_inference \
        isaac_ros_rtdetr_std
  '
}

verify_workspace() {
  "${COMPOSE[@]}" exec -T amd bash -lc '
    source install/setup.bash
    test -f /opt/onnxruntime/include/onnxruntime_cxx_api.h
    test -e /opt/onnxruntime/lib/libonnxruntime.so
    rocminfo | grep -m1 "Name:.*gfx"
    ros2 component types | grep -E "OnnxInference|RtDetr"
  '
}

case "${1:-bootstrap}" in
  bootstrap)
    check_host
    prepare_interfaces
    "${COMPOSE[@]}" build amd
    "${COMPOSE[@]}" up -d amd
    build_workspace
    verify_workspace
    ;;
  build)
    check_host
    prepare_interfaces
    "${COMPOSE[@]}" build amd
    ;;
  up)
    check_host
    prepare_interfaces
    "${COMPOSE[@]}" up -d amd
    ;;
  colcon)
    prepare_interfaces
    build_workspace
    verify_workspace
    ;;
  shell)
    "${COMPOSE[@]}" exec amd bash
    ;;
  stop)
    "${COMPOSE[@]}" stop amd
    ;;
  down)
    "${COMPOSE[@]}" down
    ;;
  *)
    echo "Usage: $0 {bootstrap|build|up|colcon|shell|stop|down}" >&2
    exit 2
    ;;
esac
