#!/usr/bin/env bash
set -euo pipefail
trap 'echo "ERROR: phase2a setup failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker-compose.phase2a-amd.yaml"
ENV_FILE="${ROOT_DIR}/.env.phase2a-amd"
COMMON_DIR="${ROOT_DIR}/third_party/isaac_ros_common"
COMMON_URL="https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git"
COMMON_REF="v4.5-0"
INTERFACES_DIR="${COMMON_DIR}/isaac_ros_tensor_list_interfaces"
INTERFACES_PATCH="${ROOT_DIR}/docker/patches/isaac-ros-common-v4.5-tensor-list-standalone.patch"
MANAGED_DIR="${GPU_ROS_MANAGED_DIR:-${ROOT_DIR}/../gpu_ros_managed}"
EXPECTED_MANAGED_COMMIT="d33381358f2259b9ad16e8847b5a6dccd0357ebc"

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
  [[ -f "${INTERFACES_DIR}/package.xml" ]] || {
    echo "ERROR: official isaac_ros_tensor_list_interfaces was not found." >&2
    exit 1
  }

  if git -C "${COMMON_DIR}" apply --reverse --check "${INTERFACES_PATCH}" \
      >/dev/null 2>&1; then
    return
  fi
  if ! git -C "${COMMON_DIR}" apply --check "${INTERFACES_PATCH}"; then
    echo "ERROR: TensorList standalone-build patch does not apply cleanly." >&2
    exit 1
  fi
  git -C "${COMMON_DIR}" apply "${INTERFACES_PATCH}"
}

check_host() {
  command -v docker >/dev/null
  docker compose version >/dev/null
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

verify_release_caches() {
  "${COMPOSE[@]}" exec -T amd bash -lc '
    set -e
    for package in isaac_ros_onnx_inference isaac_ros_rtdetr_std; do
      cache="/workspaces/amd_ros_object_detection/build/${package}/CMakeCache.txt"
      test -f "${cache}"
      grep -Fqx "CMAKE_BUILD_TYPE:STRING=Release" "${cache}" || {
        echo "ERROR: ${package} was not built with CMAKE_BUILD_TYPE=Release" >&2
        exit 1
      }
    done
  '
}

case "${1:-bootstrap}" in
  bootstrap)
    check_host
    prepare_interfaces
    "${COMPOSE[@]}" build amd
    "${COMPOSE[@]}" up -d amd
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
  shell)
    "${COMPOSE[@]}" exec amd bash -lc '
      set -e
      source "/opt/ros/${ROS_DISTRO}/setup.bash"
      source /opt/ros2_benchmark/setup.bash
      test -f "/opt/ros/${ROS_DISTRO}/share/ament_cmake_auto/cmake/ament_cmake_autoConfig.cmake" || {
        echo "ERROR: ament_cmake_auto is missing from the container image." >&2
        exit 1
      }
      if [[ -f /workspaces/amd_ros_object_detection/install/setup.bash ]]; then
        source /workspaces/amd_ros_object_detection/install/setup.bash
      fi
      exec bash -i
    '
    ;;
  verify)
    verify_release_caches
    ;;
  stop)
    "${COMPOSE[@]}" stop amd
    ;;
  down)
    "${COMPOSE[@]}" down
    ;;
  *)
    echo "Usage: $0 {bootstrap|build|up|shell|verify|stop|down}" >&2
    exit 2
    ;;
esac
