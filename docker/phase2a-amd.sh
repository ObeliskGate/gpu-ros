#!/usr/bin/env bash
set -euo pipefail
trap 'echo "ERROR: phase2a setup failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker-compose.phase2a-amd.yaml"
ENV_FILE="${ROOT_DIR}/.env.phase2a-amd"
COMMON_DIR="${ROOT_DIR}/third_party/isaac_ros_common"
COMMON_URL="https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git"
COMMON_REF="v4.4-0"
INTERFACES_DIR="${COMMON_DIR}/isaac_ros_tensor_list_interfaces"
INTERFACES_PATCH="${ROOT_DIR}/docker/patches/isaac-ros-common-v4.4-tensor-list-standalone.patch"

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
    base_paths=(
      migrated_packages/isaac_ros_onnx_inference
      migrated_packages/isaac_ros_rtdetr_std
      third_party/isaac_ros_common/isaac_ros_tensor_list_interfaces
    )
    expected="$(printf "%s\n" \
      isaac_ros_onnx_inference \
      isaac_ros_rtdetr_std \
      isaac_ros_tensor_list_interfaces | LC_ALL=C sort)"
    discovered="$(colcon list --base-paths "${base_paths[@]}" --names-only | LC_ALL=C sort)"
    if [[ "${discovered}" != "${expected}" ]]; then
      echo "ERROR: unexpected Phase 2a package discovery:" >&2
      printf "%s\n" "${discovered}" >&2
      exit 1
    fi

    colcon build \
      --base-paths "${base_paths[@]}" \
      --packages-select \
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
    grep -qx "BUILD_NITROS_TRANSPORT:BOOL=OFF" \
      build/isaac_ros_onnx_inference/CMakeCache.txt
    grep -qx "ORT_ENABLE_CUDA:BOOL=OFF" \
      build/isaac_ros_onnx_inference/CMakeCache.txt
    grep -qx "ORT_ENABLE_ROCM:BOOL=OFF" \
      build/isaac_ros_onnx_inference/CMakeCache.txt
    grep -qx "ORT_ENABLE_MIGRAPHX:BOOL=ON" \
      build/isaac_ros_onnx_inference/CMakeCache.txt

    forbidden="NEEDED.*\\[(libcuda|libcudart|libcublas|libcudnn|libnvrtc|libnvinfer|libnvonnxparser|libgxf|[^]]*nitros)"
    while IFS= read -r -d "" library; do
      if readelf -d "${library}" | grep -Eiq "${forbidden}"; then
        echo "ERROR: NVIDIA runtime dependency found in ${library}:" >&2
        readelf -d "${library}" | grep -Ei "${forbidden}" >&2
        exit 1
      fi
    done < <(find \
      /opt/onnxruntime/lib \
      install/isaac_ros_onnx_inference/lib \
      install/isaac_ros_rtdetr_std/lib \
      -type f -name "*.so*" -print0)

    rocminfo | grep -m1 "Name:.*gfx"
    ros2 component types | grep -E "OnnxInference|RtDetr"
    echo "Verified: Phase 2a target has no CUDA, TensorRT, NITROS, or GXF linkage."
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
