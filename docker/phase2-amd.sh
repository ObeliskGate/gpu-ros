#!/usr/bin/env bash
set -euo pipefail
trap 'echo "ERROR: phase2 AMD environment failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker-compose.phase2a-amd.yaml"
MANAGED_DIR="${GPU_ROS_MANAGED_DIR:-${ROOT_DIR}/../gpu_ros_managed}"
STATE_ROOT="${OVG_STATE_ROOT:-${ROOT_DIR}/.ovg}"
if [[ "${STATE_ROOT}" != /* ]]; then
  STATE_ROOT="${ROOT_DIR}/${STATE_ROOT}"
fi
ORT_STATE_HOST="${OVG_ORT_STATE_HOST:-${OVG_ORT_DIR:-${STATE_ROOT}/ort}}"
if [[ "${ORT_STATE_HOST}" != /* ]]; then
  ORT_STATE_HOST="${ROOT_DIR}/${ORT_STATE_HOST}"
fi
ORT_CONTAINER_ROOT="/workspaces/ovg-ort"
IMAGE_NAME="${OVG_IMAGE_NAME:-ovg-phase2-amd:local}"
APPTAINER_SIF="${OVG_APPTAINER_SIF:-${STATE_ROOT}phase2-amd-dev-<image-id>.sif}"
if [[ "${APPTAINER_SIF}" != /* ]]; then
  APPTAINER_SIF="${ROOT_DIR}/${APPTAINER_SIF}"
fi
APPTAINER_INSTANCE_NAME="${OVG_APPTAINER_INSTANCE_NAME:-ovg-phase2-amd}"
COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-ovg-phase2-amd}"

export COMPOSE_PROJECT_NAME
export OVG_WORKSPACE_ROOT="/workspaces/amd_ros_object_detection"
export OVG_ASSETS_ROOT="/workspaces/ovg-assets"
export OVG_CACHE_ROOT="/workspaces/ovg-cache"
export OVG_RESULTS_ROOT="/workspaces/ovg-results"
export OVG_ORT_STATE_ROOT="${ORT_CONTAINER_ROOT}"
export OVG_IMAGE_NAME="${IMAGE_NAME}"

die() { echo "ERROR: $*" >&2; exit 1; }

usage() {
  cat >&2 <<'EOF'
Usage: docker/phase2-amd.sh {bootstrap|build|up|colcon|verify|shell|stop|down}

Environment:
  OVG_RUNTIME=docker|apptainer       Select runtime explicitly.
  OVG_STATE_ROOT=/path               Persistent host state root.
  AMD_GPU_TARGETS=gfx...,gfx...      Build targets; comma or semicolon separated.
  OVG_ORT_ROOT=/workspaces/ovg-ort/install/<fingerprint>
                                      Exact external ORT install; empty uses /opt/onnxruntime.
  OVG_ORT_STATE_HOST=/path            Host directory bound at /workspaces/ovg-ort.
  OVG_APPTAINER_SIF=phase2-amd-dev-<image-id>.sif  SIF used by Apptainer.
  OVG_APPTAINER_IMAGE_URI=...        Optional URI for one-time SIF pull.
  OVG_APPTAINER_INSTANCE=1           Opt into an Apptainer instance for up/stop.
  OVG_PREPARE_ASSETS=1               Prepare assets during bootstrap.
EOF
}

repo_check() {
  [[ -f "${ROOT_DIR}/AGENTS.md" ]] || die "application repository is incomplete: ${ROOT_DIR}"
  [[ -f "${MANAGED_DIR}/gpu_ros_managed_core/package.xml" ]] || \
    die "gpu_ros_managed sibling checkout is missing: ${MANAGED_DIR}"
  echo "amd_ros_object_detection: $(git -C "${ROOT_DIR}" rev-parse HEAD)"
  echo "gpu_ros_managed: $(git -C "${MANAGED_DIR}" rev-parse HEAD)"
}

prepare_state() {
  mkdir -p \
    "${STATE_ROOT}/assets" \
    "${STATE_ROOT}/cache" \
    "${STATE_ROOT}/results" \
    "${STATE_ROOT}/images" \
    "${STATE_ROOT}/home" \
    "${ORT_STATE_HOST}"
}

normalise_targets() {
  local raw="${1//;/,}"
  raw="${raw// /}"
  tr ',' '\n' <<<"${raw}" \
    | awk 'NF && $0 ~ /^gfx[0-9a-fA-F]+$/ {print tolower($0)}' \
    | sort -u \
    | paste -sd, -
}

resolve_gpu_targets() {
  local targets="${AMD_GPU_TARGETS:-}"
  if [[ -n "${targets}" ]]; then
    targets="$(normalise_targets "${targets}")"
  elif command -v rocminfo >/dev/null 2>&1; then
    targets="$(rocminfo 2>/dev/null \
      | awk '$1 == "Name:" && $2 ~ /^gfx[0-9]/ {print $2}' \
      | sort -u \
      | paste -sd, -)"
  fi
  [[ -n "${targets}" ]] || die \
    "AMD_GPU_TARGETS is unset and rocminfo could not detect a gfx target; set AMD_GPU_TARGETS explicitly"
  AMD_GPU_TARGETS="${targets}"
  export AMD_GPU_TARGETS
  echo "Build GPU targets: ${AMD_GPU_TARGETS}"
}

docker_usable() {
  command -v docker >/dev/null 2>&1 || return 1
  docker info >/dev/null 2>&1 || return 1
  docker compose version >/dev/null 2>&1
}

select_runtime() {
  if [[ -n "${OVG_RUNTIME:-}" ]]; then
    case "${OVG_RUNTIME}" in
      docker) docker_usable || die "OVG_RUNTIME=docker was requested but Docker Compose/daemon is unavailable" ;;
      apptainer) command -v apptainer >/dev/null 2>&1 || die "OVG_RUNTIME=apptainer was requested but apptainer is unavailable" ;;
      *) die "OVG_RUNTIME must be docker or apptainer" ;;
    esac
    RUNTIME="${OVG_RUNTIME}"
  elif docker_usable; then
    RUNTIME=docker
  elif command -v apptainer >/dev/null 2>&1; then
    RUNTIME=apptainer
  else
    die "no usable Docker Compose daemon or Apptainer was found"
  fi
  export RUNTIME
  export OVG_RUNTIME_EFFECTIVE="${RUNTIME}"
  echo "Runtime: ${RUNTIME}"
}

compose_env() {
  local image_fingerprint="${1}"
  local workspace_fingerprint="${2}"
  export GPU_ROS_MANAGED_DIR="${MANAGED_DIR}"
  export OVG_ASSETS_DIR="${STATE_ROOT}/assets"
  export OVG_CACHE_DIR="${STATE_ROOT}/cache"
  export OVG_RESULTS_DIR="${STATE_ROOT}/results"
  export OVG_ORT_DIR="${ORT_STATE_HOST}"
  export OVG_BUILD_DIR="${STATE_ROOT}/build/${workspace_fingerprint}"
  export OVG_INSTALL_DIR="${STATE_ROOT}/install/${workspace_fingerprint}"
  export OVG_LOG_DIR="${STATE_ROOT}/log/${workspace_fingerprint}"
  export OVG_HOME_DIR="${STATE_ROOT}/home"
  export OVG_IMAGE_FINGERPRINT="${image_fingerprint}"
  export OVG_WORKSPACE_FINGERPRINT="${workspace_fingerprint}"
  export OVG_UID="$(id -u)"
  export OVG_GID="$(id -g)"
  mkdir -p "${OVG_BUILD_DIR}" "${OVG_INSTALL_DIR}" "${OVG_LOG_DIR}"
}

docker_fingerprint() {
  local id
  id="$(docker image inspect --format '{{.Id}}' "${IMAGE_NAME}" 2>/dev/null)" || return 1
  printf '%s\n' "${id#sha256:}"
}

apptainer_fingerprint() {
  [[ -f "${APPTAINER_SIF}" ]] || return 1
  sha256sum "${APPTAINER_SIF}" | awk '{print $1}'
}

external_ort_fingerprint() {
  local root="${OVG_ORT_ROOT:-}"
  [[ -n "${root}" ]] || return 1
  root="${root%/}"
  local prefix="${ORT_CONTAINER_ROOT}/install/"
  [[ "${root}" == "${prefix}"* ]] || die \
    "OVG_ORT_ROOT must point below ${prefix}: ${root}"
  local fingerprint="${root#"${prefix}"}"
  [[ -n "${fingerprint}" && "${fingerprint}" != */* ]] || die \
    "OVG_ORT_ROOT does not contain an ORT fingerprint: ${root}"
  [[ "${fingerprint}" =~ ^[A-Za-z0-9._+,=-]+$ ]] || die \
    "OVG_ORT_ROOT contains an unsafe ORT fingerprint: ${fingerprint}"
  printf '%s\n' "${fingerprint}"
}

set_fingerprint() {
  local image_fingerprint
  if [[ "${RUNTIME}" == docker ]]; then
    image_fingerprint="$(docker_fingerprint)" || die "Docker image ${IMAGE_NAME} is not available"
  else
    image_fingerprint="$(apptainer_fingerprint)" || die "Apptainer SIF is not available: ${APPTAINER_SIF}"
  fi
  image_fingerprint="${image_fingerprint:0:32}"
  local workspace_fingerprint="${image_fingerprint}"
  if [[ -n "${OVG_ORT_ROOT:-}" ]]; then
    workspace_fingerprint="$(external_ort_fingerprint)"
  fi
  compose_env "${image_fingerprint}" "${workspace_fingerprint}"
}

compose() {
  docker compose -f "${COMPOSE_FILE}" "$@"
}

build_docker() {
  resolve_gpu_targets
  compose build amd
  set_fingerprint
}

ensure_sif() {
  command -v apptainer >/dev/null 2>&1 || die "apptainer is unavailable"
  if [[ -f "${APPTAINER_SIF}" ]]; then
    return
  fi
  [[ -n "${OVG_APPTAINER_IMAGE_URI:-}" ]] || die \
    "SIF is missing: ${APPTAINER_SIF}; copy a prebuilt SIF there or set OVG_APPTAINER_IMAGE_URI"
  local partial="${APPTAINER_SIF}.partial.$$"
  mkdir -p "$(dirname "${APPTAINER_SIF}")"
  echo "Pulling ${OVG_APPTAINER_IMAGE_URI} to ${APPTAINER_SIF}"
  apptainer pull --name "${partial}" "${OVG_APPTAINER_IMAGE_URI}"
  [[ -f "${partial}" ]] || [[ -f "${partial}.sif" ]] || \
    die "apptainer pull did not produce a SIF at ${partial}"
  [[ -f "${partial}" ]] || partial="${partial}.sif"
  mv -- "${partial}" "${APPTAINER_SIF}"
}

apptainer_args() {
  APPTAINER_ARGS=(
    --rocm
    --cleanenv
    --pwd "${OVG_WORKSPACE_ROOT}"
    --bind "${ROOT_DIR}:${OVG_WORKSPACE_ROOT}"
    --bind "${MANAGED_DIR}:${OVG_WORKSPACE_ROOT}/src/gpu_ros_managed"
    --bind "${STATE_ROOT}/assets:${OVG_ASSETS_ROOT}"
    --bind "${STATE_ROOT}/cache:${OVG_CACHE_ROOT}"
    --bind "${STATE_ROOT}/results:${OVG_RESULTS_ROOT}"
    --bind "${OVG_BUILD_DIR}:${OVG_WORKSPACE_ROOT}/build"
    --bind "${OVG_INSTALL_DIR}:${OVG_WORKSPACE_ROOT}/install"
    --bind "${OVG_LOG_DIR}:${OVG_WORKSPACE_ROOT}/log"
    --bind "${ORT_STATE_HOST}:${ORT_CONTAINER_ROOT}"
    --bind "${STATE_ROOT}/home:/home/ovg"
    --env "HOME=/home/ovg"
    --env "OVG_WORKSPACE_ROOT=${OVG_WORKSPACE_ROOT}"
    --env "OVG_ASSETS_ROOT=${OVG_ASSETS_ROOT}"
    --env "OVG_CACHE_ROOT=${OVG_CACHE_ROOT}"
    --env "OVG_RESULTS_ROOT=${OVG_RESULTS_ROOT}"
    --env "OVG_ORT_STATE_ROOT=${ORT_CONTAINER_ROOT}"
    --env "OVG_ORT_ROOT=${OVG_ORT_ROOT:-}"
    --env "OVG_IMAGE_FINGERPRINT=${OVG_IMAGE_FINGERPRINT}"
    --env "OVG_WORKSPACE_FINGERPRINT=${OVG_WORKSPACE_FINGERPRINT}"
    --env "OVG_RUNTIME_EFFECTIVE=apptainer"
    --env "ROS2_BENCHMARK_OVERRIDE_ASSETS_ROOT=${OVG_ASSETS_ROOT}"
    --env "NGC_CLI_API_KEY=${NGC_CLI_API_KEY:-}"
    --env "NGC_CLI_ORG=${NGC_CLI_ORG:-}"
  )
  [[ -e /dev/kfd ]] && APPTAINER_ARGS+=(--bind /dev/kfd)
  [[ -d /dev/dri ]] && APPTAINER_ARGS+=(--bind /dev/dri)
}

apptainer_exec() {
  apptainer_args
  if [[ "${OVG_APPTAINER_INSTANCE:-0}" == 1 ]] && \
      apptainer instance list 2>/dev/null | awk '{print $1}' | grep -Fxq "${APPTAINER_INSTANCE_NAME}"; then
    apptainer exec "instance://${APPTAINER_INSTANCE_NAME}" \
      /usr/local/bin/phase2-amd-entrypoint.sh "$@"
  else
    apptainer exec "${APPTAINER_ARGS[@]}" "${APPTAINER_SIF}" \
      /usr/local/bin/phase2-amd-entrypoint.sh "$@"
  fi
}

docker_exec() {
  compose exec -T amd /usr/local/bin/phase2-amd-entrypoint.sh "$@"
}

run_phase2() {
  if [[ "${RUNTIME}" == docker ]]; then
    docker_exec "${OVG_WORKSPACE_ROOT}/tools/phase2" "$@"
  else
    apptainer_exec phase2 "$@"
  fi
}

start_runtime() {
  if [[ "${RUNTIME}" == docker ]]; then
    compose up -d amd
  elif [[ "${OVG_APPTAINER_INSTANCE:-0}" == 1 ]]; then
    apptainer_args
    if ! apptainer instance list 2>/dev/null | awk '{print $1}' | grep -Fxq "${APPTAINER_INSTANCE_NAME}"; then
      apptainer instance start "${APPTAINER_ARGS[@]}" "${APPTAINER_SIF}" "${APPTAINER_INSTANCE_NAME}"
    fi
  else
    echo "Apptainer one-shot mode: no persistent container was started"
  fi
}

verify_runtime() {
  run_phase2 env --verify
}

main() {
  local command="${1:-bootstrap}"
  case "${command}" in
    bootstrap|build|up|colcon|verify|shell|stop|down) ;;
    *) usage; exit 2 ;;
  esac

  prepare_state
  select_runtime
  repo_check

  case "${command}" in
    bootstrap)
      if [[ "${RUNTIME}" == docker ]]; then
        build_docker
      else
        ensure_sif
        set_fingerprint
      fi
      start_runtime
      run_phase2 env
      if [[ "${OVG_PREPARE_ASSETS:-0}" == 1 ]]; then
        run_phase2 assets
      fi
      ;;
    build)
      if [[ "${RUNTIME}" == docker ]]; then
        build_docker
      else
        ensure_sif
        set_fingerprint
      fi
      ;;
    up)
      set_fingerprint
      start_runtime
      ;;
    colcon)
      set_fingerprint
      [[ "${RUNTIME}" == docker ]] && start_runtime
      run_phase2 build
      ;;
    verify)
      set_fingerprint
      [[ "${RUNTIME}" == docker ]] && start_runtime
      verify_runtime
      ;;
    shell)
      set_fingerprint
      [[ "${RUNTIME}" == docker ]] && start_runtime
      if [[ "${RUNTIME}" == docker ]]; then
        compose exec amd /usr/local/bin/phase2-amd-entrypoint.sh bash -l
      else
        apptainer_exec bash -l
      fi
      ;;
    stop)
      if [[ "${RUNTIME}" == docker ]]; then
        compose stop amd
      elif [[ "${OVG_APPTAINER_INSTANCE:-0}" == 1 ]]; then
        apptainer instance stop "${APPTAINER_INSTANCE_NAME}"
      else
        echo "Apptainer one-shot mode: there is no instance to stop"
      fi
      ;;
    down)
      if [[ "${RUNTIME}" == docker ]]; then
        compose down
      elif [[ "${OVG_APPTAINER_INSTANCE:-0}" == 1 ]]; then
        apptainer instance stop "${APPTAINER_INSTANCE_NAME}"
      else
        echo "Apptainer one-shot mode: no persistent container was removed"
      fi
      ;;
  esac
}

main "$@"
