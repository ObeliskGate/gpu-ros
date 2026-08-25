#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${ROOT_DIR}/external/nvidia-isaac-ros.repos"

if [[ $# -ne 2 || "${1:-}" != --source-root ]]; then
  echo "Usage: $0 --source-root /absolute/path/outside/repository" >&2
  exit 2
fi
SOURCE_ROOT="$(realpath -m -- "$2")"
case "${SOURCE_ROOT}/" in
  "${ROOT_DIR}/"*)
    echo "ERROR: NVIDIA external source root must be outside ${ROOT_DIR}" >&2
    exit 1
    ;;
esac

command -v vcs >/dev/null 2>&1 || {
  echo "ERROR: vcs is required (install vcstool outside this script)" >&2
  exit 1
}
mkdir -p "${SOURCE_ROOT}"
vcs import "${SOURCE_ROOT}" < "${MANIFEST}"

verify_checkout() {
  local name="$1" expected="$2" path="${SOURCE_ROOT}/$1"
  [[ -d "${path}/.git" ]] || { echo "ERROR: missing checkout ${path}" >&2; exit 1; }
  local actual
  actual="$(git -C "${path}" rev-parse HEAD)"
  [[ "${actual}" == "${expected}" ]] || {
    echo "ERROR: ${name} HEAD=${actual}, expected ${expected}" >&2
    exit 1
  }
}

verify_checkout isaac_ros_object_detection 060ced887bd8a3a0be60b1fa454365942eefd128
verify_checkout isaac_ros_benchmark f46699e124262c5bfb6f00099061f6718f026b3f
echo "NVIDIA external sources are ready under ${SOURCE_ROOT}"
