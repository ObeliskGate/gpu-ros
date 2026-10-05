#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="${ROOT_DIR}/gpu_ros_object_detection/external/nvidia-isaac-ros.repos"

if [[ $# -ne 2 || "${1:-}" != --source-root ]]; then
  echo "Usage: ISAAC_ROS_BASE_IMAGE=nvcr.io/nvidia/isaac/ros:<tag>@sha256:<digest> $0 --source-root /absolute/path/outside/repository" >&2
  exit 2
fi
SOURCE_ROOT="$(realpath -m -- "$2")"
case "${SOURCE_ROOT}/" in
  "${ROOT_DIR}/"*)
    echo "ERROR: NVIDIA external source root must be outside ${ROOT_DIR}" >&2
    exit 1
    ;;
esac

[[ "${ISAAC_ROS_BASE_IMAGE:-}" =~ ^nvcr\.io/nvidia/isaac/ros(:[^@[:space:]]+)?@sha256:[0-9a-f]{64}$ ]] || {
  echo "BLOCKED: set ISAAC_ROS_BASE_IMAGE to the official Isaac ROS 5.0 image reference including its verified sha256 digest." >&2
  exit 1
}
command -v vcs >/dev/null 2>&1 || {
  echo "ERROR: vcs is required (install vcstool outside this script)" >&2
  exit 1
}
command -v python3 >/dev/null 2>&1 || {
  echo "ERROR: python3 is required to read the standard vcstool manifest." >&2
  exit 1
}
[[ -r "${MANIFEST}" ]] || {
  echo "BLOCKED: external source manifest is missing: ${MANIFEST}" >&2
  exit 1
}
manifest_rows="$(
  python3 - "${MANIFEST}" <<'PY'
import re
import sys
from urllib.parse import urlsplit

try:
    import yaml
except ImportError as error:
    raise SystemExit("BLOCKED: PyYAML is required to validate the vcstool manifest.") from error

with open(sys.argv[1], encoding="utf-8") as manifest_file:
    document = yaml.safe_load(manifest_file)
repositories = document.get("repositories") if isinstance(document, dict) else None
if not isinstance(repositories, dict) or not repositories:
    raise SystemExit(
        "BLOCKED: no image-paired NVIDIA source revisions are recorded in the vcstool manifest; "
        "do not import the former Isaac ROS 4.5 pins."
    )

for name, entry in sorted(repositories.items()):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or not isinstance(entry, dict):
        raise SystemExit(f"ERROR: invalid vcstool repository entry: {name!r}")
    if entry.get("type") != "git":
        raise SystemExit(f"ERROR: {name} must use a git vcstool repository.")
    url = entry.get("url")
    parsed = urlsplit(url or "")
    if parsed.scheme != "https" or parsed.netloc != "github.com" or not parsed.path.startswith("/NVIDIA-ISAAC-ROS/"):
        raise SystemExit(f"ERROR: {name} is not sourced from NVIDIA-ISAAC-ROS on GitHub.")
    revision = entry.get("version")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit(f"BLOCKED: {name} requires an exact image-paired 40-character commit.")
    print("|".join((name, url, revision)))
PY
)" || exit $?
mapfile -t REPOSITORIES <<<"${manifest_rows}"

verify_checkout() {
  local name="$1" expected="$2" path="${SOURCE_ROOT}/$1"
  [[ -d "${path}/.git" ]] || {
    echo "ERROR: missing checkout ${path}" >&2
    exit 1
  }
  local actual
  actual="$(git -C "${path}" rev-parse HEAD)"
  [[ "${actual}" == "${expected}" ]] || {
    echo "ERROR: ${name} HEAD=${actual}, expected image-paired ${expected}; refusing to change the existing checkout." >&2
    exit 1
  }
  [[ -z "$(git -C "${path}" status --porcelain)" ]] || {
    echo "ERROR: external checkout is dirty: ${path}" >&2
    exit 1
  }
}

for repository in "${REPOSITORIES[@]}"; do
  IFS='|' read -r name url expected <<<"${repository}"
  if [[ -e "${SOURCE_ROOT}/${name}" ]]; then
    verify_checkout "${name}" "${expected}"
  fi
done

mkdir -p "${SOURCE_ROOT}"
for repository in "${REPOSITORIES[@]}"; do
  IFS='|' read -r name url expected <<<"${repository}"
  [[ -e "${SOURCE_ROOT}/${name}" ]] && continue
  vcs import "${SOURCE_ROOT}" <<EOF
repositories:
  ${name}:
    type: git
    url: ${url}
    version: ${expected}
EOF
done

for repository in "${REPOSITORIES[@]}"; do
  IFS='|' read -r name url expected <<<"${repository}"
  verify_checkout "${name}" "${expected}"
done
echo "External source checkouts match the exact revisions supplied for ${ISAAC_ROS_BASE_IMAGE}."
