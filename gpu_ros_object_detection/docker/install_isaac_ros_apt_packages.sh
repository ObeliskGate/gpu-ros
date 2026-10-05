#!/usr/bin/env bash
set -euo pipefail

readonly ROS_APT_SOURCE='https://isaac.download.nvidia.com/isaac-ros/ubuntu/main'
readonly ROS_APT_KEY='https://isaac.download.nvidia.com/isaac-ros/repos.key'
readonly ROS_APT_KEYRING='/usr/share/keyrings/isaac-ros-archive-keyring.gpg'
readonly ROS_APT_SOURCE_FILE='/etc/apt/sources.list.d/isaac-ros.list'
readonly ROS_TOOLS_APT_SOURCE_VERSION='1.2.0'
readonly apt_tool_packages=(python3-colcon-common-extensions)

fail() {
  printf 'BLOCKED: %s\n' "$*" >&2
  exit 1
}

[[ "${ROS_DISTRO:-}" == lyrical ]] || fail \
  'ROS_DISTRO must be lyrical; this installer does not select another ROS release.'

image_ref="${ISAAC_ROS_BASE_IMAGE:-}"
[[ "${image_ref}" =~ ^nvcr\.io/nvidia/isaac/ros(:[^@[:space:]]+)?@sha256:[0-9a-f]{64}$ ]] || fail \
  'set ISAAC_ROS_BASE_IMAGE to the verified official Isaac ROS 5.0 image reference including its sha256 digest.'

[[ -n "${ROS_LYRICAL_APT_PACKAGE_SPECS:-}" ]] || fail \
  'ROS_LYRICAL_APT_PACKAGE_SPECS is missing; supply the exact ros-lyrical package=version set recorded from that image.'
[[ "${ROS_LYRICAL_APT_PACKAGE_SPECS}" != *$'\n'* &&
  "${ROS_LYRICAL_APT_PACKAGE_SPECS}" != *$'\r'* ]] || fail \
  'ROS_LYRICAL_APT_PACKAGE_SPECS must be a single line; refusing a truncated package lock.'

read -r -a package_specs <<<"${ROS_LYRICAL_APT_PACKAGE_SPECS}"
((${#package_specs[@]} > 0)) || fail \
  'ROS_LYRICAL_APT_PACKAGE_SPECS contains no package pins.'

declare -A locked_versions=()
for spec in "${package_specs[@]}"; do
  [[ "${spec}" =~ ^(ros-lyrical-[a-z0-9.+-]+)=([A-Za-z0-9.+:~_-]+)$ ]] || fail \
    "invalid exact ROS package spec '${spec}'; expected ros-lyrical-PACKAGE=VERSION."
  package="${BASH_REMATCH[1]}"
  version="${BASH_REMATCH[2]}"
  case "${package}" in
    ros-lyrical-cuda* | ros-lyrical-isaac* | ros-lyrical-nvidia* | \
      ros-lyrical-*tensor-list* | ros-lyrical-*tensor_list* | ros-lyrical-*nitros*)
      fail "AMD and CPU profiles cannot install NVIDIA/CUDA transport package ${package}."
      ;;
  esac
  [[ -z "${locked_versions[${package}]+present}" ]] || fail \
    "duplicate ROS package spec for ${package}."
  locked_versions["${package}"]="${version}"
done

if [[ "${1:-}" == --check-inputs ]]; then
  [[ $# -eq 1 ]] || fail 'usage: install_isaac_ros_apt_packages.sh [--check-inputs]'
  printf 'Isaac ROS image authority: %s\n' "${image_ref}"
  printf 'Exact ROS package pins supplied: %s\n' "${#package_specs[@]}"
  exit 0
fi
[[ $# -eq 0 ]] || fail 'usage: install_isaac_ros_apt_packages.sh [--check-inputs]'
[[ "$(id -u)" -eq 0 ]] || fail 'run this installer as root.'

[[ -r /etc/os-release ]] || fail 'cannot identify the operating system.'
# shellcheck source=/etc/os-release
source /etc/os-release
[[ "${VERSION_CODENAME:-}" == noble ]] || fail \
  "expected Ubuntu noble for the Lyrical package baseline, found ${VERSION_CODENAME:-unknown}."

command -v curl >/dev/null 2>&1 || fail 'curl is required.'
command -v gpg >/dev/null 2>&1 || fail 'gpg is required.'
command -v python3 >/dev/null 2>&1 || fail 'python3 is required.'
command -v apt-get >/dev/null 2>&1 || fail 'apt-get is required.'
existing_ros="$(dpkg-query -W -f='${binary:Package}=${Version}\n' 'ros-*' 2>/dev/null || true)"
[[ -z "${existing_ros}" ]] || fail "refusing to install the target ROS baseline over preinstalled ROS packages: ${existing_ros}"

# NVIDIA's Noble Lyrical repository contains ROS packages, not colcon.
# Follow the Isaac ROS buildfarm setup for distribution-independent tools.
ros_tools_source_deb="$(mktemp)"
trap 'rm -f -- "${ros_tools_source_deb}"' EXIT
curl --fail --silent --show-error --location \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_TOOLS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_TOOLS_APT_SOURCE_VERSION}.noble_all.deb" \
  --output "${ros_tools_source_deb}"
dpkg --install "${ros_tools_source_deb}"

install -d -m 0755 /usr/share/keyrings
curl --fail --silent --show-error --location "${ROS_APT_KEY}" |
  gpg --dearmor --yes --output "${ROS_APT_KEYRING}"
chmod 0644 "${ROS_APT_KEYRING}"
printf 'deb [arch=%s signed-by=%s] %s noble main\n' \
  "$(dpkg --print-architecture)" "${ROS_APT_KEYRING}" "${ROS_APT_SOURCE}" \
  >"${ROS_APT_SOURCE_FILE}"
cat >/etc/apt/preferences.d/gpu-ros-isaac-ros-lyrical.pref <<'EOF'
Package: ros-lyrical-*
Pin: origin "isaac.download.nvidia.com"
Pin-Priority: 1001
EOF
apt-get update
for spec in "${package_specs[@]}"; do
  package="${spec%%=*}"
  version="${spec#*=}"
  madison="$(apt-cache madison "${package}")" || fail "could not inspect the NVIDIA package candidate for ${package}."
  printf '%s' "${madison}" | python3 -c '
import sys

package, version = sys.argv[1:3]
for line in sys.stdin:
    fields = line.split("|", 2)
    if (
        len(fields) == 3
        and fields[0].strip() == package
        and fields[1].strip() == version
        and "isaac.download.nvidia.com/isaac-ros/ubuntu/main" in fields[2]
    ):
        raise SystemExit(0)
raise SystemExit(
    f"BLOCKED: {package}={version} is not present in the official NVIDIA Lyrical package source."
)
' "${package}" "${version}" || fail "the exact ${package}=${version} pin is absent from the official NVIDIA Lyrical package source."
done

simulation="$(apt-get --simulate install --no-install-recommends "${package_specs[@]}" "${apt_tool_packages[@]}")" || fail \
  'apt could not resolve the exact ROS package lock and fixed build tools.'
printf '%s' "${simulation}" | python3 -c '
import sys

locked = {}
for spec in sys.argv[1:]:
    name, version = spec.split("=", 1)
    locked[name] = version
for line in sys.stdin:
    fields = line.split()
    if len(fields) < 2 or fields[0] not in {"Inst", "Remv"}:
        continue
    name = fields[1].split(":", 1)[0]
    if fields[0] == "Remv" and name.startswith("ros-"):
        raise SystemExit(f"BLOCKED: apt would remove a ROS package: {line.strip()}")
    if name.startswith(("cuda", "libcuda", "libcud", "nvidia-", "libnvidia", "libnv", "isaac", "libisaac")):
        raise SystemExit(f"BLOCKED: AMD ROS dependency closure would add NVIDIA/CUDA runtime package: {line.strip()}")
    if not name.startswith("ros-"):
        continue
    if not name.startswith("ros-lyrical-"):
        raise SystemExit(f"BLOCKED: ROS dependency closure includes a non-Lyrical package: {line.strip()}")
    try:
        target_version = line[line.index("(") + 1 :].split()[0]
    except (ValueError, IndexError):
        raise SystemExit(f"BLOCKED: could not parse ROS package install plan: {line.strip()}")
    if locked.get(name) != target_version:
        raise SystemExit(
            f"BLOCKED: apt would install unlocked ROS package {name}={target_version}; "
            "the lock must include every ROS package in the transaction at its image-matched version."
        )
' "${package_specs[@]}"

apt-get install -y --no-install-recommends "${package_specs[@]}" "${apt_tool_packages[@]}"

installed_ros="$(dpkg-query -W -f='${binary:Package}=${Version}\n' 'ros-*' 2>/dev/null)" || fail \
  'could not enumerate installed ROS packages after installation.'
printf '%s\n' "${installed_ros}" | python3 -c '
import sys

locked = {}
for spec in sys.argv[1:]:
    name, version = spec.split("=", 1)
    locked[name] = version
installed = {}
for line in sys.stdin:
    item = line.strip()
    if not item:
        continue
    name, version = item.split("=", 1)
    installed[name.split(":", 1)[0]] = version
if installed != locked:
    missing = sorted(set(locked) - set(installed))
    unexpected = sorted(set(installed) - set(locked))
    mismatched = sorted(
        f"{name}: expected {locked[name]}, found {installed[name]}"
        for name in set(locked) & set(installed)
        if locked[name] != installed[name]
    )
    raise SystemExit(
        "BLOCKED: installed ros-lyrical package inventory differs from the image lock; "
        f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
    )
' "${package_specs[@]}"

lock_sha256="$(printf '%s\n' "${package_specs[@]}" | LC_ALL=C sort | sha256sum | cut -d' ' -f1)"
install -d -m 0755 /var/lib/gpu-ros
printf '{"isaac_ros_base_image":"%s","ros_distro":"lyrical","ros_apt_source":"%s","package_specs_sha256":"%s"}\n' \
  "${image_ref}" "${ROS_APT_SOURCE}" "${lock_sha256}" \
  >/var/lib/gpu-ros/isaac-ros-5-ros-package-lock.json
printf 'Installed the exact image-matched Lyrical ROS package lock (%s packages, sha256 %s).\n' \
  "${#package_specs[@]}" "${lock_sha256}"
