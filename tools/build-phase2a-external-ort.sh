#!/usr/bin/env bash
# Build a patched ONNX Runtime 1.23.1 MIGraphX install outside the container
# image.  The source checkout must be a pristine recursive v1.23.1 checkout.
set -euo pipefail
trap 'echo "ERROR: external ORT build failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORT_VERSION="${ORT_VERSION:-1.23.1}"
ORT_STATE_ROOT="${OVG_ORT_STATE_ROOT:-/workspaces/ovg-ort}"
SOURCE_ROOT="${ORT_STATE_ROOT}/source/onnxruntime"
BUILD_PARENT="${ORT_STATE_ROOT}/build"
INSTALL_PARENT="${ORT_STATE_ROOT}/install"
PATCH_DIR="${ROOT_DIR}/docker/patches"
PATCH_SERIES="${ORT_PATCH_SERIES:-${PATCH_DIR}/onnxruntime-${ORT_VERSION}.series}"
ORT_BUILD_JOBS="${ORT_BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-24}}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ "${ORT_VERSION}" == 1.23.1 ]] || die "this external builder is pinned to ORT 1.23.1"
[[ -d "${SOURCE_ROOT}/.git" ]] || die \
  "pristine recursive ORT source is missing: ${SOURCE_ROOT}"
[[ -f "${PATCH_SERIES}" ]] || die "patch series is missing: ${PATCH_SERIES}"
command -v git >/dev/null 2>&1 || die "git is required"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"

git_source() {
  git -C "${SOURCE_ROOT}" "$@"
}

source_commit="$(git_source rev-parse --verify HEAD)"
tag_commit="$(git_source rev-parse --verify "refs/tags/v${ORT_VERSION}^{commit}" 2>/dev/null || true)"
[[ -n "${tag_commit}" && "${tag_commit}" == "${source_commit}" ]] || die \
  "source must be exactly tag v${ORT_VERSION} (HEAD=${source_commit})"

if [[ -n "$(git_source status --porcelain)" ]]; then
  die "ORT source is not clean: ${SOURCE_ROOT}"
fi
if ! git_source diff --quiet; then
  die "ORT source has unstaged changes: ${SOURCE_ROOT}"
fi
if ! git_source diff --cached --quiet; then
  die "ORT source has staged changes: ${SOURCE_ROOT}"
fi

# A recursive clone reports an initialized, expected submodule with a leading
# space.  '-', '+', and 'U' mean missing, mismatched, or conflicted state.
submodule_status_output="$(git_source submodule status --recursive)" || die \
  "could not inspect ORT submodules"
while IFS= read -r submodule_status; do
  [[ -n "${submodule_status}" ]] || continue
  case "${submodule_status:0:1}" in
    -|+|U) die "ORT submodule is not pristine: ${submodule_status}" ;;
  esac
done <<<"${submodule_status_output}"
if ! git_source diff --submodule=short --exit-code; then
  die "ORT submodule content differs from the recorded commit"
fi
if ! git_source submodule foreach --recursive \
  'test -z "$(git status --porcelain)"'; then
  die "ORT submodule has local changes or untracked files"
fi

normalise_targets() {
  local raw="${1//;/,}"
  raw="${raw// /}"
  tr ',' '\n' <<<"${raw}" \
    | awk 'NF && $0 ~ /^gfx[0-9a-fA-F]+$/ {print tolower($0)}' \
    | sort -u \
    | paste -sd, -
}

[[ -n "${AMD_GPU_TARGETS:-}" ]] || die \
  "AMD_GPU_TARGETS must contain the actual gfx target for this machine"
AMD_GPU_TARGETS="$(normalise_targets "${AMD_GPU_TARGETS}")"
[[ -n "${AMD_GPU_TARGETS}" ]] || die "AMD_GPU_TARGETS contains no valid gfx target"
[[ "${ORT_BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]] || die \
  "ORT_BUILD_JOBS must be a positive integer"

mapfile -t PATCHES < <(
  sed -e 's/[[:space:]]*#.*$//' -e '/^[[:space:]]*$/d' "${PATCH_SERIES}"
)
((${#PATCHES[@]} > 0)) || die "patch series is empty: ${PATCH_SERIES}"
for patch_name in "${PATCHES[@]}"; do
  [[ "${patch_name}" != */* ]] || die "patch series entry must be a basename: ${patch_name}"
  [[ -f "${PATCH_DIR}/${patch_name}" ]] || die \
    "patch listed by series is missing: ${PATCH_DIR}/${patch_name}"
done

patchset_sha256="$({
  printf 'series-sha256 %s\n' "$(sha256sum "${PATCH_SERIES}" | awk '{print $1}')"
  for patch_name in "${PATCHES[@]}"; do
    printf 'patch %s %s\n' "${patch_name}" \
      "$(sha256sum "${PATCH_DIR}/${patch_name}" | awk '{print $1}')"
  done
} | sha256sum | awk '{print $1}')"
patchset_sha12="${patchset_sha256:0:12}"
commit12="${source_commit:0:12}"
fingerprint="${commit12}-${patchset_sha12}-${AMD_GPU_TARGETS}"
BUILD_ROOT="${BUILD_PARENT}/${fingerprint}"
BUILD_SOURCE="${BUILD_ROOT}/source"
INSTALL_ROOT="${INSTALL_PARENT}/${fingerprint}"

[[ ! -e "${INSTALL_ROOT}" ]] || die \
  "refusing to overwrite existing external ORT install: ${INSTALL_ROOT}"
if [[ -e "${BUILD_ROOT}" ]]; then
  die "refusing to reuse existing external ORT build directory: ${BUILD_ROOT}
If the previous build failed, remove only that fingerprint directory and rerun:
  rm -rf -- ${BUILD_ROOT}
Keep ${SOURCE_ROOT} and any completed install directory; they are not part of this cleanup."
fi
mkdir -p "${BUILD_PARENT}" "${BUILD_ROOT}" "${INSTALL_PARENT}"

echo "ORT source commit: ${source_commit}"
echo "ORT patch set: ${patchset_sha12}"
echo "GPU targets: ${AMD_GPU_TARGETS}"
echo "External ORT fingerprint: ${fingerprint}"
echo "Copying pristine source to ${BUILD_SOURCE}"
cp -a --reflink=auto "${SOURCE_ROOT}" "${BUILD_SOURCE}"

for patch_name in "${PATCHES[@]}"; do
  echo "Applying ${patch_name}"
  git -C "${BUILD_SOURCE}" apply --check "${PATCH_DIR}/${patch_name}"
  git -C "${BUILD_SOURCE}" apply "${PATCH_DIR}/${patch_name}"
done

CMAKE_TARGETS="$(tr ',' ';' <<<"${AMD_GPU_TARGETS}")"
printf -v CMAKE_TARGETS_QUOTED '%q' "${CMAKE_TARGETS}"
printf -v ORT_BUILD_JOBS_QUOTED '%q' "${ORT_BUILD_JOBS}"
EXACT_COMMAND="./build.sh --config Release --parallel ${ORT_BUILD_JOBS_QUOTED} --build_shared_lib --skip_tests --allow_running_as_root --use_migraphx --migraphx_home /opt/rocm --cmake_extra_defines GPU_TARGETS=${CMAKE_TARGETS_QUOTED} CMAKE_HIP_ARCHITECTURES=${CMAKE_TARGETS_QUOTED}"
echo "+ ${EXACT_COMMAND}"
(
  cd "${BUILD_SOURCE}"
  ./build.sh \
    --config Release \
    --parallel "${ORT_BUILD_JOBS}" \
    --build_shared_lib \
    --skip_tests \
    --allow_running_as_root \
    --use_migraphx \
    --migraphx_home /opt/rocm \
    --cmake_extra_defines \
      "GPU_TARGETS=${CMAKE_TARGETS}" \
      "CMAKE_HIP_ARCHITECTURES=${CMAKE_TARGETS}"
)

mkdir -p "${INSTALL_ROOT}/include" "${INSTALL_ROOT}/lib"
cp -a "${BUILD_SOURCE}/include/onnxruntime/core/session/"*.h "${INSTALL_ROOT}/include/"
find "${BUILD_SOURCE}/build/Linux/Release" -maxdepth 1 \
  \( -type f -o -type l \) -name 'libonnxruntime*.so*' \
  -exec cp -a {} "${INSTALL_ROOT}/lib/" \;

for required_file in \
  "${INSTALL_ROOT}/include/onnxruntime_cxx_api.h" \
  "${INSTALL_ROOT}/lib/libonnxruntime.so" \
  "${INSTALL_ROOT}/lib/libonnxruntime_providers_shared.so" \
  "${INSTALL_ROOT}/lib/libonnxruntime_providers_migraphx.so"; do
  [[ -e "${required_file}" ]] || die "external ORT build did not produce ${required_file}"
done

printf '%s\n' "${fingerprint}" > "${INSTALL_ROOT}/.ovg-ort-fingerprint"
{
  printf 'source_commit=%s\n' "${source_commit}"
  printf 'patchset_sha256=%s\n' "${patchset_sha256}"
  for patch_name in "${PATCHES[@]}"; do
    printf 'patch.%s.sha256=' "${patch_name}"
    sha256sum "${PATCH_DIR}/${patch_name}" | awk '{print $1}'
  done
  printf 'gpu_targets=%s\n' "${AMD_GPU_TARGETS}"
  printf 'command=%s\n' "${EXACT_COMMAND}"
} > "${INSTALL_ROOT}/build-info.txt"

echo "PASS: external ORT installed at ${INSTALL_ROOT}"
