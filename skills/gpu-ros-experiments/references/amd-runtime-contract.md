# AMD runtime contract

This contract defines the software, mounts, devices, and environment for AMD
experiments. Docker is the default (`OVG_RUNTIME=docker`). An authorized Slurm
compute allocation without Docker may use the explicit Apptainer branch.
Site deployment and allocation details stay outside public methods.

The monorepo contains both library collections. It is mounted once at
`/workspaces/gpu-ros`; the detection directory is not a new workspace root.

## Required software identity

The following identities are the baseline for formal Phase 2A and Phase 2B
runs:

| Layer | Required identity | Public source of truth |
| --- | --- | --- |
| Base OS | Ubuntu 24.04 userland | `AMD_BASE_IMAGE` below |
| ROS | ROS 2 Jazzy | `ROS_DISTRO=jazzy` and the Dockerfile |
| ROCm/HIP | ROCm 7.1.1 | `rocm/dev-ubuntu-24.04:7.1.1-complete` and `phase2 env --verify` |
| GPU target | The actual target reported by the device; the campaign baseline is `gfx950` | `AMD_GPU_TARGETS` and `rocminfo` |
| GPU driver | The compatible AMD kernel/user-space driver release for the device | `rocminfo`, `rocminfo --version`, and the site package query; record it in the result archive |
| MIGraphX | MIGraphX 2.14 behavior shipped by the ROCm 7.1.1 package set | `migraphx`, `migraphx-dev`, and the provider audit |
| ONNX Runtime | 1.23.1 at the commit in `gpu_ros_object_detection/config/onnxruntime.lock` | `gpu_ros_object_detection/config/onnxruntime.lock` |
| ORT provider patches | The tracked GridSample and int64-div CPU-fallback patches | `gpu_ros_object_detection/docker/patches/onnxruntime-1.23.1-migraphx-*.patch` and `.series` |
| Benchmark framework | `ros2_benchmark` `v4.5-0`, plus the tracked standalone patch | `gpu_ros_object_detection/docker/phase2a-amd.Dockerfile` |

MIGraphX is installed from the ROCm package repository rather than built from a
separately pinned source checkout in this repository. Capture its package
revision in every raw run archive:

```bash
dpkg-query -W -f='${Package}=${Version}\n' migraphx migraphx-dev
migraphx-driver --version 2>/dev/null || true
```

A different package revision is a different software baseline even when the
ROCm marketing version is unchanged.

## Container recipe and image identity

The OCI recipes are
`gpu_ros_object_detection/docker/phase2a-amd.Dockerfile` and
`gpu_ros_object_detection/docker/docker-compose.phase2a-amd.yaml`.
Reuse a verified existing dependency image or SIF first. Build only when one
is unavailable and a dependency build is authorized:

```bash
export OVG_RUNTIME=docker
export AMD_BASE_IMAGE=rocm/dev-ubuntu-24.04:7.1.1-complete
export AMD_GPU_TARGETS=gfx950
docker compose \
  -f gpu_ros_object_detection/docker/docker-compose.phase2a-amd.yaml build amd
```

The image builds ROS 2 Jazzy, MIGraphX, the benchmark framework, and a
build-only ORT copy. Formal runs must use the separately built external ORT
install selected by `OVG_ORT_ROOT`; `/opt/onnxruntime` in the image is not a
formal fallback. If a new external install is required, use
`gpu_ros_object_detection/tools/build-phase2a-external-ort.sh` with the locked
recursive source, and retain its `build-info.txt` and fingerprint marker.

The Apptainer recipe is `gpu_ros_object_detection/apptainer/phase2-amd.def`.
Record the full SIF SHA-256 and adapter identity. Docker and Apptainer share
the software contract, but running one does not verify the other. The SIF
filename is a site choice. Record an immutable digest when available; otherwise
retain the recipe revision and base identity rather than saying only that a
runtime was approved.

## Container paths and mounts

The launcher exposes these stable paths inside either runtime:

| Container path | Required content | Host-side choice |
| --- | --- | --- |
| `/workspaces/gpu-ros` | The single GPU ROS repository checkout, including `gpu_ros_managed/` and application packages | Any source checkout |
| `/workspaces/ovg-assets` | Imported model and R2B assets | External asset directory |
| `/workspaces/ovg-cache` | Persistent MIGraphX compilation cache | External cache directory |
| `/workspaces/ovg-results` | Capture, benchmark, audit, and comparison output | External result directory |
| `/workspaces/ovg-ort` | External ORT source/build/install state | External ORT state directory |
| `/workspaces/gpu-ros/build` | Persistent colcon build tree | Optional but recommended state mount |
| `/workspaces/gpu-ros/install` | Persistent colcon install tree | Optional but recommended state mount |
| `/workspaces/gpu-ros/log` | Colcon logs | Optional state mount |
| `/home/ovg` | Runtime home state | Separate host state directory |

The repository, assets, cache, results, and ORT mounts are required for a formal
run. Build/install/log may be ephemeral only when the complete build is
reproduced in the same invocation. Keep the home mount separate from the source
mount. Do not mount a hidden fallback ORT, a sibling transport checkout, or a
second asset root.

## GPU and runtime capabilities

The runtime must expose both AMD device interfaces:

```text
/dev/kfd
/dev/dri
```

The process must access the `video` and `render` device groups. The Docker
adapter also uses host networking, host IPC, an 8 GiB shared-memory segment,
and an unconfined seccomp profile, matching the tracked Compose file. The
Apptainer adapter uses `--rocm`, `--cleanenv`, and the same stable binds. These
are capability requirements, not site-specific scheduler settings.

## Environment variables

Set the host values before invoking
`gpu_ros_object_detection/docker/phase2-amd.sh`. The launcher exports stable
container paths:

| Variable | Required | Meaning |
| --- | --- | --- |
| `OVG_RUNTIME` | Yes | Explicitly selects `docker` or `apptainer`; do not rely on auto-detection for a formal run. |
| `OVG_STATE_ROOT` | Yes | Host root for persistent assets, cache, results, image, build, install, log, and home state. |
| `OVG_ORT_STATE_HOST` | Yes | Host directory mounted at `/workspaces/ovg-ort`. |
| `OVG_ORT_ROOT` | Yes for formal work | Container path of the complete external ORT install, normally `/workspaces/ovg-ort/install/<fingerprint>`. |
| `AMD_GPU_TARGETS` | Yes | Comma- or semicolon-separated `gfx...` target(s) used for ORT and application compilation. |
| `OVG_APPTAINER_SIF` | Apptainer only | Host path of the validated SIF. It is an adapter input, not a canonical filename. |
| `OVG_ASSETS_ROOT` | Inside runtime | Stable asset path; the launcher sets `/workspaces/ovg-assets`. |
| `OVG_CACHE_ROOT` | Inside runtime | Stable cache path; the launcher sets `/workspaces/ovg-cache`. |
| `OVG_RESULTS_ROOT` | Inside runtime | Stable result path; the launcher sets `/workspaces/ovg-results`. |
| `GPU_ROS_REPO_ROOT` | Inside runtime | Stable repository path; the launcher sets `/workspaces/gpu-ros`. |
| `OVG_WORKSPACE_ROOT` | Inside runtime | Workspace path; the launcher sets `/workspaces/gpu-ros`. |
| `ROS_DISTRO` | Inside runtime | Must be `jazzy`. |
| `COLCON_DEFAULTS_FILE` | Inside runtime | Must be the tracked `gpu_ros_object_detection/docker/colcon-defaults-phase2a-amd.yaml` under `/workspaces/gpu-ros`. |

The launcher derives `OVG_RUNTIME_EFFECTIVE`, `OVG_IMAGE_FINGERPRINT`, and
`OVG_WORKSPACE_FINGERPRINT`. Record them, but do not hand-edit them to bypass
the gates. `OVG_ORT_BUILD_MODE=1` and `OVG_ALLOW_BUILD_ONLY_SHELL=1` are only
for constructing the external ORT install; they are prohibited as the formal
benchmark runtime.

## Optional site adapters

Slurm, Kubernetes, login/compute-node separation, and local allocation limits
are adapters around this contract. They are not required steps in the
canonical runbook. If a site wants the launcher to enforce an allocation, set
`OVG_REQUIRE_SLURM=1`; the default `0` only warns. A site may also set
`OVG_SLURM_PARTITION_REGEX` for its own policy. Neither variable changes the
software, mounts, devices, or benchmark acceptance criteria. Allocation details
and private host configuration stay outside the repository and this public
contract.

## Non-interactive Docker execution

Create an override outside Git with the verified existing image ID, a unique
Compose project, fresh build/install/log and result binds, and the verified
assets, cache, ORT, and home state. Keep device, network, IPC, group, and security
settings from the tracked Compose file. Use the mounted checkout entrypoint.

In the override, set the existing `OVG_IMAGE_FINGERPRINT` to the first 32
characters of the verified Docker image ID after removing `sha256:`. Set
`OVG_WORKSPACE_FINGERPRINT` to the selected external ORT fingerprint and
`OVG_ORT_ROOT` to that complete install. These are the values derived by
`set_fingerprint()`, not arbitrary labels or `unresolved`. Keep host binds
absolute and outside Git.

```bash
export OVG_RUNTIME=docker
AMD_COMPOSE="${REPO_ROOT}/gpu_ros_object_detection/docker/docker-compose.phase2a-amd.yaml"
AMD_OVERRIDE="${VERIFY_ROOT}/amd.override.yaml"
docker compose -f "${AMD_COMPOSE}" -f "${AMD_OVERRIDE}" config
docker compose -f "${AMD_COMPOSE}" -f "${AMD_OVERRIDE}" up -d --no-build amd
docker compose -f "${AMD_COMPOSE}" -f "${AMD_OVERRIDE}" exec -T amd \
  /bin/bash \
  /workspaces/gpu-ros/gpu_ros_object_detection/docker/phase2a-amd-entrypoint.sh \
  /bin/bash -c 'set -e; phase2 build; phase2 env --verify'
```

Every subsequent invocation uses both files. Do not call a launcher branch
that starts another container without the override. For a requested command
sequence in `PAYLOAD`, use:

```bash
docker compose -f "${AMD_COMPOSE}" -f "${AMD_OVERRIDE}" exec -T amd \
  /bin/bash \
  /workspaces/gpu-ros/gpu_ros_object_detection/docker/phase2a-amd-entrypoint.sh \
  /bin/bash -c "set -e; ulimit -c 0; ${PAYLOAD}"
```

The 2026-09-18 campaign exercised Apptainer on AMD, not Docker GPU execution or
a fresh dependency-image build. Do not report those untested branches as
verified by that campaign.

## Explicit Slurm Apptainer execution

Obtain an authorized compute allocation outside this method. Confirm the
existing SIF, external ORT install, device target, assets, and state paths.
Use one-shot execution, with asset regeneration disabled:

```bash
export OVG_RUNTIME=apptainer
export OVG_APPTAINER_INSTANCE=0
export OVG_REQUIRE_SLURM=1
export OVG_PREPARE_ASSETS=0
unset APPTAINERENV_HOME
./gpu_ros_object_detection/docker/phase2-amd.sh preflight
./gpu_ros_object_detection/docker/phase2-amd.sh colcon
./gpu_ros_object_detection/docker/phase2-amd.sh verify
```

The launcher does not expose a general `exec` subcommand. For further payloads,
use its existing `apptainer_args()` bind/environment table. The example below
uses the state directories already created by the launcher. Its fingerprints
are derived from the verified SIF and selected ORT path exactly as in
`set_fingerprint()`; check them against `phase2 env --verify`. Do not source the
launcher as a shell library or invent another runtime wrapper.

```bash
SIF_DIGEST="$(sha256sum "${OVG_APPTAINER_SIF}")"
SIF_DIGEST="${SIF_DIGEST%% *}"
ORT_FINGERPRINT="${OVG_ORT_ROOT##*/}"
test -d "${OVG_STATE_ROOT}/build/${ORT_FINGERPRINT}"
test -d "${OVG_STATE_ROOT}/install/${ORT_FINGERPRINT}"
apptainer exec --rocm --cleanenv --pwd /workspaces/gpu-ros \
  --bind "${REPO_ROOT}:/workspaces/gpu-ros" \
  --bind "${OVG_STATE_ROOT}/assets:/workspaces/ovg-assets" \
  --bind "${OVG_STATE_ROOT}/cache:/workspaces/ovg-cache" \
  --bind "${OVG_STATE_ROOT}/results:/workspaces/ovg-results" \
  --bind "${OVG_STATE_ROOT}/build/${ORT_FINGERPRINT}:/workspaces/gpu-ros/build" \
  --bind "${OVG_STATE_ROOT}/install/${ORT_FINGERPRINT}:/workspaces/gpu-ros/install" \
  --bind "${OVG_STATE_ROOT}/log/${ORT_FINGERPRINT}:/workspaces/gpu-ros/log" \
  --bind "${OVG_ORT_STATE_HOST}:/workspaces/ovg-ort" \
  --bind "${OVG_STATE_ROOT}/home:/home/ovg" \
  --env HOME=/home/ovg \
  --env OVG_WORKSPACE_ROOT=/workspaces/gpu-ros \
  --env GPU_ROS_REPO_ROOT=/workspaces/gpu-ros \
  --env OVG_ASSETS_ROOT=/workspaces/ovg-assets \
  --env OVG_CACHE_ROOT=/workspaces/ovg-cache \
  --env OVG_RESULTS_ROOT=/workspaces/ovg-results \
  --env OVG_ORT_STATE_ROOT=/workspaces/ovg-ort \
  --env "OVG_ORT_ROOT=${OVG_ORT_ROOT}" \
  --env OVG_ORT_BUILD_MODE=0 \
  --env "OVG_IMAGE_FINGERPRINT=${SIF_DIGEST:0:32}" \
  --env "OVG_WORKSPACE_FINGERPRINT=${ORT_FINGERPRINT}" \
  --env OVG_RUNTIME_EFFECTIVE=apptainer \
  --env ROS2_BENCHMARK_OVERRIDE_ASSETS_ROOT=/workspaces/ovg-assets \
  "${OVG_APPTAINER_SIF}" /bin/bash \
  /workspaces/gpu-ros/gpu_ros_object_detection/docker/phase2a-amd-entrypoint.sh \
  /bin/bash -c "set -e; ulimit -c 0; ${PAYLOAD}"
```

`shell` is for manual debugging, not acceptance. Do not start Docker on a login
node or publish site submission commands, accounts, or host paths.

## Runtime gates and provenance capture

Inside the initialized payload, inspect the environment and selected assets:

```bash
phase2 env --verify
phase2 assets status
dpkg-query -W -f='${Package}=${Version}\n' migraphx migraphx-dev
colcon test --event-handlers console_direct+
colcon test-result --all --verbose
```

Keep command failures visible. A pre-existing failure still appears as FAIL;
only explicitly authorized diagnostics or measurements may proceed without
promotion. Record component logs because wrapper exit codes alone can miss
teardown faults.

`phase2 env --verify` must show Jazzy, ROCm 7.1.1, the actual device target,
external ORT fingerprint/provider library, and image manifest. Record the
loaded library paths and digests alongside the configured path. The
external archive also retains source revision and dirty/untracked hashes,
model and per-file data hashes, runtime digest, GPU/driver identity, commands,
reports, and post-exit process lists.

Matrix manifests use schema 3 and one monorepo identity. Capture logs and
promoted summaries remain unversioned. See
[Result records](../../../docs/results/README.md) for legacy interpretation
and [result acceptance](result-acceptance.md) for gates.
