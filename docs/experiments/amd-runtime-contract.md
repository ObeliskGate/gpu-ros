# AMD runtime contract

This document is the public, canonical contract for GPU ROS AMD experiments.
It defines the software, container, mount, device, and environment interface
needed to reproduce an equivalent run. A site may wrap it in Docker, Apptainer,
or a scheduler adapter, but a private hostname, partition/account name, SIF
filename, or deployment path is not part of the experiment definition.

The AMD layout has one repository checkout. The transport packages are under
that checkout; there is no managed sibling repository, sibling state, or second
source bind.

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
| ONNX Runtime | 1.23.1 at the commit in `config/onnxruntime.lock` | `config/onnxruntime.lock` |
| ORT provider patches | The tracked GridSample and int64-div CPU-fallback patches | `docker/patches/onnxruntime-1.23.1-migraphx-*.patch` and `.series` |
| Benchmark framework | `ros2_benchmark` `v4.5-0`, plus the tracked standalone patch | `docker/phase2a-amd.Dockerfile` |

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

The canonical OCI recipe is tracked in `docker/phase2a-amd.Dockerfile` and
`docker-compose.phase2a-amd.yaml`:

```bash
export OVG_RUNTIME=docker
export AMD_BASE_IMAGE=rocm/dev-ubuntu-24.04:7.1.1-complete
export AMD_GPU_TARGETS=gfx950
docker compose -f docker-compose.phase2a-amd.yaml build amd
```

The image builds ROS 2 Jazzy, MIGraphX, the benchmark framework, and a
build-only ORT copy. Formal runs must use the separately built external ORT
install selected by `OVG_ORT_ROOT`; `/opt/onnxruntime` in the image is not a
formal fallback. Build that install from the locked recursive ORT checkout with
`tools/build-phase2a-external-ort.sh`, and retain its `build-info.txt` and
fingerprint marker.

Apptainer is an equivalent packaging adapter for the same OCI image. Convert
the validated image with `apptainer/phase2-amd.def`, then record the full SIF
SHA-256. The SIF filename and location are site choices. A formal result must
contain either that immutable SIF digest or the recipe revision and base-image
identity above; “approved runtime” alone is not enough.

## Container paths and mounts

The launcher exposes these stable paths inside either runtime:

| Container path | Required content | Host-side choice |
| --- | --- | --- |
| `/workspaces/gpu-ros` | The single GPU ROS repository checkout, including `transport/` and application packages | Any source checkout |
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

Set the host-side values below before invoking `docker/phase2-amd.sh`. The
launcher exports the stable in-container paths:

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
| `COLCON_DEFAULTS_FILE` | Inside runtime | Must be the tracked `docker/colcon-defaults-phase2a-amd.yaml` under `/workspaces/gpu-ros`. |

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

## Runtime gates and provenance capture

After the mounts and external ORT are ready, use these gates:

```bash
./docker/phase2-amd.sh preflight
./docker/phase2-amd.sh colcon
./docker/phase2-amd.sh verify
phase2 env --verify
dpkg-query -W -f='${Package}=${Version}\n' migraphx migraphx-dev
```

`phase2 env --verify` must show ROS Jazzy, ROCm 7.1.1, the actual `gfx` target,
the external ORT fingerprint, the MIGraphX provider library, and the image
manifest. A result archive additionally retains the monorepo revision,
monorepo diff/untracked fingerprints, ORT source/patch identities, GPU model,
and container recipe revision or immutable image/SIF digest. Host, user,
scheduler, and deployment-path values may remain in a private archive and are
not substitutes for this provenance.

New matrix manifests use schema 3 and one `monorepo_revision`; fixed-input
capture logs remain unversioned and use `monorepo_revision` plus
`monorepo_worktree_diff_sha256`. Promoted summaries also remain unversioned and
use `monorepo_revision`. See [`../results/README.md`](../results/README.md) for
legacy records and the separate format rules.
