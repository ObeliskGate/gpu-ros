# AMD runtime contract

This document is the public, canonical contract for the AMD experiments. It
defines the software and container interface needed to reproduce an equivalent
run. A site may wrap it in Docker, Apptainer, or a scheduler adapter, but a
private hostname, partition name, SIF filename, or deployment path is not part
of the experiment definition.

## Required software identity

The following identities are the baseline for formal Phase 2A and Phase 2B
runs:

| Layer | Required identity | Public source of truth |
| --- | --- | --- |
| Base OS | Ubuntu 24.04 userland | `AMD_BASE_IMAGE` below |
| ROS | ROS 2 Jazzy | `ROS_DISTRO=jazzy` and the Dockerfile |
| ROCm/HIP | ROCm 7.1.1 | `rocm/dev-ubuntu-24.04:7.1.1-complete` and `phase2 env --verify` |
| GPU target | The actual target reported by the device; the campaign baseline is `gfx950` | `AMD_GPU_TARGETS` and `rocminfo` |
| GPU driver | The compatible AMD kernel/user-space driver release for the device | `rocminfo`, `rocminfo --version`, and the site package query; record the release in the result archive |
| MIGraphX | The MIGraphX 2.14 behavior shipped by the ROCm 7.1.1 package set | `migraphx`, `migraphx-dev`, and the provider audit |
| ONNX Runtime | 1.23.1 at commit `d9b2048791efb5804fe3d53a04b4971256addebf` | `config/onnxruntime.lock` |
| ORT provider patches | `onnxruntime-1.23.1-migraphx-enable-gridsample.patch` and `onnxruntime-1.23.1-migraphx-int64-div-cpu-fallback.patch` | `docker/patches/onnxruntime-1.23.1-migraphx-*.patch` and `.series` |
| Benchmark framework | `ros2_benchmark` `v4.5-0`, plus the tracked standalone patch | `docker/phase2a-amd.Dockerfile` |

MIGraphX is installed from the ROCm package repository rather than built from
a separately pinned source checkout in this repository. The package revision
must therefore be captured in each raw run archive with:

```bash
dpkg-query -W -f='${Package}=${Version}\n' migraphx migraphx-dev
migraphx-driver --version 2>/dev/null || true
```

If a site supplies a different package revision, that is a different software
baseline even when the ROCm marketing version is unchanged.

## Container recipe and image identity

The canonical OCI recipe is tracked in
`docker/phase2a-amd.Dockerfile` and
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
SHA-256. The SIF's local filename and location are site choices. A formal
result must contain either that immutable SIF digest or the recipe revision and
base-image identity above; a bare statement such as “approved runtime” is not
enough.

## Container paths and mounts

The launcher exposes the following stable paths inside either runtime:

| Container path | Required content | Host-side choice |
| --- | --- | --- |
| `/workspaces/amd_ros_object_detection` | this application checkout | any source checkout |
| `/workspaces/amd_ros_object_detection/src/gpu_ros_managed` | selected `gpu_ros_managed` sibling checkout | any sibling checkout |
| `/workspaces/ovg-assets` | imported model and R2B assets | external asset directory |
| `/workspaces/ovg-cache` | persistent MIGraphX compilation cache | external cache directory |
| `/workspaces/ovg-results` | benchmark, audit, and comparison output | external result directory |
| `/workspaces/ovg-ort` | external ORT source/build/install state | external ORT state directory |
| `/workspaces/amd_ros_object_detection/build` | persistent colcon build tree | optional but recommended state mount |
| `/workspaces/amd_ros_object_detection/install` | persistent colcon install tree | optional but recommended state mount |
| `/workspaces/amd_ros_object_detection/log` | colcon logs | optional state mount |

The application, sibling, assets, cache, results, and ORT mounts are required
for a formal run. Build/install/log may be ephemeral only when the complete
build is reproduced in the same invocation. Do not mount a hidden fallback ORT
or a second asset root.

## GPU and runtime capabilities

The runtime must expose both AMD device interfaces:

```text
/dev/kfd
/dev/dri
```

The process must be able to access the `video` and `render` device groups. The
Docker adapter also uses host networking, host IPC, an 8 GiB shared-memory
segment, and an unconfined seccomp profile, matching the tracked compose file.
The Apptainer adapter uses `--rocm`, `--cleanenv`, and the same stable binds.
These are capability requirements, not site-specific scheduler settings.

## Environment variables

Set the following before invoking `docker/phase2-amd.sh`:

| Variable | Required | Meaning |
| --- | --- | --- |
| `OVG_RUNTIME` | yes | Explicitly selects `docker` or `apptainer`; do not rely on auto-detection for a formal run. |
| `GPU_ROS_MANAGED_DIR` | yes | Host path of the sibling checkout; it is mounted at the stable container path above. |
| `OVG_STATE_ROOT` | yes | Host root for persistent assets, cache, results, image, build, install, log, and home state. |
| `OVG_ORT_STATE_HOST` | yes | Host directory mounted at `/workspaces/ovg-ort`. |
| `OVG_ORT_ROOT` | yes for formal work | Container path of the complete external ORT install, normally `/workspaces/ovg-ort/install/<fingerprint>`. |
| `AMD_GPU_TARGETS` | yes | Comma- or semicolon-separated `gfx...` target(s) used for ORT and application compilation. |
| `OVG_APPTAINER_SIF` | Apptainer only | Host path of the validated SIF. It is an adapter input, not a canonical filename. |
| `OVG_ASSETS_ROOT` | inside runtime | Stable container path for assets; the launcher sets `/workspaces/ovg-assets`. |
| `OVG_CACHE_ROOT` | inside runtime | Stable container path for caches; the launcher sets `/workspaces/ovg-cache`. |
| `OVG_RESULTS_ROOT` | inside runtime | Stable container path for results; the launcher sets `/workspaces/ovg-results`. |
| `OVG_WORKSPACE_ROOT` | inside runtime | Stable application path; the launcher sets `/workspaces/amd_ros_object_detection`. |
| `ROS_DISTRO` | inside runtime | Must be `jazzy`. |
| `COLCON_DEFAULTS_FILE` | inside runtime | Must be the tracked `docker/colcon-defaults-phase2a-amd.yaml`. |

The launcher derives `OVG_RUNTIME_EFFECTIVE`, `OVG_IMAGE_FINGERPRINT`, and
`OVG_WORKSPACE_FINGERPRINT`. Record them, but do not hand-edit them to bypass
the gates. `OVG_ORT_BUILD_MODE=1` and `OVG_ALLOW_BUILD_ONLY_SHELL=1` are only
for constructing the external ORT install; they are prohibited as the formal
benchmark runtime.

## Optional site adapters

Slurm, Kubernetes, a login/compute-node distinction, and local allocation
limits are adapters around this contract. They are not required steps in the
canonical runbook. If a site wants the launcher to enforce a Slurm allocation,
set `OVG_REQUIRE_SLURM=1`; the default `0` only warns. A site may also set
`OVG_SLURM_PARTITION_REGEX` for its own policy. Neither variable changes the
software, mounts, devices, or benchmark acceptance criteria.

## Runtime gate and provenance capture

After the mounts and external ORT are ready, the canonical gates are:

```bash
./docker/phase2-amd.sh preflight
./docker/phase2-amd.sh colcon
./docker/phase2-amd.sh verify
phase2 env --verify
dpkg-query -W -f='${Package}=${Version}\n' migraphx migraphx-dev
```

`phase2 env --verify` must show ROS Jazzy, ROCm 7.1.1, the actual `gfx` target,
the external ORT fingerprint, the MIGraphX provider library, and the image
manifest. A result archive must additionally retain the application HEAD, the
managed sibling HEAD, the ORT source/patch identities, the GPU model, and the
container recipe revision or immutable image digest. Host/user/scheduler IDs
and deployment paths may remain in a private archive and must not be used as a
substitute for this provenance.
