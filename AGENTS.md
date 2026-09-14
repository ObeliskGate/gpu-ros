# AGENTS.md

## Project overview

GPU ROS is a ROS 2 monorepo for RT-DETRv2 and YOLOv8 object detection and
explicit GPU-buffer transport. It targets ROS 2 Jazzy on Ubuntu 24.04 with
ROCm 7.1.1, the selected AMD GPU target, and the ONNX Runtime/MIGraphX build
locked in `config/onnxruntime.lock`. NVIDIA reference work uses the pinned
Isaac ROS 4.5-compatible runtime. These are compatibility targets, not a claim
that every ROS, GPU, driver, or provider combination is supported.

The project works without an agent skill. The optional public skill at
[`skills/gpu-ros-experiments/SKILL.md`](skills/gpu-ros-experiments/SKILL.md)
only points agents to the checked-out runbooks and existing commands.

## Monorepo architecture and ownership

Top-level directories classify components. Do not rename, shorten, split, or
relocate the actual package directories or ROS package names:

- `transport/` contains the standalone `gpu_ros_managed_*` C++17 core and
  optional CUDA/HIP/ROS/TensorBundle adapters.
- `interfaces/` contains `gpu_ros_tensor_bundle_msgs`.
- `perception/` contains `gpu_ros_detection_common`,
  `gpu_ros_onnx_inference`, `gpu_ros_rtdetr`, and `gpu_ros_yolov8`.
- `compat/` contains the optional
  `gpu_ros_nvidia_tensor_bundle_compat` package. It is the only project package
  allowed to depend on NVIDIA TensorList interfaces and is excluded from AMD
  profiles.
- `evaluation/` contains validation tools and `benchmarks/`; the latter is a
  graph-definition directory, not a ROS package.
- `docker/`, `apptainer/`, `config/`, and `external/` contain runtime,
  configuration, and pinned external-source inputs.

The managed transport is an intra-process ownership and synchronization layer.
It does not own model logic, preprocessing, decoding, provider policy, or an
inter-process zero-copy promise. ROS serialization of device-backed tensors
performs a host copy. Keep the project TensorBundle schema distinct from the
NVIDIA TensorList compatibility types.

When changing a public `ITensorBundleIO` or managed-buffer contract, update all
callers and the affected transport/application packages together. Preserve
model input/output names, shapes, provider selection, stream/device rules,
plugins, CMake targets, and package exports.

## Repository and runtime paths

From the AMD runtime, the one source checkout is:

```text
/workspaces/gpu-ros
```

`GPU_ROS_REPO_ROOT` and `OVG_WORKSPACE_ROOT` refer to that repository path in
the AMD runtime. Assets, MIGraphX cache, results, external ORT state,
colcon build/install/log state, and home state use separate mounts. Do not add
a sibling checkout or a second source bind. AMD launchers require the external
ORT install selected by `OVG_ORT_ROOT`; `/opt/onnxruntime` in an image is not a
formal runtime fallback.

The NVIDIA runtime deliberately retains an outer workspace:

```text
/workspaces/isaac_ros-dev
/workspaces/isaac_ros-dev/src/gpu-ros
/workspaces/isaac_ros-dev/src/nvidia_external
```

The NVIDIA Compose file mounts this repository once at `src/gpu-ros`, keeps
`/workspaces/isaac_ros-dev` as the working directory, and preserves pinned
external checkouts and runtime hooks. NVIDIA scripts use
`GPU_ROS_REPO_ROOT` or default it to
`${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}/src/gpu-ros`.

Do not reintroduce obsolete managed-checkout variables, sibling checkout
checks, or old-path aliases. Do not change unrelated `OVG_*` meanings. Keep all
model/data/runtime outputs outside the source tree.

## Build and run commands

Use the existing launchers rather than duplicating their orchestration. From
the repository root, the AMD sequence is:

```bash
./docker/phase2-amd.sh preflight
./docker/phase2-amd.sh up
./docker/phase2-amd.sh colcon
./docker/phase2-amd.sh verify
./docker/phase2-amd.sh shell
```

Use `build` or `bootstrap` only when the existing dependency image/SIF is not
usable. Inside the runtime, run `phase2 env --verify` and
`phase2 assets status`; `tools/phase2` has only its existing `env`, `assets`,
`cache`, and `build` groups. Capture and benchmark scripts retain their
existing arguments and thresholds under `evaluation/`.

For NVIDIA, resolve the Compose file from the repository checkout, then use:

```bash
docker compose -f docker-compose.yaml config
./docker/phase1-nvidia.sh up
./docker/phase1-nvidia.sh colcon
./docker/phase1-nvidia.sh verify
./docker/phase1-nvidia.sh shell
```

Inside the NVIDIA shell, work from `/workspaces/isaac_ros-dev`; repository
scripts and benchmark definitions are under `src/gpu-ros/`.

The standalone transport core has no ROS or GPU SDK dependency. From the
monorepo root configure with `cmake -S transport`; from `transport/` configure
with `cmake -S .`. Use a fresh out-of-tree build directory. ROS adapter builds
must include the `transport` and `interfaces` roots in the same monorepo.

## Documentation and experiment rules

Read the relevant canonical runbook before changing or running a lane:

- `docs/experiments/amd-runtime-contract.md` for AMD mounts, devices,
  environment, ORT, and optional site allocation gates;
- `docs/experiments/phase2a-amd.md` for standard ROS 2 plus MIGraphX;
- `docs/experiments/phase2b-managed.md` for managed HIP topology;
- `docs/experiments/phase1-nvidia.md` and `phase2b-nvidia.md` for NVIDIA;
- `docs/experiments/rtdetrv2-validation.md` for the external RT-DETR export;
- `docs/results/README.md` for provenance and promotion semantics.

The public skill and these runbooks never contain Slurm allocation commands,
private hostnames, accounts, deployment paths, driver repair, or model bytes.
Those are external prerequisites. Do not add a scheduler adapter or claim a
smoke run reestablished a published performance result.

New matrix manifests use schema 3 and one `monorepo_revision`. Fixed-input
capture logs and promoted summaries remain unversioned and use
`monorepo_revision`/`monorepo_worktree_diff_sha256` as specified in the result
documentation. Matrix v2 and legacy capture records retain their historical
two-identity interpretation. Historical result archives remain byte-for-byte
unchanged.

## Assets, licenses, and security

The repository does not redistribute model weights, checkpoints, ONNX files,
engines, datasets, bags, traces, profiles, logs, caches, SIF images, or
external source trees. Import user-held assets with `tools/phase2-assets` and
record their configured hashes. A hash is a compatibility check, not a license
grant. Do not silently substitute a model or provider when an asset/provider
gate fails.

Preserve file-level copyright and modification notices. The root Apache-2.0
license, component notices, and external licenses remain authoritative. Never
commit credentials, private paths, scheduler/device identifiers, or generated
runtime output. Follow `CONTRIBUTING.md` for changes and `SECURITY.md` for
private vulnerability reports.

## Change and verification discipline

Keep a change within its owning top-level directory where possible. Update
relative documentation links and package resource paths when a file moves, but
do not refactor algorithms or benchmark parameters as part of a path migration.
Run focused checks for the affected layer, and report only commands actually
run. For runtime work record the monorepo commit, dirty/untracked fingerprints,
image or SIF identity, ORT/provider identity, model/data hashes, command, and
external result location.
