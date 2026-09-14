# GPU ROS

GPU ROS is a ROS 2 monorepo for object-detection pipelines and explicit
GPU-buffer transport. It contains RT-DETRv2 and YOLOv8 application packages,
the project-owned TensorBundle message, the managed CUDA/HIP transport, and the
runbooks used to compare standard ROS 2 message boundaries with same-process
managed buffers.

The primary supported experiment target is ROS 2 Jazzy on Ubuntu 24.04 with
ROCm 7.1.1, a target reported by the AMD device, and the project-built ONNX
Runtime 1.23.1/MIGraphX installation selected by [`config/onnxruntime.lock`](config/onnxruntime.lock).
The NVIDIA lane uses the pinned Isaac ROS 4.5-compatible environment described
in the NVIDIA runbooks. Other combinations may work, but are not the current
compatibility target.

> [!IMPORTANT]
> The packages are versioned `0.1.0` and remain pre-release. This repository
> does not claim support for every Isaac ROS object-detection package. Grounding
> DINO and DetectNet are historical reference workloads, not migrated packages.

## Architecture

Top-level directories classify components; they do not rename or split ROS
packages. Package directory basenames and ROS package names remain `gpu_ros_*`.

| Directory | Contents and boundary |
| --- | --- |
| [`transport/`](transport/) | Standalone C++17 managed buffers, optional CUDA/HIP backends, and intra-process ROS adapters. The core has no ROS or GPU-SDK dependency. |
| [`interfaces/`](interfaces/) | Project-owned `gpu_ros_tensor_bundle_msgs` message definitions. |
| [`perception/`](perception/) | Shared detection utilities, ONNX Runtime integration, RT-DETRv2, and YOLOv8 packages. |
| [`compat/`](compat/) | Optional NVIDIA TensorList compatibility package. It is not part of the AMD profile and is the only project package allowed to depend on NVIDIA TensorList interfaces. |
| [`evaluation/`](evaluation/) | Fixed-input capture, comparison, provider/copy audits, and benchmark graph definitions. `evaluation/benchmarks/` is a non-package directory. |
| [`docker/`](docker/) and [`apptainer/`](apptainer/) | AMD and NVIDIA runtime entry points and pinned build profiles. |
| [`config/`](config/) | ONNX Runtime and model-profile identities used by the runbooks. |
| [`external/`](external/) | Pinned NVIDIA external-source manifest; external checkouts remain outside this repository's package tree. |
| [`docs/`](docs/) | Public experiment and result runbooks. |
| [`skills/gpu-ros-experiments/SKILL.md`](skills/gpu-ros-experiments/SKILL.md) | Optional concise instructions for agents. The project works without installing this skill. |

The standard ROS path uses ordinary TensorBundle messages:

```text
sensor_msgs/Image
        |
        v
image encoder -> TensorBundle -> ONNX Runtime -> TensorBundle -> decoder
                                                                  |
                                                                  v
                                              vision_msgs/Detection2DArray
```

The managed path keeps the graph in one component process and passes owned
CUDA or HIP buffers through the transport packages:

```text
Image -> managed preprocessing -> ORT/provider -> managed decoder
                                                        |
                                                        v
                                    Detection2DArray on the CPU
```

Managed transport is an intra-process ownership and synchronization contract.
Crossing a ROS serialization boundary materializes device tensors in host
memory, and neither the complete ROS graph nor model execution is promised to
be copy-free.

## Requirements and external inputs

For an AMD run, provide:

- an AMD GPU with `/dev/kfd` and `/dev/dri` access and the target reported by
  `rocminfo`;
- Docker Compose or Apptainer;
- the ROS/ROCm environment specified by
  [`amd-runtime-contract.md`](docs/experiments/amd-runtime-contract.md);
- an external ORT install built from the locked source revision and selected by
  `OVG_ORT_ROOT`; the image's `/opt/onnxruntime` copy is build-only content;
- model, R2B, and (when applicable) dataset assets imported into an external
  asset root.

For NVIDIA, use the pinned image and external sources selected by
[`external/nvidia-isaac-ros.repos`](external/nvidia-isaac-ros.repos). External
checkouts belong under `/workspaces/isaac_ros-dev/src/nvidia_external`, not in
the repository checkout. The compatibility package is an opt-in NVIDIA lane;
AMD profiles do not build it.

Models, checkpoints, ONNX files, engines, datasets, bags, traces, profiles,
logs, SIF images, caches, and benchmark output are not redistributed. A
SHA-256 value verifies compatibility; it does not grant a model or dataset
license. Do not add download or export steps for user-owned YOLOv8 weights.

## Checkout and AMD workflow

Clone one repository. The transport and application sources are already in the
same worktree; no sibling checkout or second source bind is required.

```bash
git clone https://github.com/gpu-ros/gpu-ros.git
cd gpu-ros
```

The AMD launcher uses `/workspaces/gpu-ros` inside Docker or Apptainer for both
the checkout and `GPU_ROS_REPO_ROOT`/`OVG_WORKSPACE_ROOT`. Keep persistent state,
assets, cache, results, ORT, and build/install/log directories on their separate
mounts. Choose a fresh state root for a migration-specific run:

```bash
export OVG_RUNTIME=docker                 # or apptainer
export OVG_STATE_ROOT=/absolute/path/to/gpu-ros-state
export OVG_ORT_STATE_HOST="${OVG_STATE_ROOT}/ort"
export OVG_ORT_ROOT=/workspaces/ovg-ort/install/<ort-fingerprint>
export AMD_GPU_TARGETS=<target-reported-by-rocminfo>
# Apptainer only:
# export OVG_APPTAINER_SIF=/absolute/path/to/validated-runtime.sif

# Build the image/SIF only when a usable dependency runtime is not available.
./docker/phase2-amd.sh build
./docker/phase2-amd.sh preflight
./docker/phase2-amd.sh up
./docker/phase2-amd.sh colcon
./docker/phase2-amd.sh verify
./docker/phase2-amd.sh shell
```

Inside the runtime shell, use the existing helper and selected external assets:

```bash
phase2 env --verify
phase2 assets status
phase2 assets prepare \
  --model-profile rtdetrv2_r50 \
  --execution-provider migraphx
```

The helper is offline. Import user-held files from the repository root with
`./tools/phase2-assets import-model` or `import-r2b`; the exact model and
checkpoint gates are in [`rtdetrv2-validation.md`](docs/experiments/rtdetrv2-validation.md).
Run a fixed-input capture only after the asset and provider gates pass:

```bash
CAPTURE_EXECUTION_PROVIDER=migraphx \
CAPTURE_TRANSPORT=managed \
CAPTURE_MIN_MESSAGES=20 \
ros2 run gpu_ros_detection_validation \
  run_amd_phase2a_fixed_input_capture.sh \
  "monorepo_rtdetr_managed_<run-id>"
```

Use the existing `yolov8` profile only when compatible user-owned YOLOv8
assets are available. Do not label a different model as RT-DETRv2. The
capture, bag comparator, audit, and matrix entry points are documented under
[`evaluation/`](evaluation/) and the [experiment index](docs/experiments/README.md).
A benchmark matrix is explicit work, not a default smoke command.

## NVIDIA workflow

NVIDIA retains an outer Isaac ROS workspace and a repository checkout below it:

```text
/workspaces/isaac_ros-dev/                 outer colcon workspace
/workspaces/isaac_ros-dev/src/gpu-ros/     this repository
/workspaces/isaac_ros-dev/src/nvidia_external/  pinned external checkouts
```

The Compose file mounts the monorepo once at `src/gpu-ros` and keeps the outer
workspace as the working directory. From the checked-out repository path,
resolve the Compose configuration and run the existing launcher:

```bash
cd /workspaces/isaac_ros-dev/src/gpu-ros
docker compose -f docker-compose.yaml config
./docker/phase1-nvidia.sh up
./docker/phase1-nvidia.sh colcon
./docker/phase1-nvidia.sh verify
./docker/phase1-nvidia.sh shell
```

Inside the runtime shell, the workspace is `/workspaces/isaac_ros-dev` and the
fresh install is sourced from there. A managed capture keeps its repository-
relative path under `src/gpu-ros`:

```bash
src/gpu-ros/evaluation/gpu_ros_detection_validation/scripts/run_nvidia_fixed_input_capture.sh \
  yolov8-managed "monorepo_yolov8_managed_<run-id>"
```

Use `rtdetr-managed` only when the licensed NVIDIA RT-DETR assets are present.
Do not change the external source pins, image boundary, or NVIDIA runtime
hooks to make another environment resemble AMD.

## Standalone transport build

The backend-neutral core builds without ROS or a GPU SDK. From the monorepo
root, use an out-of-tree build directory:

```bash
cmake -S transport -B /tmp/gpu-ros-transport-core \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_DISABLE_FIND_PACKAGE_ament_cmake=TRUE \
  -DGPU_ROS_MANAGED_BUILD_TESTING=ON \
  -DGPU_ROS_MANAGED_BUILD_CUDA=OFF \
  -DGPU_ROS_MANAGED_BUILD_HIP=OFF
cmake --build /tmp/gpu-ros-transport-core
ctest --test-dir /tmp/gpu-ros-transport-core --output-on-failure
```

The same standalone tree can be configured from its own directory. The
working directory matters:

```bash
cd transport
cmake -S . -B /tmp/gpu-ros-transport-core \
  -DGPU_ROS_MANAGED_BUILD_CUDA=OFF \
  -DGPU_ROS_MANAGED_BUILD_HIP=OFF
cmake --build /tmp/gpu-ros-transport-core
ctest --test-dir /tmp/gpu-ros-transport-core --output-on-failure
```

Enable CUDA or HIP only when the matching SDK and device are available. ROS
adapters are built with colcon from a ROS workspace using the package roots
`transport` and `interfaces`; see [`transport/README.md`](transport/README.md).

## Development and verification

Keep changes within their owning top-level category and preserve package names,
public CMake targets, message schemas, include namespaces, plugins, model
contracts, external pins, and provider settings. Use the runbook for the lane
being changed:

- [AMD runtime contract](docs/experiments/amd-runtime-contract.md)
- [Phase 2A AMD standard path](docs/experiments/phase2a-amd.md)
- [Phase 2B managed HIP](docs/experiments/phase2b-managed.md)
- [Phase 1 NVIDIA reference](docs/experiments/phase1-nvidia.md)
- [Phase 2B NVIDIA managed comparison](docs/experiments/phase2b-nvidia.md)
- [RT-DETRv2 validation](docs/experiments/rtdetrv2-validation.md)
- [Result and provenance rules](docs/results/README.md)

Use `./tools/phase2` only for its existing `env`, `assets`, `cache`, and
`build` groups. It does not provide `phase2 test`, `phase2 smoke`, or
`phase2 benchmark` subcommands. Capture and matrix scripts retain their
existing arguments and thresholds.

New run records use one monorepo identity. Matrix manifests use schema 3;
fixed-input capture logs and promoted summaries remain unversioned and use
`monorepo_revision` fields. Legacy records are interpreted according to the
separate rules in [`docs/results/README.md`](docs/results/README.md), and
historical archives are not rewritten.

Run focused checks for the layer you changed, keep generated state outside Git,
and report the actual command, revision, runtime/provider, model/data identity,
and result location. Do not claim performance or correctness from a smoke run.

## Contributing and security

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before submitting a change and
[`SECURITY.md`](SECURITY.md) before reporting a vulnerability. Do not commit
credentials, private deployment paths, scheduler identifiers, runtime images,
model/data bytes, bags, traces, caches, or generated results.

Project-authored code is Apache-2.0. File-level notices and external component
terms remain authoritative; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
and [`transport/THIRD_PARTY_NOTICES.md`](transport/THIRD_PARTY_NOTICES.md).
This is an independent project and is not affiliated with or endorsed by
NVIDIA or AMD.
