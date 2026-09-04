# amd_ros_object_detection

ROS 2 object detection with ONNX Runtime on AMD and NVIDIA GPUs.

This repository contains composable RT-DETRv2 and YOLOv8 pipelines, a
project-owned TensorBundle message, and the tools used to compare standard ROS 2
transport with same-process GPU buffer transport. The primary runtime is AMD
ROCm with the ONNX Runtime MIGraphX execution provider. CPU and NVIDIA
reference configurations are also available where noted.

The project started as a migration study of NVIDIA Isaac ROS object detection.
The application code under `migrated_packages/` now has its own package names
and interfaces. The external NVIDIA TensorList message dependency is confined to the optional compatibility package. Other CUDA/NITROS reference components retain their own external dependencies and license requirements.

> [!IMPORTANT]
> The packages are versioned `0.1.0`, and their APIs may still change. The
> project supports the pipelines listed below; it is not a drop-in replacement
> for every Isaac ROS object detection package.

## What is included

- RT-DETRv2 R50 and YOLOv8 image-to-detection pipelines
- ONNX Runtime inference with CPU and AMD MIGraphX, plus CUDA/NITROS reference
  configurations
- Standard ROS 2 transport using
  `gpu_ros_tensor_bundle_msgs/msg/TensorBundle`
- Direct Managed HIP pipelines that retain intermediate tensors in device
  memory inside one component container
- Launch files, fixed-input validation, provider-placement checks, transport
  audits, and benchmark graphs
- An explicit NVIDIA TensorList compatibility boundary for historical
  comparison runs

Grounding DINO and DetectNet are not implemented by the migrated application
packages.

## Pipeline overview

The standard path uses ordinary ROS messages between the preprocessing,
inference, and decoder components:

```text
sensor_msgs/Image
        |
        v
 image encoder -> TensorBundle -> ONNX Runtime -> TensorBundle -> decoder
                                                                  |
                                                                  v
                                              vision_msgs/Detection2DArray
```

The AMD managed path runs the components in one process and passes owned HIP
device buffers through
[`gpu_ros_managed`](https://github.com/ObeliskGate/gpu_ros_managed):

```text
Image -> Managed HIP preprocessing -> ORT/MIGraphX -> Managed HIP decoder
                                                               |
                                                               v
                                           Detection2DArray on the CPU
```

This managed path avoids application-level TensorBundle materialization at
the inference boundary. It does not claim that model execution or the complete
ROS graph performs no copies.

## Support matrix

| Lane | Models | Status |
| --- | --- | --- |
| AMD standard ROS 2 + MIGraphX | RT-DETRv2 R50, YOLOv8 | Supported |
| AMD Managed HIP + MIGraphX | RT-DETRv2 R50, YOLOv8 | Supported in one component process |
| CPU standard ROS 2 | RT-DETRv2 R50, YOLOv8 | Development and validation |
| NVIDIA CUDA, TensorRT, and NITROS | SyntheticaDETR, YOLOv8 | Reference and historical comparison |
| Grounding DINO and DetectNet | None | Not migrated |

The actively tested environment is ROS 2 Jazzy, Ubuntu 24.04, ROCm 7.1.1,
and ONNX Runtime 1.23.1 at the commit recorded in
[`config/onnxruntime.lock`](config/onnxruntime.lock). NVIDIA reference runs use
an Isaac ROS 4.5-compatible environment. Other version combinations may work,
but they are not the compatibility target of the current experiment.

## Requirements

For the primary AMD path you need:

- An AMD GPU supported by the selected ROCm release
- `/dev/kfd` and `/dev/dri` access
- Docker with Compose, or Apptainer
- ROS 2 Jazzy and ROCm 7.1.1 in the runtime environment
- A project-built ONNX Runtime with MIGraphX from the locked source revision
- A sibling checkout of `gpu_ros_managed`. The current package manifests use
  its shared types in both standard and managed builds.
- A compatible ONNX model and, for validation or benchmarks, an input bag or
  dataset

The supplied Dockerfile and launcher set up the application environment. ONNX
Runtime is built separately because its MIGraphX build is tied to the target
GPU and is treated as a recorded experiment input.

## Getting started on AMD

### 1. Check out both repositories

Keep the application and managed transport repositories next to each other:

```bash
git clone https://github.com/ObeliskGate/amd_ros_object_detection.git
git clone https://github.com/ObeliskGate/gpu_ros_managed.git
cd amd_ros_object_detection
```

### 2. Prepare the runtime and ONNX Runtime

Choose the container runtime and persistent state directory. Replace
`gfx942` with the target reported for your GPU by `rocminfo`:

```bash
export OVG_RUNTIME=docker
export GPU_ROS_MANAGED_DIR="$(realpath ../gpu_ros_managed)"
export OVG_STATE_ROOT="$(pwd)/.ovg"
export OVG_ORT_STATE_HOST="${OVG_STATE_ROOT}/ort"
export AMD_GPU_TARGETS=gfx942

./docker/phase2-amd.sh build
```

Build the locked ONNX Runtime revision into `OVG_ORT_STATE_HOST`, then set
`OVG_ORT_ROOT` to the fingerprinted install path. The exact checkout, patch,
and build procedure is in the
[`AMD runtime contract`](docs/experiments/amd-runtime-contract.md). This step
is required; the image's `/opt/onnxruntime` content is not a supported runtime
fallback.

For subsequent commands, the value is the path as seen inside the container:

```bash
export OVG_ORT_ROOT=/workspaces/ovg-ort/install/<ort-fingerprint>

./docker/phase2-amd.sh preflight
./docker/phase2-amd.sh bootstrap
./docker/phase2-amd.sh colcon
./docker/phase2-amd.sh verify
./docker/phase2-amd.sh shell
```

The launcher can also use Apptainer. See the runtime contract for the required
SIF and bind-mount variables.

### 3. Import a model

The asset helper works offline. It validates or imports files that you already
have; it does not download weights.

From the repository root on the host:

```bash
export OVG_ASSETS_ROOT="${OVG_STATE_ROOT}/assets"

./tools/phase2-assets import-model \
  --profile rtdetrv2_r50 \
  --source /absolute/path/to/rtdetrv2_r50.onnx

./tools/phase2-assets status
```

The formal RT-DETRv2 export recipe and expected digest are documented in
[`rtdetrv2-validation.md`](docs/experiments/rtdetrv2-validation.md). For
YOLOv8, import a compatible user-owned model with `--profile yolov8` instead.

### 4. Run a pipeline

Inside the prepared runtime shell, source the workspace and launch the standard
RT-DETR path:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 launch gpu_ros_rtdetr rtdetr_ort_std_image.launch.py \
  model_profile:=rtdetrv2_r50 \
  model_assets_root:="${OVG_ASSETS_ROOT}" \
  execution_provider:=migraphx \
  image_topic:=/image
```

Publish `sensor_msgs/msg/Image` messages on `/image`. Detections are published
as `vision_msgs/msg/Detection2DArray` on `/rtdetr/detections_output`.

Run the direct Managed HIP graph with:

```bash
ros2 launch gpu_ros_onnx_inference rtdetr_ort_managed_amd.launch.py \
  model_profile:=rtdetrv2_r50 \
  model_assets_root:="${OVG_ASSETS_ROOT}" \
  image_topic:=/image
```

For YOLOv8, use `yolov8_ort_std_image.launch.py` from `gpu_ros_yolov8` or
`yolov8_ort_managed_amd.launch.py` from `gpu_ros_onnx_inference` and pass
`model_file_path:=/absolute/path/to/yolov8s.onnx` when the default asset path
is not used.

## Build without the container launcher

A prepared ROS 2 workspace can build the package set directly. Starting from
the application repository root, with `gpu_ros_managed` checked out next to it:

```bash
source /opt/ros/jazzy/setup.bash
export ONNXRUNTIME_ROOT=/absolute/path/to/onnxruntime/install
export COLCON_DEFAULTS_FILE="$(pwd)/docker/colcon-defaults-phase2a-amd.yaml"

colcon build \
  --base-paths \
    migrated_packages \
    ../gpu_ros_managed
source install/setup.bash
```

The AMD defaults disable NITROS and CUDA, require the HIP SDK, and enable the
MIGraphX provider. If you create a custom build, do not request
`execution_provider:=migraphx` unless `ORT_ENABLE_MIGRAPHX=ON` was used at
configure time.

## Repository layout

| Path | Contents |
| --- | --- |
| `migrated_packages/gpu_ros_detection_common` | Shared image preprocessing and geometry |
| `migrated_packages/gpu_ros_onnx_inference` | ONNX Runtime sessions, providers, I/O binding, and transport selection |
| `migrated_packages/gpu_ros_rtdetr` | RT-DETR image encoder, preprocessor, and decoder |
| `migrated_packages/gpu_ros_yolov8` | YOLOv8 image encoder and decoder |
| `migrated_packages/gpu_ros_tensor_bundle_msgs` | Project-owned TensorBundle ROS messages |
| `migrated_packages/gpu_ros_nvidia_tensor_bundle_compat` | Optional NVIDIA TensorList adapter |
| `migrated_packages/gpu_ros_detection_validation` | Bag comparison, profiling, and audit tools |
| `migrated_packages/benchmarks` | Benchmark graph definitions |
| `docker` and `apptainer` | Reproducible AMD and NVIDIA runtime entry points |
| `docs/experiments` | Build, validation, and benchmark runbooks |
| `docs/results` | Published result summaries and their interpretation |

## Models and data

No model weights or datasets are redistributed by this repository.

`model_profile=auto` selects `rtdetrv2_r50` for CPU, ROCm, and MIGraphX, and
the historical `nvidia_synthetica` profile for CUDA. A selected asset must
exist and pass its configured digest check. The code never falls back to a
different model silently.

Available RT-DETR profiles and their expected paths, input contracts, source
revisions, and hashes live in
[`config/model-profiles.json`](config/model-profiles.json). The RT-DETRv2 R50
pipeline expects `images` and `orig_target_sizes` inputs and produces `labels`,
`boxes`, and `scores`. A generic Transformers RT-DETR export is not
interchangeable with this contract.

Users are responsible for the licenses and redistribution terms of their model
and dataset files. A recorded SHA-256 value is a compatibility check, not a
license grant.

## Tests

Run the package tests in a built workspace:

```bash
colcon test \
  --base-paths \
    migrated_packages \
    ../gpu_ros_managed \
  --packages-select \
    gpu_ros_managed_core \
    gpu_ros_managed_hip \
    gpu_ros_managed_ros \
    gpu_ros_managed_tensor_bundle \
    gpu_ros_tensor_bundle_msgs \
    gpu_ros_detection_common \
    gpu_ros_onnx_inference \
    gpu_ros_rtdetr \
    gpu_ros_yolov8 \
    gpu_ros_detection_validation \
  --event-handlers console_direct+

colcon test-result --all --verbose
```

Source-level checks that do not need a ROS installation or GPU are also
available:

```bash
git diff --check

uv run --with pytest --no-project python -m pytest -q \
  migrated_packages/gpu_ros_detection_validation/test/test_capture_runner.py \
  migrated_packages/gpu_ros_detection_validation/test/test_managed_topology.py

uv run --isolated --no-project python -B \
  tools/test_open_source_namespace_boundaries.py
```

Real-model, provider-placement, and transport checks require the matching GPU
runtime and external assets. Follow the runbook for the lane being tested.

## Benchmarks and results

Benchmark scripts cover the standard, staged-control, and direct managed
graphs. Raw bags, JSON, profiles, traces, and logs must be written outside the
source tree through `OVG_RESULTS_ROOT`.

- [Experiment index](docs/experiments/README.md)
- [AMD standard path runbook](docs/experiments/phase2a-amd.md)
- [AMD managed path runbook](docs/experiments/phase2b-managed.md)
- [NVIDIA reference runbook](docs/experiments/phase1-nvidia.md)
- [Published result summaries](docs/results/README.md)

Performance numbers from different GPU models or software stacks are reported
as separate observations. They are not presented as a single-variable backend
comparison.

## Known limitations

- RT-DETR and YOLOv8 use model-specific tensor contracts and 640 by 640 model
  inputs.
- Some RT-DETR operations may be placed on the CPU by ONNX Runtime. Provider
  placement is reported per tested model and runtime rather than enforced with
  a permanent node allowlist.
- The NVIDIA compatibility lane needs external Isaac ROS packages and source
  checkouts. It is excluded from AMD builds.
- Grounding DINO and DetectNet have not been migrated.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a change. Do not commit
models, datasets, ROS bags, profiles, runtime images, credentials, or generated
benchmark output.

Report security issues using the private contact in
[SECURITY.md](SECURITY.md), not a public issue.

## License and third-party code

Project-authored source is licensed under the
[Apache License 2.0](LICENSE). Some files were derived from file-level
Apache-2.0 NVIDIA Isaac ROS sources and retain their original copyright and
modification notices. External checkouts and model files keep their own terms.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for component-level
details. This is an independent project and is not affiliated with or endorsed
by NVIDIA or AMD.

Maintainer: Boshen Chen ([@ObeliskGate](https://github.com/ObeliskGate))
