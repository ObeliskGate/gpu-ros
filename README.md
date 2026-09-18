# GPU ROS

GPU ROS contains two library collections: managed GPU-buffer transport and
ROS 2 object detection with ONNX Runtime. They share one Git checkout and keep
all ROS package names, C++ namespaces, public includes, and CMake targets under
`gpu_ros_*`.

The packages are versioned `0.1.0` and remain pre-release. This repository is
an experimental source release, not a supported runtime distribution. It ships
no prebuilt binaries, runtime images, models, or datasets. Grounding DINO and
DetectNet appear in historical NVIDIA records; they are not packages in these
collections.

## Collections

| Collection | Contents |
| --- | --- |
| [`gpu_ros_managed/`](gpu_ros_managed/README.md) | Seven packages: backend-neutral C++17 buffers, CUDA/HIP backends, ROS wrappers, TensorBundle messages and adapters, and optional NVIDIA TensorList conversion. |
| [`gpu_ros_object_detection/`](gpu_ros_object_detection/README.md) | Six packages: shared detection utilities, ONNX Runtime integration, RT-DETR, YOLOv8, validation, and NVIDIA reference compositions. Detection benchmarks, runtime recipes, tools, configuration, and external-source manifests live here too. |

Neither collection is an aggregate ROS package. The managed collection's
standalone CMake project builds the core and optional CUDA/HIP backends only.
ROS, message, and compatibility packages are built individually with colcon.
The detection `benchmarks/` directory contains graph definitions, not a ROS
package.

`gpu_ros_nvidia_tensor_bundle_compat` owns generic TensorList conversions and
its boundary launch. It is the only project package with a direct dependency
on NVIDIA TensorList interfaces. Complete NVIDIA/Isaac detection compositions
belong to `gpu_ros_nvidia_reference`, including the five reference launch files
listed in the [detection library](gpu_ros_object_detection/README.md).
AMD profiles exclude both NVIDIA-specific packages. The core RT-DETR and
YOLOv8 packages have no Isaac runtime dependencies.

## Transport boundary

The standard path uses ordinary ROS TensorBundle messages:

```text
Image -> image encoder -> TensorBundle -> ORT -> TensorBundle -> decoder
                                                               |
                                                               v
                                                        Detection2DArray
```

The managed path keeps device buffers in one component process:

```text
Image -> managed preprocessing -> ORT/provider -> managed decoder
                                                       |
                                                       v
                                                Detection2DArray on CPU
```

Managed transport defines ownership, readiness, and stream/device lifetimes.
It is not an inference engine or an inter-process GPU transport. ROS
serialization materializes device tensors in host memory. A verified
inference boundary does not imply that the complete graph is copy-free.

## Build and experiment entry points

Clone one repository; no sibling transport checkout is needed:

```bash
git clone https://github.com/gpu-ros/gpu-ros.git
cd gpu-ros
```

- For a library build without ROS or a GPU SDK, use the
  [standalone managed build](gpu_ros_managed/README.md#build-the-standalone-library).
- For AMD detection, start with the
  [AMD runtime contract](skills/gpu-ros-experiments/references/amd-runtime-contract.md).
  Docker is the default. A Slurm site without Docker uses the explicit
  Apptainer branch inside an authorized compute allocation.
- For NVIDIA reference graphs, use the
  [NVIDIA method](skills/gpu-ros-experiments/references/phase2b-nvidia.md).
- For capture, comparison, and copy evidence, use
  [detection validation](skills/gpu-ros-experiments/references/detection-validation.md).

Methods live under [`skills/gpu-ros-experiments/`](skills/gpu-ros-experiments/).
Users can read the method files directly; no skill installation is required.
Results and their provenance live under [`docs/results/`](docs/results/README.md).
A fixed-input capture is not a throughput measurement. `launch_test` on a
benchmark graph runs a performance experiment; it is not a smoke check.
The current layout campaign used Apptainer on AMD and an existing NVIDIA Docker
image. AMD Docker GPU execution was not tested, and neither lane included a
fresh dependency-image build.

## Runtime and asset boundaries

The AMD compatibility target is ROS 2 Jazzy, Ubuntu 24.04, ROCm 7.1.1, the
actual device target, and the external ONNX Runtime 1.23.1/MIGraphX install
selected by the [ORT lock](gpu_ros_object_detection/config/onnxruntime.lock).
The checkout and workspace remain `/workspaces/gpu-ros`. The detection
subdirectory is not `OVG_WORKSPACE_ROOT`. Image `/opt/onnxruntime` content is
not a formal runtime fallback.

NVIDIA uses the pinned Isaac ROS 4.5-compatible runtime and
[external-source manifest](gpu_ros_object_detection/external/nvidia-isaac-ros.repos).
Its outer workspace remains `/workspaces/isaac_ros-dev`; this repository is
mounted at `src/gpu-ros`, with external checkouts under `src/nvidia_external`.

Reuse verified model and runtime bytes. Keep external checkouts, ORT installs,
models, checkpoints, datasets, bags, caches, traces, images, build/install/log
state, and results outside Git. A digest checks identity; it does not grant a
model or dataset license. Do not download or export replacement YOLOv8 weights
as an implicit setup step.

## Development and evidence

Read [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md). Preserve
package interfaces, model contracts, provider settings, and external pins.
Run checks for the affected layer and record the actual source, runtime,
model/data identity, command, and result location. Keep command status,
component teardown, numeric correctness, copy evidence, and throughput separate.
Current acceptance and result records are linked from the
[documentation index](docs/README.md); a historical PASS does not validate a
new checkout.

Project-authored code is Apache-2.0. File-level notices and external component
terms remain authoritative. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
and the [managed notices](gpu_ros_managed/THIRD_PARTY_NOTICES.md). This project
is not affiliated with or endorsed by NVIDIA or AMD.
