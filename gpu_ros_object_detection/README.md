# gpu_ros_object_detection

ROS 2 detection packages for RT-DETRv2 and YOLOv8, with shared preprocessing
and decoding utilities, ONNX Runtime integration, and validation tools. The
collection consumes [`gpu_ros_managed`](../gpu_ros_managed/) from the same
checkout. It is not an aggregate ROS package.
The collection is versioned `0.1.0` and remains pre-release. It publishes
source and runtime recipes, not prebuilt binaries or a supported runtime image.

## Packages

| Package | Responsibility |
| --- | --- |
| `gpu_ros_detection_common` | Shared detection utilities and contracts. |
| `gpu_ros_onnx_inference` | ONNX Runtime integration, provider selection, and standard/managed inference paths. |
| `gpu_ros_rtdetr` | RT-DETR preprocessing, decoding, and vendor-neutral image launches. |
| `gpu_ros_yolov8` | YOLOv8 preprocessing, decoding, and vendor-neutral image launches. |
| `gpu_ros_detection_validation` | Fixed-input capture, bag comparison, transport audits, and the AMD benchmark matrix runner. |
| `gpu_ros_nvidia_reference` | Opt-in NVIDIA/Isaac reference compositions; no C++ library. |

Core RT-DETR and YOLOv8 package manifests have no Isaac runtime dependencies.
NVIDIA TensorList conversion components remain in the managed collection's
`gpu_ros_nvidia_tensor_bundle_compat` package. The reference package composes
those components with inference, model, and Isaac packages; it does not own a
second conversion implementation.

## NVIDIA reference launches

Use `ros2 launch gpu_ros_nvidia_reference` for these files:

| Launch file | Composition |
| --- | --- |
| `rtdetr_ort_managed.launch.py` | RT-DETR with ORT and managed TensorBundle bridges. |
| `yolov8_ort_transport.launch.py` | YOLOv8 ORT reference and managed transport variants. |
| `rtdetr_ort_nitros.launch.py` | RT-DETR ORT/NITROS reference. |
| `rtdetr_ort_std.launch.py` | RT-DETR ORT with a standard ROS boundary and Isaac components. |
| `rtdetr_trt_std.launch.py` | RT-DETR TensorRT with a standard ROS boundary and Isaac components. |

Inspect the installed arguments before selecting a model or provider:

```bash
ros2 launch gpu_ros_nvidia_reference rtdetr_ort_managed.launch.py --show-args
```

`nvidia_tensor_bundle_boundary.launch.py` belongs to the compatibility package.
The vendor-neutral `*_ort_std_image.launch.py` files belong to their model
packages. AMD managed launches belong to `gpu_ros_onnx_inference`.

## Detection facilities

| Directory | Contents |
| --- | --- |
| [`benchmarks/`](benchmarks/) | Existing graph definitions, not a ROS package. |
| [`config/`](config/) | ORT lock and model profiles. |
| [`external/`](external/) | Pinned NVIDIA source manifest; actual checkouts stay external. |
| [`docker/`](docker/) | Compose files, Dockerfiles, launchers, entrypoints, colcon defaults, and patches. |
| [`apptainer/`](apptainer/) | AMD runtime recipe. |
| [`tools/`](tools/) | Runtime/asset helpers, ORT build tooling, NVIDIA bootstrap, and RT-DETR export tooling. |

Run repository-relative commands from the monorepo root. AMD uses
`/workspaces/gpu-ros` as its workspace; NVIDIA uses `/workspaces/isaac_ros-dev`
with the repository at `src/gpu-ros`. The detection directory is neither the
monorepo identity root nor a replacement workspace root.

## Methods and results

[Experiment methods](../skills/gpu-ros-experiments/) cover standalone managed
builds, AMD Docker or explicit Apptainer execution, and NVIDIA reference work.
The [validation package](gpu_ros_detection_validation/README.md) lists the
installed command names. Reuse verified dependencies and assets; keep fresh
build/install/log state and raw results outside Git.
The current layout campaign used Apptainer on AMD and an existing NVIDIA Docker
image. AMD Docker GPU execution was not tested, and neither lane included a
fresh dependency-image build. These observations do not establish a supported
runtime release.

[Result records](../docs/results/README.md) distinguish execution, correctness,
copy closure, teardown, and performance reproduction. Running `launch_test` on
a benchmark graph measures performance; it is not an ordinary smoke check.
