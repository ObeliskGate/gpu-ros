# Phase 2a AMD Docker Environment

This Docker profile is for the Phase 2a target path only:

```
Image -> RtDetrImageEncoderNode -> RtDetrPreprocessorNode
      -> OnnxInferenceNode(transport=std, execution_provider=migraphx)
      -> RtDetrDecoderNode -> Detection2DArray
```

The existing `dev` compose service remains the NVIDIA Isaac ROS baseline environment. Use the `amd` profile for AMD work.

## Host Requirements

- AMD GPU compute device visible to containers as `/dev/kfd` and `/dev/dri`.
- Docker Engine with permission to pass those devices into the container.
- Git and network access for the first image build.

If `/dev/kfd` is not present on the host, the `amd` profile will not run the GPU path. On WSL, verify device exposure first; otherwise use an AMD Linux server.

## Pinned Runtime Versions

The image builds its own C++ ONNX Runtime installation. No host ORT installation
or `ONNXRUNTIME_ROOT_HOST` mount is required.

| Dependency | Default | Reason |
|------------|---------|--------|
| ONNX Runtime | 1.23.1 | Matches the final Phase 1 NVIDIA environment |
| ROCm | 7.1.1 | AMD-supported ROCm pairing for ORT 1.23.1 |
| Isaac ROS common | v4.4-0 | Matches the Phase 0/1 Isaac ROS release |

The Dockerfile builds ORT with MIGraphX in a builder stage. Only its headers and
runtime libraries are copied to the final ROS image.

## Isaac ROS TensorList Interface

Phase 2a still uses the official message package:

```bash
isaac_ros_tensor_list_interfaces/msg/TensorList
```

The bootstrap script checks out the official `isaac_ros_common` v4.4 release
under the ignored `third_party/` directory and builds only the TensorList
interface package needed by Phase 2a.

Do not add a custom TensorList message package in this repository.

## One-command Setup

On a new AMD development machine:

```bash
./docker/phase2a-amd.sh bootstrap
```

This command:

1. Checks `/dev/kfd`, `/dev/dri`, Docker, and Docker Compose.
2. Fetches the official TensorList interface at the pinned Isaac ROS release.
3. Builds ORT 1.23.1 with MIGraphX inside the ROCm 7.1.1 image.
4. Starts the AMD container.
5. Builds and verifies the Phase 2a ROS packages.

The first run builds ONNX Runtime from source and is slow. Docker caches that
stage for subsequent runs. Colcon `build`, `install`, and `log` directories use
named volumes, so the bind-mounted repository does not receive root-owned build
artifacts.

## Optional Configuration

Defaults work without an env file. To override them:

```bash
cp .env.phase2a-amd.example .env.phase2a-amd
```

`ORT_BUILD_JOBS` limits the ORT source-build parallelism. Keep
`INSTALL_BENCHMARK_DEPS=0` unless `ros-jazzy-ros2-benchmark` is available.
The bootstrap script detects `AMD_GPU_TARGETS` from `rocminfo`; set it in the
env file only when cross-building or overriding detection.

## Enter The Container

```bash
./docker/phase2a-amd.sh shell
```

## Build Phase 2a Packages

Bootstrap builds the workspace automatically. To rebuild it later:

```bash
./docker/phase2a-amd.sh colcon
```

The compose service sets:

```bash
COLCON_DEFAULTS_FILE=/workspaces/amd_ros_object_detection/docker/colcon-defaults-phase2a-amd.yaml
```

That file passes:

```bash
-DBUILD_NITROS_TRANSPORT=OFF
-DORT_ENABLE_CUDA=OFF
-DORT_ENABLE_ROCM=OFF
-DORT_ENABLE_MIGRAPHX=ON
-DONNXRUNTIME_ROOT=/opt/onnxruntime
```

## Run The Target Launch

```bash
ros2 launch isaac_ros_rtdetr_std rtdetr_ort_std_image.launch.py \
  model_file_path:=/workspaces/isaac_ros-dev/assets/models/synthetica_detr_v1.0.0_onnx/sdetr_grasp.onnx \
  execution_provider:=migraphx \
  image_topic:=/camera_1/color/image_raw
```

For bag replay, run `ros2 bag play` in another shell and remap or set `image_topic` to the bag image topic.

## Benchmark

The benchmark script is:

```bash
migrated_packages/benchmarks/isaac_ros_rtdetr_phase2a_amd_graph.py
```

It requires `ros2_benchmark`. Build the Docker image with `INSTALL_BENCHMARK_DEPS=1` only if that apt package is available, or provide `ros2_benchmark` in the workspace.

Expected result path:

```bash
migrated_packages/benchmark_results/phase2a-rtdetr-amd-migraphx.json
```

## Common Commands

```bash
./docker/phase2a-amd.sh build
./docker/phase2a-amd.sh up
./docker/phase2a-amd.sh shell
./docker/phase2a-amd.sh stop
./docker/phase2a-amd.sh down
```
