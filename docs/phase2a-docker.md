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
- A MIGraphX-enabled ONNX Runtime installation on the host, mounted into the container at `/opt/onnxruntime`.

If `/dev/kfd` is not present on the host, the `amd` profile will not run the GPU path. On WSL, verify device exposure first; otherwise use an AMD Linux server.

## ONNX Runtime Layout

The compose profile mounts `ONNXRUNTIME_ROOT_HOST` to `/opt/onnxruntime`. The mounted directory must contain:

```bash
/opt/onnxruntime/include/onnxruntime_cxx_api.h
/opt/onnxruntime/lib/libonnxruntime.so
```

The `libonnxruntime.so` must be built with MIGraphX EP support. If it is not, `execution_provider:=migraphx` should fail during session creation instead of falling back to CPU.

## Isaac ROS TensorList Interface

Phase 2a still uses the official message package:

```bash
isaac_ros_tensor_list_interfaces/msg/TensorList
```

The Dockerfile installs `ros-jazzy-isaac-ros-tensor-list-interfaces` only if that package is available from configured apt sources. If your AMD base image does not have the Isaac ROS apt source, provide the official `isaac_ros_tensor_list_interfaces` package in the workspace before building `isaac_ros_onnx_inference` and `isaac_ros_rtdetr_std`.

Do not add a custom TensorList message package in this repository.

## Configure

Start from the example env file:

```bash
cp .env.phase2a-amd.example .env.phase2a-amd
```

Edit at least:

```bash
AMD_BASE_IMAGE=rocm/dev-ubuntu-24.04:7.2.2-complete
ONNXRUNTIME_ROOT_HOST=/absolute/path/to/migraphx-enabled/onnxruntime
```

`INSTALL_BENCHMARK_DEPS=1` is optional and should only be used when `ros-jazzy-ros2-benchmark` is available from apt.

## Build And Start

```bash
docker compose --env-file .env.phase2a-amd -f docker-compose.phase2a-amd.yaml build amd
docker compose --env-file .env.phase2a-amd -f docker-compose.phase2a-amd.yaml up -d amd
docker compose --env-file .env.phase2a-amd -f docker-compose.phase2a-amd.yaml exec amd bash
```

Inside the container:

```bash
rocminfo | head
test -f /opt/onnxruntime/include/onnxruntime_cxx_api.h
test -f /opt/onnxruntime/lib/libonnxruntime.so
```

## Build Phase 2a Packages

Inside the container:

```bash
colcon build \
  --packages-select isaac_ros_onnx_inference isaac_ros_rtdetr_std
source install/setup.bash
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
# Stop only the AMD service
docker compose --env-file .env.phase2a-amd -f docker-compose.phase2a-amd.yaml stop amd

# Rebuild after Dockerfile changes
docker compose --env-file .env.phase2a-amd -f docker-compose.phase2a-amd.yaml build amd

# Open a second shell
docker compose --env-file .env.phase2a-amd -f docker-compose.phase2a-amd.yaml exec amd bash
```
