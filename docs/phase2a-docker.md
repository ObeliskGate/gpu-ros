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
| Isaac ROS TensorList source | v4.4-0 | Matches the Phase 0/1 Isaac ROS release |
| ros2_benchmark | v4.4-0 | Matches the Phase 0/1 benchmark controller and report format |

The Dockerfile builds ORT with MIGraphX and `ros2_benchmark` in separate builder
stages. Only their install artifacts are copied to the final ROS image.
`ros2_benchmark` uses generic ROS 2 playback and monitor nodes for this graph;
the NVIDIA-only `isaac_ros_benchmark` and its NITROS playback plug-in are not
installed. A small repository-owned patch removes the upstream
`isaac_ros_common` dependency that is used only to generate version metadata.

## Isaac ROS TensorList Interface

Phase 2a still uses the official message package:

```bash
isaac_ros_tensor_list_interfaces/msg/TensorList
```

The bootstrap script checks out the official `isaac_ros_common` v4.4 release
under the ignored `third_party/` directory and exposes only the TensorList
interface package to the Phase 2a colcon build. It applies a repository-owned
build patch that removes the interface package's version-metadata dependency on
the top-level `isaac_ros_common` package. The message definitions are unchanged;
the CUDA-only top-level package is not discovered or built on AMD.

Do not add a custom TensorList message package in this repository.

## One-command Environment Setup

On a new AMD development machine:

```bash
./docker/phase2a-amd.sh bootstrap
```

This command:

1. Checks `/dev/kfd`, `/dev/dri`, Docker, and Docker Compose.
2. Fetches the official TensorList interface at the pinned Isaac ROS release.
3. Builds ORT 1.23.1 with MIGraphX inside the ROCm 7.1.1 image.
4. Starts the AMD container.

ROS package builds and tests are intentionally run interactively after entering
the container, so individual commands and CMake options are easy to change
while debugging.

The first run builds ONNX Runtime from source and is slow. Docker caches that
stage for subsequent runs. Colcon `build`, `install`, and `log` directories use
named volumes, so the bind-mounted repository does not receive root-owned build
artifacts.

## Optional Configuration

Defaults work without an env file. To override them:

```bash
cp .env.phase2a-amd.example .env.phase2a-amd
```

`ORT_BUILD_JOBS` limits the ORT source-build parallelism. The bootstrap script
detects `AMD_GPU_TARGETS` from `rocminfo`; set it in the env file only when
cross-building or overriding detection.

## Enter The Container

```bash
./docker/phase2a-amd.sh shell
```

The shell command sources the base ROS installation and any existing Phase 2a
workspace install before opening the interactive prompt. This is required
because processes started by `docker compose exec` do not run the image
entrypoint again.

## Build Phase 2a Packages

Inside the container:

```bash
colcon build
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
-DBUILD_MIGRAPHX_POL_TEST=ON
-DBUILD_TESTING=ON
```

## Run The Target Launch

```bash
ros2 launch isaac_ros_rtdetr_std rtdetr_ort_std_image.launch.py \
  model_file_path:=/workspaces/isaac_ros-dev/assets/models/synthetica_detr_v1.0.0_onnx/sdetr_grasp.onnx \
  execution_provider:=migraphx \
  image_topic:=/camera_1/color/image_raw
```

For bag replay, run `ros2 bag play` in another shell and remap or set `image_topic` to the bag image topic.

## Audit ONNX Runtime Provider Placement

Normal inference allows ONNX Runtime to place unsupported or CPU-preferred graph
nodes on its default CPU EP. This matches the Phase 1 NVIDIA behavior. Requesting
an execution provider that was not built remains an error.

Enable an ORT profile for a short representative run:

```bash
mkdir -p /tmp/ort_profiles
ros2 launch isaac_ros_rtdetr_std rtdetr_ort_std_image.launch.py model_file_path:=/workspaces/isaac_ros-dev/assets/models/synthetica_detr_v1.0.0_onnx/sdetr_grasp.onnx execution_provider:=migraphx image_topic:=/camera_1/color/image_raw ort_profile_prefix:=/tmp/ort_profiles/rtdetr_migraphx
```

After at least one inference, stop the launch cleanly. Then inspect which
providers actually executed graph nodes:

```bash
ros2 run isaac_ros_onnx_inference summarize_ort_profile.py /tmp/ort_profiles/rtdetr_migraphx_*.json --expected-provider MIGraphXExecutionProvider --output-json /tmp/ort_profiles/provider_report_migraphx.json
```

To make CPU assignment fail an automated audit, add
`--require-no-cpu-nodes`. Do not enable profiling for final benchmark numbers;
the per-node event collection adds overhead and can produce a large JSON file.

## Run Port Tests

After building and sourcing the workspace inside the container:

```bash
colcon test --event-handlers console_direct+
colcon test-result --verbose
```

The checked-in colcon defaults force `CMAKE_BUILD_TYPE=Release`. From the host,
`./docker/phase2a-amd.sh verify` confirms that the ONNX Runtime and RT-DETR
package caches are Release builds before collecting benchmark results.

The container's colcon defaults restrict build discovery to the three Phase 2a
packages and test discovery to the two migrated packages. The tests check the
ONNX Runtime provider-selection core and the ported proof-of-life graph. A
deterministic ONNX test model is executed by MIGraphX between the standard
TensorList preprocessor and decoder.

To rebuild and rerun only the proof-of-life package while debugging:

```bash
colcon build --packages-select isaac_ros_rtdetr_std --cmake-clean-cache
source install/setup.bash
colcon test --packages-select isaac_ros_rtdetr_std \
  --event-handlers console_direct+ \
  --ctest-args -R isaac_ros_std_rtdetr_pol_test --output-on-failure
colcon test-result --verbose
```

The proof-of-life model validates graph wiring and execution-provider behavior,
not RT-DETR numeric accuracy. After it passes, use the target launch above with
the FP32 `sdetr_grasp.onnx` model for model-level validation.

## Benchmark

The benchmark script is:

```bash
migrated_packages/benchmarks/isaac_ros_rtdetr_phase2a_amd_graph.py
```

The AMD image includes the official `ros2_benchmark` v4.4 install. Verify it
after entering the container:

```bash
ros2 pkg prefix ros2_benchmark
```

Run the benchmark from the repository root with ORT profiling disabled:

```bash
unset ORT_PROFILE_PREFIX
launch_test migrated_packages/benchmarks/isaac_ros_rtdetr_phase2a_amd_graph.py
```

The script sends one buffered frame and waits for the first detection before it
enters the measured trial and throughput search. This excludes lazy MIGraphX
compilation from the measured runs. A warm-up from a different launch process
does not replace this step by itself. The container also sets
`ORT_MIGRAPHX_MODEL_CACHE_PATH` to the persistent assets volume. The first
session compiles and saves an `.mxr` model; later containers on the same machine
can load it instead of compiling the graph again. The default warm-up timeout is
900 seconds; override it with `MIGRAPHX_WARMUP_TIMEOUT_SEC` if needed.

Clear `/workspaces/isaac_ros-dev/assets/cache/migraphx` after changing the ONNX
model, GPU architecture, ROCm/MIGraphX version, ORT version, or the MIGraphX EP
patches. Compiled `.mxr` files should not be moved between different hardware or
runtime combinations.

By default, the framework writes a timestamped `r2b-log-*.json` file so repeated
runs cannot append into the same JSON. Set `R2B_RESULT_FILE` to request a
specific filename:

```bash
R2B_RESULT_FILE=phase2a-rtdetr-amd-migraphx-mi300x.json launch_test migrated_packages/benchmarks/isaac_ros_rtdetr_phase2a_amd_graph.py
```

The generic resource profiler records CPU utilization on AMD. It does not
currently report AMD GPU utilization because upstream v4.4 only supports
`nvidia-smi`/`gpustat` for GPU metrics; collect AMD GPU utilization separately
rather than enabling ORT profiling during the performance run.

Default result directory:

```bash
migrated_packages/benchmark_results/
```

## Common Commands

```bash
./docker/phase2a-amd.sh build
./docker/phase2a-amd.sh up
./docker/phase2a-amd.sh shell
./docker/phase2a-amd.sh stop
./docker/phase2a-amd.sh down
```
