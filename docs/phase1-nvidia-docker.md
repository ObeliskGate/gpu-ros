# Phase 1 NVIDIA Docker Environment

This helper provides the current NVIDIA Isaac ROS 4.5 environment. It wraps the
existing root `Dockerfile` and `docker-compose.yaml`; it does not replace them
with the AMD image or rebuild ONNX Runtime from source. The checked-in Phase 1
matrix was collected on 4.4 and remains historical unless it is rerun on 4.5.

Pinned environment:

| Dependency | Value |
|------------|-------|
| Isaac ROS release | 4.5 |
| Isaac ROS base image | `isaac_ros_89df02a734965ed64c227ef531c09d65-amd64` |
| ONNX Runtime | 1.23.1 |
| ORT runtime library | Triton CUDA-matched `libonnxruntime.so` |
| ROS distribution | Jazzy |

## One-command Setup

```bash
./docker/phase1-nvidia.sh bootstrap
```

Bootstrap checks the NVIDIA GPU, validates the image and ORT pins, builds and
starts the existing `dev` service, and verifies the benchmark packages. It does
not build the migrated packages or run a benchmark automatically.

## Commands

```bash
./docker/phase1-nvidia.sh shell
./docker/phase1-nvidia.sh verify
./docker/phase1-nvidia.sh colcon
./docker/phase1-nvidia.sh stop
./docker/phase1-nvidia.sh down
```

Use `colcon` only when testing the migrated Phase 1 implementations. The
official TensorRT + NITROS baseline is available directly from the benchmark
packages installed in the image.

The helper builds all migrated C++ packages with `CMAKE_BUILD_TYPE=Release`.
Its `colcon` and `verify` commands reject benchmark workspaces whose relevant
`CMakeCache.txt` files do not record a Release build.

## Configuration C Device Boundary

Configuration C keeps tensor payloads resident on the CUDA device across the
NITROS/ORT boundary. The NITROS adapter borrows the incoming device pointer,
the shared inference core binds it with ORT CUDA I/O Binding, and the outgoing
NITROS tensor wraps the ORT-owned CUDA allocation with a release callback.
Only tensor metadata is copied by the adapter; it performs no D2H or H2D copy.

Configurations C and D and the Phase 2a AMD target share the same
`OnnxInferenceNode` and inference core. Runtime transport selection chooses the
NITROS or standard adapter; provider selection chooses CUDA or MIGraphX. The
core contains no NITROS/GXF or model-specific types, and the NITROS adapter is
omitted entirely from AMD builds.

This transport guarantee is separate from ORT operator placement. CPU-preferred
shape and bookkeeping operators are allowed. A separate, short launch may use
the node's `ort_profile_prefix` parameter to confirm that the requested GPU EP
executes model kernels and the graph has not fallen back entirely to CPU.
Benchmark graphs never enable ORT profiling.

RT-DETR has FP32 TensorRT A/B graphs for the fair A/C and B/D backend
comparisons. YOLOv8 A/B use TensorRT FP16 while C/D use ORT FP32, so YOLOv8
results isolate transport only within A/B and within C/D; its A/C delta is not
reported as a pure backend comparison.

Results collected before the device-resident I/O Binding and Release-build
checks should be retained as historical data and labeled
`host-staged/build-type-unknown`, not overwritten by the corrected matrix.

## Benchmark And Audit Discipline

Formal A/B/C/D benchmarks run without ORT profiling or bridge timing. Provider
placement and bridge timing are separate diagnostic runs because both add
instrumentation to the measured process. To time the standard-to-NITROS bridge,
set `enable_timing:=true` on `TensorListBridgeNode` in a short diagnostic launch;
the default is `false`.

Run the matrix scripts from the workspace root after `colcon` completes:

```bash
launch_test src/amd_ros_object_detection/migrated_packages/benchmarks/isaac_ros_rtdetr_config_a_fp32_graph.py
launch_test src/amd_ros_object_detection/migrated_packages/benchmarks/isaac_ros_rtdetr_config_b_fp32_graph.py
launch_test src/amd_ros_object_detection/migrated_packages/benchmarks/isaac_ros_rtdetr_config_c_graph.py
launch_test src/amd_ros_object_detection/migrated_packages/benchmarks/isaac_ros_rtdetr_config_d_graph.py
```

```bash
launch_test src/amd_ros_object_detection/migrated_packages/benchmarks/isaac_ros_yolov8_config_a_graph.py
launch_test src/amd_ros_object_detection/migrated_packages/benchmarks/isaac_ros_yolov8_config_b_graph.py
launch_test src/amd_ros_object_detection/migrated_packages/benchmarks/isaac_ros_yolov8_config_c_graph.py
launch_test src/amd_ros_object_detection/migrated_packages/benchmarks/isaac_ros_yolov8_config_d_graph.py
```

See [`phase1-results.md`](phase1-results.md) for the completed matrix and its
interpretation. In particular, the graph labels describe complete pipeline
configurations; they are not guaranteed to be a purely additive factorial
experiment because the TensorRT and ORT integrations place copies and
synchronization in different stages.
