# Phase 1 NVIDIA Docker Environment

This helper preserves the NVIDIA environment used for the Phase 1 matrix. It
wraps the existing root `Dockerfile` and `docker-compose.yaml`; it does not
replace them with the AMD image or rebuild ONNX Runtime from source.

Pinned environment:

| Dependency | Value |
|------------|-------|
| Isaac ROS base image | `isaac_ros_28556f8bc78a98822bd08b2d7c6fcf9b-amd64` |
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
