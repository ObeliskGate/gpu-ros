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
