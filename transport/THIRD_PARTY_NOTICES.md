# Third-party notices

This file describes the source-level boundary for the code in this repository.
It is project release documentation; each dependency remains under its own
license.

## NVIDIA Isaac ROS NITROS

The following local files are derived from file-level Apache-2.0 source in
NVIDIA Isaac ROS NITROS `v4.5-0`, commit
`82310fce298d3d9db26945a3a988d5c471d14973`:

- `gpu_ros_managed_core/include/gpu_ros_managed_core/buffer.hpp`
- `gpu_ros_managed_core/src/buffer.cpp`
- `gpu_ros_managed_core/include/gpu_ros_managed_core/fixed_device_memory_pool.hpp`
- `gpu_ros_managed_core/src/fixed_device_memory_pool.cpp`
- `gpu_ros_managed_tensor_list/include/gpu_ros_managed_tensor_list/tensor_list.hpp`
- `gpu_ros_managed_tensor_list/src/tensor_list.cpp`
- `gpu_ros_managed_ros/include/gpu_ros_managed_ros/managed_pub_sub.hpp`

The exact upstream file attribution is retained in each derived file, and the
files contain a notice describing the local backend-neutral or GXF-free
modification. NITROS repository/package metadata is not treated as a blanket
Apache license for all of NITROS.

## Project-original code

The backend interface and implementations, host-buffer utilities, ROS
TypeAdapter, tests, and build files are maintained as project-original code
unless a source file says otherwise. They are distributed under the Apache
License 2.0 in `LICENSE`.

## ROS 2 and system dependencies

ROS 2 packages, CUDA, HIP/ROCm, and their headers and libraries are external
build/runtime dependencies. This repository does not vendor those SDKs or
relicense them. Consumers must follow the license terms supplied by their
chosen ROS distribution, CUDA toolkit, and ROCm installation.

## Models, datasets, and generated output

No model, dataset, bag, trace, profile, log, SIF image, or benchmark result is
required or distributed by this source tree. An application may use external
assets subject to the asset provider's terms; source-code license and model or
dataset license are separate questions.
