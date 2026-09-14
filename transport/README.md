# gpu_ros_managed

Managed GPU buffers and intra-process ROS 2 transport for CUDA and HIP.

`gpu_ros_managed` is the transport component of the GPU ROS monorepo. It
provides a small C++17 ownership and synchronization layer for passing
GPU-resident data between components. It keeps allocation, readiness, and
lifetime rules independent of the GPU runtime, then supplies CUDA and HIP
backends plus optional ROS 2 and TensorBundle adapters.

The library is intended for composable ROS 2 nodes running in one process. It
does not contain an inference engine, model code, or an inter-process GPU
transport.

> [!NOTE]
> The packages are currently versioned `0.1.0`. The API is usable, but should
> be treated as pre-release while it is exercised by more applications.

## Why this exists

A raw device pointer is not enough to move GPU data safely through a graph.
The consumer also needs to know who owns the allocation, whether the producer
has finished, which device and runtime created it, and when the memory can be
reused.

This project puts those rules into a shared buffer object:

```text
producer stream                  consumer stream(s)
      |                                  |
      v                                  v
 WriteHandle  ->  DeviceBuffer  ->  ReadHandle
      |               |                  |
 records readiness    owns allocation    records completion
```

The buffer also retains any external owners attached by the producer. Its
allocation returns to the pool only after the producer and every reader have
completed. A failed or uncertain operation is kept out of the pool rather than
being reused unsafely.

## Packages

| Package | Purpose | Requires ROS 2 |
| --- | --- | --- |
| `gpu_ros_managed_core` | Device buffers, streams, handles, and fixed-size pools | No |
| `gpu_ros_managed_cuda` | CUDA allocation, stream, event, and copy backend | No |
| `gpu_ros_managed_hip` | HIP allocation, stream, event, and copy backend | No |
| `gpu_ros_managed_ros` | Intra-process publisher and subscriber wrappers | Yes |
| `gpu_ros_managed_tensor_bundle` | Managed TensorBundle type and ROS type adapter | Yes |

The core public headers do not include CUDA, HIP, ONNX Runtime, GXF, or
`rclcpp`. Applications select a backend explicitly at build time and in code.

## Requirements

Common requirements:

- Linux and a C++17 compiler
- CMake 3.22 or newer

Optional requirements depend on the packages being built:

- CUDA Toolkit for `gpu_ros_managed_cuda`
- ROCm with HIP for `gpu_ros_managed_hip`
- ROS 2 and `rclcpp` for the ROS wrappers
- `gpu_ros_tensor_bundle_msgs` for the TensorBundle adapter. The message
  package is in the same monorepo under
  [`../interfaces/gpu_ros_tensor_bundle_msgs`](../interfaces/gpu_ros_tensor_bundle_msgs).

ROS 2 Jazzy is the currently tested ROS distribution. CUDA, ROCm, and driver
compatibility follows the toolchain used to build the application; this
repository does not ship prebuilt binaries.

## Build the standalone library

The backend-neutral core has no ROS or GPU SDK dependency. From the monorepo
root, the explicit source directory is `transport`:

```bash
cd /path/to/gpu-ros
cmake -S transport -B build/transport-core \
  -DGPU_ROS_MANAGED_BUILD_CUDA=OFF \
  -DGPU_ROS_MANAGED_BUILD_HIP=OFF
cmake --build build/transport-core --parallel
ctest --test-dir build/transport-core --output-on-failure
```

The same standalone tree can be configured from its own directory. In that
case the working directory is `transport/` and the source option is `.`:

```bash
cd /path/to/gpu-ros/transport
cmake -S . -B build/core \
  -DGPU_ROS_MANAGED_BUILD_CUDA=OFF \
  -DGPU_ROS_MANAGED_BUILD_HIP=OFF
cmake --build build/core --parallel
ctest --test-dir build/core --output-on-failure
```

Enable one or both backends when the matching SDK is installed. From the
monorepo root:

```bash
cd /path/to/gpu-ros
cmake -S transport -B build/transport-hip \
  -DGPU_ROS_MANAGED_BUILD_CUDA=OFF \
  -DGPU_ROS_MANAGED_BUILD_HIP=ON \
  -DGPU_ROS_MANAGED_HIP_REQUIRE_SDK=ON
cmake --build build/transport-hip --parallel
ctest --test-dir build/transport-hip --output-on-failure

cmake -S transport -B build/transport-cuda \
  -DGPU_ROS_MANAGED_BUILD_CUDA=ON \
  -DGPU_ROS_MANAGED_BUILD_HIP=OFF
cmake --build build/transport-cuda --parallel
ctest --test-dir build/transport-cuda --output-on-failure
```

For a focused standalone run, invoke CTest from the matching build directory:

```bash
ctest --test-dir build/transport-core -R gpu_ros_managed_core_test --output-on-failure
ctest --test-dir build/transport-cuda -R gpu_ros_managed_cuda_stream_test --output-on-failure
ctest --test-dir build/transport-hip -R 'gpu_ros_managed_(hip_copy|hip_dso_identity)_test' \
  --output-on-failure
```

The CUDA and HIP device tests use CTest skip code 77 when no compatible device
is available. A missing SDK can therefore leave the corresponding backend out
of the build; it does not make standalone configure fail unless the HIP
`GPU_ROS_MANAGED_HIP_REQUIRE_SDK` option above is enabled.


## Build in a ROS 2 workspace

The TensorBundle message package is part of this monorepo under `interfaces/`.
From the monorepo root, build the transport and interface roots together:

```bash
cd /path/to/gpu-ros
source /opt/ros/jazzy/setup.bash

rosdep install \
  --from-paths \
    transport \
    interfaces \
  --ignore-src \
  --rosdistro jazzy \
  -r -y

colcon build \
  --symlink-install \
  --base-paths \
    transport \
    interfaces \
  --packages-select \
    gpu_ros_managed_core \
    gpu_ros_managed_hip \
    gpu_ros_managed_ros \
    gpu_ros_tensor_bundle_msgs \
    gpu_ros_managed_tensor_bundle
source install/setup.bash
```

When working from `transport/`, the corresponding interface root is
`../interfaces`:

```bash
cd /path/to/gpu-ros/transport
rosdep install --from-paths . ../interfaces --ignore-src --rosdistro jazzy -r -y
colcon build --symlink-install --base-paths . ../interfaces \
  --packages-select \
    gpu_ros_managed_core \
    gpu_ros_managed_hip \
    gpu_ros_managed_ros \
    gpu_ros_tensor_bundle_msgs \
    gpu_ros_managed_tensor_bundle
source install/setup.bash
```

Replace `gpu_ros_managed_hip` with `gpu_ros_managed_cuda` for a CUDA workspace.
To build only the ROS wrappers without TensorBundle support, omit the two
TensorBundle packages.

## Basic buffer lifecycle

The backend factories return the same core types. This HIP example submits a
write on one stream and reads the result on another:

```cpp
#include <cstdint>

#include <hip/hip_runtime_api.h>

#include "gpu_ros_managed_hip/hip_backend.hpp"

auto producer = gpu_ros_managed::hip::make_stream(0);
auto consumer = gpu_ros_managed::hip::make_stream(0);
auto buffer = gpu_ros_managed::hip::allocate(4096, 0);

{
  auto writer = buffer->get_write_handle(producer);
  hipMemsetAsync(writer.data(), 0, writer.size(), producer.get());
  writer.finalize();
}

uint8_t first_byte = 0;
{
  auto reader = buffer->get_read_handle(consumer);
  hipMemcpyAsync(
    &first_byte, reader.data(), 1, hipMemcpyDeviceToHost, consumer.get());
  reader.finish();
}
hipStreamSynchronize(consumer.get());
```

`finalize()` records producer completion on the producer stream. A read handle
waits for that readiness on its consumer stream and records reader completion
when `finish()` is called or the handle is destroyed.

Destroying an unfinished `WriteHandle` or `SynchronizedWriteHandle` marks the
buffer failed and non-recyclable. It never makes an incomplete write visible
as ready data.

Use `copy_from_host_blocking()` or a `SynchronizedWriteHandle` only when the
producer operation has completed synchronously. If work may have been
submitted but completion cannot be established, call `fail()`. Use `cancel()`
only before submitting work.

Fixed pools are available through `make_fixed_device_pool()` in each backend.
Pool acquisition is bounded and supports a timeout; it never falls back to an
untracked allocation.

See the
[`CUDA test`](gpu_ros_managed_cuda/test/cuda_stream_test.cpp),
[`HIP test`](gpu_ros_managed_hip/test/hip_copy_test.cpp), and
[`core tests`](gpu_ros_managed_core/test/core_test.cpp) for complete examples.

## ROS 2 transport

`ManagedPublisher<T>` and `ManagedSubscriber<ViewT>` enable ROS 2
intra-process communication and keep the managed message owner alive for the
duration of the callback. All participating nodes must use intra-process
communication and run in the same process to retain device-backed storage.

`gpu_ros_managed_tensor_bundle` adds:

- `ManagedTensor` for host or device storage
- `ManagedTensorBundle` and its builder
- `ManagedTensorBundleView` for subscriber callbacks
- a ROS type adapter for `gpu_ros_tensor_bundle_msgs/msg/TensorBundle`

Crossing the ROS serialization boundary materializes device tensors in host
memory with a blocking copy. The adapter is therefore useful for compatibility
and inspection, but it is not an inter-process zero-copy protocol.

## Safety contract

Callers are responsible for the following rules:

1. The buffer and every stream used with it must have the same backend and
   device ID.
2. A writer must report exactly one terminal state with `finalize()`,
   `finalize_synchronously()`, `fail()`, or `cancel()`.
3. `finalize()` is valid only after producer work has been submitted to the
   handle's stream.
4. A consumer must keep its read handle alive until all work using the pointer
   has been submitted.
5. External allocations and borrowed streams must carry an owner that remains
   valid for their full lifetime.

The library enforces state and device checks, but it cannot determine whether
an arbitrary external kernel or inference runtime used the correct stream.

## Project status and limitations

- Intra-process managed transport is implemented for CUDA and HIP;
  cross-process device-memory transport is not.
- Fixed pools use a configured capacity and do not grow automatically.
- Serialization of a device-backed TensorBundle performs a device-to-host
  copy.
- ABI compatibility across arbitrary ROS, CUDA, ROCm, and driver versions is
  not guaranteed.

The application packages in this monorepo use the HIP backend and ONNX Runtime
MIGraphX for their AMD managed path. See the root README and
`../docs/experiments/phase2b-managed.md` for the application runbook.

## Contributing

Changes should include tests for ownership, readiness, error, and shutdown
behavior. For standalone changes, configure and build the relevant tree, then
run its tests from that build directory:

```bash
cd /path/to/gpu-ros/transport
ctest --test-dir build/core --output-on-failure
ctest --test-dir build/hip --output-on-failure
ctest --test-dir build/cuda --output-on-failure
```

Only run the backend command for a tree configured with that backend. Those
tests need the matching SDK, and the device tests skip with code 77 when no
compatible device is available.

For ROS package changes, source the ROS distribution and the workspace overlay
from the monorepo root, then run only the affected packages:

```bash
cd /path/to/gpu-ros
source /opt/ros/jazzy/setup.bash
source install/setup.bash
colcon test --base-paths transport interfaces --packages-select \
  gpu_ros_managed_core gpu_ros_managed_ros gpu_ros_managed_tensor_bundle
colcon test-result --verbose
```

The TensorBundle test uses the `gpu_ros_tensor_bundle_msgs` package under the
same monorepo's `interfaces/` root. `colcon test` runs from the monorepo root
after a successful `colcon build`; it does not require a merged install.
Keep generated build trees, profiles, traces, and GPU artifacts out of commits.

## License and provenance

The repository is licensed under the [Apache License 2.0](LICENSE).

Some files are ports of Apache-2.0 source files from NVIDIA Isaac ROS NITROS
4.5. Their original copyright headers and local modification notices are kept.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the
[port map](docs/nitros-4.5-port-map.md) for the exact file-level provenance.

This is an independent project and is not affiliated with or endorsed by
NVIDIA.

Maintainer: Boshen Chen ([@ObeliskGate](https://github.com/ObeliskGate))
