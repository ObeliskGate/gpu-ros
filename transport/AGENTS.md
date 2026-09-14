# Transport agent guide

## Scope

`transport/` provides C++17 managed GPU buffers with explicit CUDA and HIP
backends, plus optional ROS 2 wrappers and a TensorBundle adapter. The core
headers are backend neutral. Do not describe this component as an inference
engine, model runtime, or inter-process GPU transport.

This directory is part of the GPU ROS monorepo. Package directory basenames and
ROS package names remain unchanged (`gpu_ros_managed_*`). The project-owned
`gpu_ros_tensor_bundle_msgs` package is in the sibling monorepo root directory
`../interfaces/`, not in another Git checkout.

## Buffer contract

`DeviceBuffer` owns the allocation. A `WriteHandle` publishes asynchronous
producer work with exactly one terminal operation: `finalize()` after work was
submitted to its stream, `fail()` when completion cannot be trusted, or
`cancel()` before any work is submitted. Destroying an unfinished writer marks
the buffer failed and non-recyclable. Use
`SynchronizedWriteHandle::finalize_synchronously()` only after the external
operation has completed synchronously. A read handle waits for producer
readiness on its consumer stream and must remain alive until all uses of its
pointer have been submitted; call `finish()` when reader work is complete.

Every buffer and stream must use the same backend and device ID. Borrowed
streams and external allocations need a live owner for their full lifetime.
Pool release waits for producer and reader completion; failed or uncertain
buffers are not recycled. See
[`gpu_ros_managed_core/buffer.hpp`](gpu_ros_managed_core/include/gpu_ros_managed_core/buffer.hpp)
and the lifecycle example in [README.md](README.md).

## Transport and build boundaries

ROS transport is intra-process only. All participating nodes must run in one
process to retain device-backed storage. Crossing ROS serialization performs a
blocking device-to-host copy; it is not cross-process zero-copy.

The standalone CMake tree builds `gpu_ros_managed_core` without ROS or a GPU
SDK. CUDA and HIP are optional subdirectories. HIP can require its SDK with
`-DGPU_ROS_MANAGED_HIP_REQUIRE_SDK=ON`; CUDA has no corresponding require
option and is omitted when its toolkit is unavailable. ROS builds use colcon.
The `gpu_ros_managed_tensor_bundle` adapter consumes the project-owned
`gpu_ros_tensor_bundle_msgs` package from `../interfaces/`.

From the monorepo root, use `cmake -S transport`; from this directory, use
`cmake -S .`. Keep build directories outside the source tree when possible:

```bash
# Working directory: the monorepo root.
cmake -S transport -B /tmp/gpu-ros-transport-core \
  -DGPU_ROS_MANAGED_BUILD_TESTING=ON \
  -DGPU_ROS_MANAGED_BUILD_CUDA=OFF \
  -DGPU_ROS_MANAGED_BUILD_HIP=OFF
cmake --build /tmp/gpu-ros-transport-core
ctest --test-dir /tmp/gpu-ros-transport-core --output-on-failure

# Working directory: transport/.
cmake -S . -B /tmp/gpu-ros-transport-core \
  -DGPU_ROS_MANAGED_BUILD_TESTING=ON \
  -DGPU_ROS_MANAGED_BUILD_CUDA=OFF \
  -DGPU_ROS_MANAGED_BUILD_HIP=OFF
```

ROS adapter builds run from the monorepo root with both `transport` and
`interfaces` as colcon base paths, or from `transport/` with `.` and
`../interfaces`. Source the ROS setup and the resulting overlay before running
ROS tests. A merged install is not required.

Typical focused checks, run only after configuring the matching tree, are:

```bash
ctest --test-dir build/transport-core --output-on-failure
ctest --test-dir build/transport-hip --output-on-failure
ctest --test-dir build/transport-cuda --output-on-failure
colcon test --base-paths transport interfaces \
  --packages-select gpu_ros_managed_core gpu_ros_managed_ros gpu_ros_managed_tensor_bundle
colcon test-result --verbose
```

Backend device tests use CTest skip code 77 when no compatible device exists.
Do not claim a device test passed when it was skipped.

## Verification matrix

For buffer changes, cover core ownership/readiness/failure behavior and the
relevant backend stream, event, copy, device, and pool paths. For ROS changes,
cover the affected wrapper or adapter package and inspect `colcon test-result`.
Run only the layer affected by a change, plus its direct dependencies. Keep
algorithm, provider, and application behavior in the owning `perception/`
packages.

## License boundary

The repository is Apache-2.0. Preserve file-level copyright and modification
notices, and consult [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the
[NITROS port map](docs/nitros-4.5-port-map.md) before changing ported code.
`gpu_ros_tensor_bundle_msgs` is a project-owned Apache-2.0 message package.
The historical `isaac_ros_tensor_list_interfaces` package retains separate
NVIDIA licensing; neither package's license is changed by this adapter. Model,
checkpoint, ONNX, engine, and other runtime assets have their own applicable
terms and are not licensed by this source repository.
