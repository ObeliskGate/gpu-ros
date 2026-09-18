# Managed library agent guide

## Scope

`gpu_ros_managed/` provides C++17 managed GPU buffers with explicit CUDA and HIP
backends, plus optional ROS 2 wrappers and a TensorBundle adapter. The core
headers are backend neutral. Do not describe this component as an inference
engine, model runtime, or inter-process GPU transport.

This collection contains seven ROS packages in the GPU ROS monorepo, including
`gpu_ros_tensor_bundle_msgs` and `gpu_ros_nvidia_tensor_bundle_compat`.
Preserve all package basenames, ROS names, includes, and exported targets.
The compatibility package owns generic TensorList conversions and its boundary
launch. Full NVIDIA detection compositions belong to the detection collection's
`gpu_ros_nvidia_reference` package. Neither package is part of the AMD profile.

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

The standalone CMake project builds only `gpu_ros_managed_core` and optional
CUDA/HIP backends. It does not aggregate ROS, message, or compatibility packages
and must not gain a parent package.xml. Core headers require neither ROS nor a
GPU SDK. HIP can require its SDK with
`-DGPU_ROS_MANAGED_HIP_REQUIRE_SDK=ON`; CUDA has no corresponding require
option and is omitted when its toolkit is unavailable.

From the monorepo root use `cmake -S gpu_ros_managed`; from this directory use
`cmake -S .`. Use an out-of-tree build directory:

```bash
cmake -S gpu_ros_managed -B /tmp/gpu-ros-managed-core \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_DISABLE_FIND_PACKAGE_ament_cmake=TRUE \
  -DBUILD_TESTING=ON -DGPU_ROS_MANAGED_BUILD_TESTING=ON \
  -DGPU_ROS_MANAGED_BUILD_CUDA=OFF \
  -DGPU_ROS_MANAGED_BUILD_HIP=OFF
cmake --build /tmp/gpu-ros-managed-core
ctest --test-dir /tmp/gpu-ros-managed-core \
  --output-on-failure --no-tests=error
```

ROS builds use colcon with individual package directories as base paths, not
the parent CMake project. The project message package is in this collection.
Use the explicit example in [README.md](README.md) or the detection runtime's
tracked colcon defaults. Source ROS and the resulting overlay before package
tests; a merged install is not required. Inspect `colcon test-result` as well
as the command exit.

Backend device tests use CTest skip code 77 when no compatible device exists.
Do not claim a device test passed when it was skipped.

## Verification matrix

For buffer changes, cover core ownership/readiness/failure behavior and the
relevant backend stream, event, copy, device, and pool paths. For ROS changes,
cover the affected wrapper or adapter package and inspect `colcon test-result`.
Run only the affected layer and its dependencies. Algorithm, provider, and
application behavior belongs to `../gpu_ros_object_detection/`. Runtime and
experiment methods are in [the project skill](../skills/gpu-ros-experiments/).

## License boundary

The repository is Apache-2.0. Preserve file-level copyright and modification
notices, and consult [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the
[NITROS port map](docs/nitros-4.5-port-map.md) before changing ported code.
`gpu_ros_tensor_bundle_msgs` is a project-owned Apache-2.0 message package.
The historical `isaac_ros_tensor_list_interfaces` package retains separate
NVIDIA licensing; neither package's license is changed by this adapter. Model,
checkpoint, ONNX, engine, and other runtime assets have their own applicable
terms and are not licensed by this source repository.
