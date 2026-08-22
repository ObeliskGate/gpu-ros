# gpu_ros_managed

Maintainer: Boshen Chen (@ObeliskGate)

`gpu_ros_managed` is a small ROS 2 transport and ownership layer for device
buffers and TensorLists. It provides one backend-neutral core with CUDA and
HIP implementations, plus ROS 2 publisher/subscriber and TensorList adapters.
Inference engines, model code, and application graphs remain outside this
repository.

## Architecture

The core exposes `DeviceBuffer`, explicit write/read handles, readiness states,
and a fixed-capacity device-memory pool. A backend supplies allocation, stream,
event, and copy operations through a narrow internal interface. The CUDA and
HIP packages only provide those backend factories; there is no runtime plugin
registry and the core public headers do not include CUDA, HIP, ONNX Runtime,
GXF, or `rclcpp` headers.

The ROS packages are:

| Package | Responsibility |
| --- | --- |
| `gpu_ros_managed_core` | backend-neutral buffers, handles, streams, and pools |
| `gpu_ros_managed_cuda` | CUDA allocation, stream, event, and copy backend |
| `gpu_ros_managed_hip` | HIP allocation, stream, event, and copy backend |
| `gpu_ros_managed_ros` | intra-process managed publisher/subscriber wrappers |
| `gpu_ros_managed_tensor_list` | managed TensorList value type and ROS adapter |

## Ownership and synchronization contract

An allocation is released only after the producer event and all reader events
are complete. A write handle must be finalized after recording a trustworthy
completion event, or explicitly failed when completion cannot be proven. A
synchronous writer must only finalize after its external operation has
completed. Failed or uncertain allocations are orphaned rather than returned
to a pool. Reader leases keep the allocation and its producer owners alive.

Streams must belong to the same backend and device as the buffer. Event-backed
readiness is not a substitute for stream ordering: consumers still acquire a
read lease and honor the lease contract. Fixed pools have bounded capacity and
report timeout/exhaustion; they do not silently allocate an untracked temporary
buffer.

## CUDA and HIP support

CUDA and HIP are separate build/runtime paths. The backend-neutral core can be
built and tested without either SDK. HIP support is intended for ROCm systems;
CUDA support is intended for CUDA systems. This project does not promise that a
particular ROS 2, CUDA, HIP, or driver release is ABI-compatible with every
other release. Consumers should build all sibling packages against the same
workspace and toolchain.

## Build and minimal example

The historical TensorList adapter requires the separately licensed NVIDIA message package and is not a pure-AMD message path. The core and HIP backend remain independent.


Standalone core verification does not require ROS or a GPU SDK:

```bash
cmake -S . -B build -DGPU_ROS_MANAGED_BUILD_CUDA=OFF
cmake --build build
ctest --test-dir build --output-on-failure
```

For ROS, place this repository and the application repository in one colcon
workspace, source the ROS distribution, and select the backend package needed
by the application:

```bash
colcon build --merge-install \
  --packages-select \
    gpu_ros_managed_core \
    gpu_ros_managed_ros \
    gpu_ros_managed_hip
```

A producer acquires a `WriteHandle`, writes the device payload, records its
backend event, and calls `finalize()`. A consumer obtains a read lease, uses
the payload on the agreed stream, and releases the lease when its work has
been submitted. See the public headers and tests for the complete API.

## NITROS provenance and license boundary

Several buffer, pool, TensorList, and managed ROS files are file-level ports
of Apache-2.0 source files from NVIDIA Isaac ROS NITROS `v4.5-0`, commit
`82310fce298d3d9db26945a3a988d5c471d14973`. Their headers retain the NVIDIA
copyright and identify the local modifications. The NITROS repository and
some of its package metadata use the NVIDIA Isaac ROS Software License; this
repository does not relicense that repository or claim that every NVIDIA Isaac
ROS component is Apache-2.0. The local TypeAdapter has no one-to-one upstream
NITROS source and is documented as project-original.

The component-level boundary and external dependency policy are recorded in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). This is an independent
project and is not an NVIDIA official project.

## Limitations

- The library manages ownership and ordering; it does not make an arbitrary
  kernel, ROS callback, or inference provider asynchronous-safe.
- Correctness depends on callers recording real completion events and using
  the matching backend/device/stream.
- Pool capacity and shutdown timeouts are application policy choices.
- The ROS TensorList adapter copies host data and performs blocking device to
  host conversion when serializing a device tensor; it is not a claim of
  end-to-end zero-copy.
- GPU benchmarks, models, datasets, traces, and runtime artifacts are not
  source-release contents.
