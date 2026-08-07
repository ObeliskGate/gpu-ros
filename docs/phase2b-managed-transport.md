# Phase 2B: Managed Device TensorList Transport

Phase 2A is complete for the AMD RT-DETR and YOLOv8 standard ROS 2 paths.
Those paths are the reference for Phase 2B benchmark and correctness
comparisons. Phase 2B must not make either Phase 2A path
unavailable or require managed transport for standard ROS 2 operation.

## Ownership and scope

The reusable transport and ownership layer lives in the required sibling
repository gpu_ros_managed. This application owns the detection graph, model
preprocessing/decoding, ORT provider selection, I/O Binding, synchronization,
and Phase 2A standard ROS 2 path.

gpu_ros_managed owns device allocation ownership, ready events, reader leases,
deferred release, TensorList views, ROS transport, CUDA/HIP backends, and
component wiring.

The common lifetime contract is:

~~~text
shared message -> ManagedTensorListView -> OnnxInferenceCore
  -> ReadHandle or BlockingReadyLease -> Ort::Value
  -> synchronized Run -> lease destruction -> message/buffer release
~~~

Select the adapter with transport=managed. The inference core must not depend
on ROS message types, NITROS/GXF types, or model-specific decoder types.

## NVIDIA comparison graph

The NVIDIA Managed graph compares transport; it does not replace the official
pipeline:

~~~text
official NITROS preprocessor
  -> NitrosToManagedTensorListNode
  -> OnnxInferenceNode(transport=managed, execution_provider=cuda)
  -> ManagedToNitrosTensorListNode
  -> official NITROS decoder
~~~

Both bridges must wait for upstream readiness, preserve the existing device
allocation owner, preserve stream/event ordering, and add no tensor-payload
copy. Callback timing is diagnostic only and is enabled with
enable_timing=true on the bridge nodes.

## Zero-copy meaning

Zero-copy means that the Managed transport boundary adds no tensor-payload
copy. It does not mean the complete application has no memcpy: preprocessing,
provider kernels, decoders, CPU fallback, and serialization may copy.

A zero-copy result requires pointer identity, equivalent input and graph,
frame-normalized system-level copy counts, no Managed-only payload-copy
signature, detection-output equivalence, and explained provider placement.

## Backend-specific build and tests

NVIDIA uses the CUDA profile: CUDA and NITROS enabled, ROCm and MIGraphX
disabled. AMD builds the managed HIP backend for its ownership tests, while
the application profile keeps ORT CUDA, ORT ROCm, and NITROS disabled and uses
MIGraphX only. Do not run CUDA and HIP backend tests as one cross-platform
command.

NVIDIA test command:

~~~bash
colcon test \
  --merge-install \
  --packages-select \
    gpu_ros_managed_core \
    gpu_ros_managed_cuda \
    gpu_ros_managed_ros \
    gpu_ros_managed_tensor_list \
    isaac_ros_onnx_inference \
  --event-handlers console_direct+
~~~

AMD test command:

~~~bash
colcon test \
  --merge-install \
  --packages-select \
    gpu_ros_managed_core \
    gpu_ros_managed_hip \
    gpu_ros_managed_ros \
    gpu_ros_managed_tensor_list \
    isaac_ros_onnx_inference \
  --event-handlers console_direct+
~~~

Run colcon test-result --all --verbose after either command.

Before transport benchmarking, run the backend-neutral lifecycle, event
failure, safe-orphan, non-default-stream, multi-reader, pool-destruction,
pending-cleanup, external-owner, device-mismatch, session-lifetime, and
pointer-identity tests.

The existing NVIDIA validation entry points are documented in
migrated_packages/isaac_ros_detection_validation/README.md. Run the
proof-of-life test, transport probe, fixed-input comparison, and benchmark
only after the package tests pass.

## Numeric and system audits

Managed and reference lanes must use equivalent input. Detection comparison
must report paired/unpaired frames, class match, box IoU, score delta, and
frame pass rate. Throughput alone is not a correctness result.

Pointer identity proves allocation identity at the tested boundary; it does
not prove that the whole application has no other copy. Nsight Systems or the
platform profiler must compare H2D, D2H, and D2D counts and bytes per output
frame, payload sizes, frame counts, and Managed-only signatures.

ORT profiles are provider-placement controls, not replacements for system
copy traces. Bridge timing reports callback/readiness/publish cost, not copy
latency.

## AMD Phase 2B staging policy

The AMD graph makes host staging explicit at the ROS boundary:

~~~text
host TensorList -> StdToManagedHipTensorListNode
  -> OnnxInferenceNode(transport=managed, execution_provider=migraphx)
  -> ManagedHipToStdTensorListNode -> host TensorList
~~~

`StdToManagedHipTensorListNode` allocates HIP `DeviceBuffer` objects and uses
the public blocking H2D API. `ManagedHipToStdTensorListNode` uses the public
blocking D2H API. Neither adapter calls HIP runtime copy functions directly.

Inside the Managed inference boundary, device-backed inputs are passed to ORT
with their original pointer and a MIGraphX-compatible HIP device memory
descriptor. Static output metadata is probed with preallocated Managed HIP
buffers; if the external binding probe fails, the real failure is logged and
ORT-owned device output is adopted only after synchronization. Dynamic output
metadata follows the ORT-owned path without changing the ONNX graph.

This establishes that the Managed TensorList boundary adds no additional
tensor-payload copy beyond the two explicit application staging adapters. It
does not claim that the complete pipeline has no memcpy: host encoders,
decoders, explicit adapters, provider kernels, and CPU fallback remain outside
that boundary. These AMD changes do not affect the independent Phase 2A
standard ROS 2 path.

Concrete hardware, dates, benchmark values, copy audits, timings, and
shutdown limitations belong in phase2b-results.md.
