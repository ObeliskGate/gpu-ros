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
not prove that the whole application has no other copy. The audit archives a
machine-readable first-frame binding report containing tensor names, byte
counts, storage, pointers, pointer equality, and the lifetime path.

Nsight Systems or the platform profiler must compare H2D, D2H, and D2D counts
and bytes per output frame, payload sizes, frame counts, and Managed-only
signatures. Explicit `memory_copy` records are the primary copy evidence.
Kernel traces are retained to discover and explain possible payload movement;
a kernel name containing `copy`, `memcpy`, or `blit` is not automatically a
copy failure. A confirmed Managed-only tensor-sized memory copy is `FAIL`; an
unresolved payload risk or a memory-copy record without a byte count is
`INCONCLUSIVE`. The direct AMD production lane expects no application-level
adapter copies; the staged-control lane may explicitly require both H2D and
D2H adapter directions. Direction and bytes alone classify a record as
staging-shaped; they do not prove that it came from the Managed adapter.

ORT profiles are provider-placement controls, not replacements for system
copy traces. Bridge timing reports callback/readiness/publish cost, not copy
latency.

## AMD Phase 2B Managed HIP production policy

The production AMD graphs are direct, same-process Managed HIP paths. Their
application-level topology is:

~~~text
YOLOv8:
Image -> YoloV8ManagedHipImageEncoderNode
      -> OnnxInferenceNode(transport=managed, hip_managed_strict)
      -> YoloV8ManagedHipDecoderNode -> Detection2DArray

RT-DETR:
Image -> RtDetrManagedHipImageEncoderNode
      -> RtDetrManagedHipPreprocessorNode
      -> OnnxInferenceNode(transport=managed, hip_managed_strict)
      -> RtDetrManagedHipDecoderNode -> Detection2DArray
~~~

Every intermediate TensorList is an in-process Managed HIP device buffer.
Production launch files and direct benchmark graphs must not instantiate
`StdToManagedHipTensorListNode` or `ManagedHipToStdTensorListNode`. The five
production plugins are:

~~~text
YoloV8ManagedHipImageEncoderNode
YoloV8ManagedHipDecoderNode
RtDetrManagedHipImageEncoderNode
RtDetrManagedHipPreprocessorNode
RtDetrManagedHipDecoderNode
~~~

The encoder performs asynchronous H2D plus the shared HIP preprocessing kernel
and retains the ROS Image owner until the producer event is safe. The RT-DETR
preprocessor aliases the image buffer and writes `orig_target_sizes` into a
separate Managed HIP pool block. Decoders take device read leases and perform
only the required D2H before calling the shared CPU decoder core.

### Strict Managed I/O contract

Production ORT uses `managed_io_contract=hip_managed_strict` and explicit
contracts such as:

~~~text
images=float32[1,3,640,640]
output0=float32[1,84,8400]
orig_target_sizes=int64[1,2]
labels=int64[1,300]
boxes=float32[1,300,4]
scores=float32[1,300]
~~~

The session validates model names, dtype, rank, concrete dimensions, and byte
size during initialization. Symbolic model dimensions are resolved only by
the explicit graph contract. Unknown dynamic outputs and contract mismatch
fail initialization. Each output has a fixed Managed HIP pool; ORT is given a
pre-bound `Ort::Value`, and after `SynchronizeOutputs()` the implementation
checks output name, shape, dtype, memory info, and exact pointer identity before
publishing the pool block. Strict mode never adopts ORT-owned output.

`gpu_ros_managed` exposes three read-only readiness states: not ready,
synchronously ready, and event-backed ready. HIP producers finalize with the
real producer event. ORT synchronized writes use a separate synchronized-write
handle and become ready only after ORT output synchronization. A submitted
ORT failure marks the output block failed and the strict core unhealthy; it is
not recycled. An unsubmitted reservation is canceled and returned to the pool.

The production defaults are `managed_pool_capacity=16` and
`managed_pool_wait_timeout_ms=100`. Pool exhaustion drops the frame and never
switches transport, allocates a temporary buffer, or falls back to ORT-owned
output. Encoder, preprocessor, and ORT pools drain during shutdown; a timeout
is reported as a contract failure rather than force-releasing live storage.

### Staged control lane

The staged-control benchmarks are intentionally separate from production. They
reuse the Managed HIP preprocessing path, strict ORT configuration, and shared
standard decoder core, but insert the existing adapters around the inference
boundary:

~~~text
Managed HIP preprocessing
  -> ManagedHipToStdTensorListNode
  -> standard TensorList materialization
  -> StdToManagedHipTensorListNode
  -> strict Managed ORT
  -> ManagedHipToStdTensorListNode
  -> shared standard decoder core
~~~

They are named
`isaac_ros_yolov8_phase2b_amd_staged_control_graph.py` and
`isaac_ros_rtdetr_phase2b_amd_staged_control_graph.py`. The only intentional
difference from direct Managed is the adapter and ROS standard-message
materialization cost.

The `StdToManagedHipTensorListNode` input adapter uses a fixed-capacity HIP
pool per tensor byte size and an event-backed H2D write. It must drop a staged
frame on pool exhaustion rather than perform an unbounded per-frame
`hipMalloc` or send an untracked buffer into strict ORT. The
`ManagedHipToStdTensorListNode` direction remains an explicit blocking D2H
materialization for this control lane.

The AMD ROCprof audit defaults to the direct production interpretation: no
adapter direction is required. For staged-control diagnostics, pass
`--require-adapter-directions` to require the expected H2D and D2H evidence.
In either mode ROCprof is copy/kernel evidence only; it is not readiness,
owner, or lease safety evidence.

The unified AMD audit performs the post-warm-up attach itself:

```bash
./src/amd_ros_object_detection/migrated_packages/isaac_ros_detection_validation/scripts/run_amd_transport_audit.sh \
  rtdetr \
  rtdetr_amd_transport_audit_20260808
```

It uses `rocprofv3 --attach <PID>` with memory-copy and kernel tracing. The
installed ROCprofiler-SDK 1.0 requires the target process to opt in with
`ROCP_TOOL_ATTACH=1`; the runner sets that variable only for each capture lane
and does not modify the surrounding Apptainer or shell environment. The
runner uses `--attach-duration-msec` when available so attachment is
non-interactive. After fixed-input playback and drain, the capture process
signals completion while keeping the graph alive; the audit runner detaches
rocprofv3 and acknowledges completion before the capture process stops the
graph. It uses `--attach-sync-output` when supported and otherwise waits for
stable non-empty JSON output before parsing.
Attach/detach or missing JSON output remains a hard tooling failure. Each lane
writes a manifest containing the requested domains and output formats. The
AMD-specific parser consumes only `buffer_records.memory_copy` and
`buffer_records.kernel_dispatch`; malformed records remain diagnostics and
make the report `INCONCLUSIVE`. Warm-up events are not used as a substitute
for a fixed-input trace.

For real-model fixed-input capture, the existing single-terminal runner keeps
`CAPTURE_TRANSPORT=std` as its Phase 2A default. Set
`CAPTURE_TRANSPORT=managed` to select the Phase 2B direct Managed HIP launch
files:

~~~bash
CAPTURE_TRANSPORT=managed \
CAPTURE_EXECUTION_PROVIDER=migraphx \
ros2 run isaac_ros_detection_validation \
  run_amd_phase2a_fixed_input_capture.sh \
  amd_phase2b_rtdetr_managed

CAPTURE_TRANSPORT=managed \
CAPTURE_EXECUTION_PROVIDER=migraphx \
ros2 run isaac_ros_detection_validation \
  run_amd_phase2a_fixed_input_capture.sh \
  yolov8 amd_phase2b_yolov8_managed
~~~

The resulting Managed bags are candidates for comparison against the
corresponding standard AMD bags or the existing NVIDIA reference. Formal
comparisons use the fixed report-only mode so a post hoc threshold cannot turn
the result into an aggregate PASS/FAIL claim:

~~~bash
ros2 run isaac_ros_detection_validation compare_detection2d_bags.py \
  --reference-bag /path/to/reference \
  --candidate-bag /path/to/managed \
  --match-policy stamp \
  --report-only \
  --output-json /path/to/reports/detection_comparison.json
~~~

The AMD direct and staged-control throughput benchmarks are separate from the
CUDA Managed benchmark graphs. Run the topology, preprocessing, POL, strict
contract, and fixed-input checks before the real-model benchmark:

~~~bash
launch_test \
  migrated_packages/benchmarks/isaac_ros_rtdetr_phase2b_amd_managed_graph.py

launch_test \
  migrated_packages/benchmarks/isaac_ros_yolov8_phase2b_amd_managed_graph.py

launch_test \
  migrated_packages/benchmarks/isaac_ros_rtdetr_phase2b_amd_staged_control_graph.py

launch_test \
  migrated_packages/benchmarks/isaac_ros_yolov8_phase2b_amd_staged_control_graph.py
~~~

The `*_amd_managed_graph.py` files are the production direct entries and have
no staging plugins. The `*_staged_control_graph.py` files are the explicit
control lane. The existing `isaac_ros_rtdetr_managed_graph.py` and
`isaac_ros_yolov8_managed_graph.py` remain CUDA/NVIDIA graphs.

Run the three-lane benchmark matrix as three rounds of independent processes;
the runner rotates `std`, `staged-control`, and direct Managed order and
archives every JSON/log under a unique matrix directory:

~~~bash
ros2 run isaac_ros_detection_validation run_amd_phase2b_benchmark_matrix.sh \
  rtdetr rtdetr_phase2b_matrix_20260812
ros2 run isaac_ros_detection_validation run_amd_phase2b_benchmark_matrix.sh \
  yolov8 yolov8_phase2b_matrix_20260812
~~~

The fixed-rate 10/30/60 Hz trials are part of each graph configuration. The
separate high-load gate runs the production direct lane for at least ten
minutes and 10,000 input images, and retains the input counter, detections bag,
binding report, ORT profile, and lifecycle log:

~~~bash
ros2 run isaac_ros_detection_validation run_amd_phase2b_managed_high_load.sh \
  rtdetr rtdetr_managed_high_load_20260812
~~~

Formal real-model comparison uses exact source-header-stamp FIFO pairing,
greedy one-to-one matching of all detections by descending positive IoU,
class-independent pairing, and reports matched/unmatched counts, IoU, score
delta, and class equality. The report records per-frame and aggregate
distributions, the comparison schema, fixed CLI parameters, and comparator
source hash; it does not produce an aggregate PASS/FAIL. Each archive also
records topology, contract, binding, ORT profile, model/dataset hashes, and
both repository revisions.

Concrete hardware, dates, benchmark values, copy audits, timings, and
shutdown limitations belong in phase2b-results.md.
