# Phase 2B Managed HIP runbook

## Scope and topology

Phase 2B validates the managed HIP TensorBundle transport in the same GPU ROS
monorepo as the application. The managed implementation lives under
`gpu_ros_managed/`; it is not a sibling checkout. There are two AMD lanes per model:

- direct production: managed HIP end to end around strict managed ORT;
- staged control: explicit managed-to-standard-to-managed adapters around ORT,
  with the standard decoder after ORT.

The actual staged topology is:

```text
Managed preprocessing -> Managed -> standard -> Managed ORT
                      -> Managed -> standard -> standard decoder
```

The adapters are not silently inserted into the production direct graph.

## Build and contract gates

Use the [AMD runtime contract](amd-runtime-contract.md) and the verified
external ORT install described in [Phase 2A](phase2a-amd.md). Build the managed,
message, and detection packages from one checkout at `/workspaces/gpu-ros`.
Use the contract's non-interactive Docker payload by default, or its explicit
Apptainer branch inside an authorized compute allocation. No `uv` command is
needed in the AMD SIF.

Run these package checks inside the initialized payload:

```bash
colcon test \
  --packages-select \
    gpu_ros_managed_core \
    gpu_ros_managed_hip \
    gpu_ros_managed_ros \
    gpu_ros_managed_tensor_bundle \
    gpu_ros_tensor_bundle_msgs \
    gpu_ros_onnx_inference \
    gpu_ros_rtdetr \
    gpu_ros_yolov8 \
    gpu_ros_detection_validation \
  --event-handlers console_direct+
colcon test-result --all --verbose
```

The gates cover abandoned/incomplete `WriteHandle` behavior, event and device
ownership, non-default devices, pool destruction, pending cleanup, staging
single/totals/entry/overflow limits, and the standalone installed CMake
consumer for `gpu_ros_managed_core`.

## Fixed-input capture and transport audit

Use the Phase 2A runner with `CAPTURE_TRANSPORT=managed` for the direct path.
Keep the baseline parameters explicit as described in
[detection validation](detection-validation.md#amd):

```bash
CAPTURE_TRANSPORT=managed \
CAPTURE_EXECUTION_PROVIDER=migraphx \
ros2 run gpu_ros_detection_validation \
  run_amd_phase2a_fixed_input_capture.sh \
  "rtdetr_phase2b_direct_${RUN_ID}"

CAPTURE_TRANSPORT=managed \
CAPTURE_EXECUTION_PROVIDER=migraphx \
ros2 run gpu_ros_detection_validation \
  run_amd_phase2a_fixed_input_capture.sh \
  yolov8 "yolov8_phase2b_direct_${RUN_ID}"
```

Run the native std/managed transport audit separately for both models:

```bash
ros2 run gpu_ros_detection_validation \
  run_amd_transport_audit.sh rtdetr "rtdetr_amd_transport_${RUN_ID}"
ros2 run gpu_ros_detection_validation \
  run_amd_transport_audit.sh yolov8 "yolov8_amd_transport_${RUN_ID}"
```

`rocprofv3` memory-copy records are the primary copy evidence. Kernel names
containing `copy`, `memcpy`, or `blit` are diagnostic only. An unresolved or
incomplete record is `INCONCLUSIVE`, not proof of zero-copy.

For separately collected staged-control traces, the analyzer can require
adapter directions; this is not an extra argument to the capture runner.
Compare each migration lane with its own pre-change bag using strict
class-aware stamp matching. Archive std/managed cross-lane observations
separately, without using them to waive a failed same-lane gate.

## Graph proof and matrix

`launch_test` on the managed or staged graph files runs a benchmark, not a
cheap graph smoke test. Do not run those four graphs again as a prerequisite
to the matrix; the native matrix already exercises them. Performance work
requires an explicit request.

The native runner performs three independent rounds for standard, staged,
and direct lanes, in the order std/staged/direct, staged/direct/std, then
direct/std/staged. Each graph requests fixed 10/30/60 Hz trials. Use the
existing runner as a runtime payload, one model at a time:

```bash
ros2 run gpu_ros_detection_validation \
  run_amd_phase2b_benchmark_matrix.sh \
  rtdetr "rtdetr_phase2b_matrix_${RUN_ID}"
ros2 run gpu_ros_detection_validation \
  run_amd_phase2b_benchmark_matrix.sh \
  yolov8 "yolov8_phase2b_matrix_${RUN_ID}"
```

A schema 3 manifest records the executed monorepo revision, dirty/untracked
fingerprints, assets, and runtime. The matrix shell's PASS means its nine
processes emitted JSON. Check component exit logs, actual fixed-rate fields,
and post-exit processes separately. Missing fields remain MISSING.

## Acceptance

Phase 2B is not closed by throughput alone. Both models require package/POL
tests, direct and staged topology checks, managed contract checks, fixed-input
comparison, provider/copy audit, and the three-lane matrix. The removed
high-load experiment is not an acceptance gate. Promote a performance baseline
only after publishing each lane's three peak reports, fixed 10/30/60 rows,
aggregation rule, and correctness comparison. Follow
[result acceptance](result-acceptance.md), without changing tolerances to match
a historical label. Compare an out-of-tolerance observation with preserved
pre-change source on the current device before attributing it to a migration.
