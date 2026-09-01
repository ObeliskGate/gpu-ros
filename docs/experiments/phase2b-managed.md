# Phase 2B Managed HIP runbook

## Scope and topology

Phase 2B validates the Managed HIP TensorBundle transport supplied by the
`gpu_ros_managed` sibling. It has two AMD lanes per model:

- direct production: Managed HIP end to end around strict Managed ORT;
- staged control: the explicit Managed-to-standard-to-Managed adapters around
  ORT, with the standard decoder after ORT.

The actual staged topology is:

```text
Managed preprocessing -> Managed -> standard -> Managed ORT
                      -> Managed -> standard -> standard decoder
```

The adapters are not silently inserted into the production direct graph.

## Build and contract gates

Use the public [AMD runtime contract](amd-runtime-contract.md) and the same
external ORT procedure as [Phase 2A](phase2a-amd.md). Build and source the
application and sibling workspace once. Run the Managed sibling's
lifecycle/event/device tests and the application package tests before any
real-model benchmark. The contract is complete without a private hostname,
partition, SIF filename, or deployment path; those are optional site-adapter
inputs. No `uv` command is needed in the AMD SIF.

If the login shell exports `APPTAINERENV_HOME`, clear it before starting the
Apptainer launcher:

```bash
unset APPTAINERENV_HOME
```

```bash
./docker/phase2-amd.sh preflight
./docker/phase2-amd.sh colcon
./docker/phase2-amd.sh verify
./docker/phase2-amd.sh shell

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

The gates must cover abandoned/incomplete `WriteHandle` behavior, event and
device ownership, non-default devices, pool destruction, pending cleanup,
staging single/totals/entry/overflow limits, and the standalone installed
CMake consumer for `gpu_ros_managed_core`.

## Fixed-input direct and staged capture

Use the Phase 2A runner with `CAPTURE_TRANSPORT=managed` for the direct
production path:

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

Run the dedicated AMD transport audit for both models. For staged-control
audits, add the explicit adapter-direction requirement only when the audit
needs to prove the expected staging-shaped H2D/D2H evidence:

```bash
ros2 run gpu_ros_detection_validation \
  run_amd_transport_audit.sh rtdetr "rtdetr_amd_transport_${RUN_ID}"
ros2 run gpu_ros_detection_validation \
  run_amd_transport_audit.sh yolov8 "yolov8_amd_transport_${RUN_ID}"
```

`rocprofv3` memory-copy records are the primary copy evidence. Kernel names
containing `copy`, `memcpy`, or `blit` are diagnostic only. An unresolved or
incomplete record is `INCONCLUSIVE`, not proof of zero-copy.

Compare standard, direct Managed, and staged bags with strict class-aware
matching. Record all unmatched detections, IoU, score deltas, class equality,
and frame-level pass rates.

## Graph proof and matrix

Run the four graph-level checks before the matrix:

```bash
launch_test migrated_packages/benchmarks/gpu_ros_rtdetr_phase2b_amd_managed_graph.py
launch_test migrated_packages/benchmarks/gpu_ros_yolov8_phase2b_amd_managed_graph.py
launch_test migrated_packages/benchmarks/gpu_ros_rtdetr_phase2b_amd_staged_control_graph.py
launch_test migrated_packages/benchmarks/gpu_ros_yolov8_phase2b_amd_staged_control_graph.py
```

The matrix runner performs three independent rounds, rotates lane order, and
executes fixed 10/30/60 Hz trials for standard, staged-control, and direct
Managed lanes:

```bash
ros2 run gpu_ros_detection_validation \
  run_amd_phase2b_benchmark_matrix.sh \
  rtdetr "rtdetr_phase2b_matrix_${RUN_ID}"
ros2 run gpu_ros_detection_validation \
  run_amd_phase2b_benchmark_matrix.sh \
  yolov8 "yolov8_phase2b_matrix_${RUN_ID}"
```

The result manifest records application and sibling HEADs, staged and
unstaged diff hashes, untracked paths/content hashes, asset hashes, and dirty
state. Dirty state is evidence, not a run gate.

## Acceptance

Phase 2B is not closed by throughput alone. Both models require package/POL
tests, direct and staged topology checks, strict Managed contract checks,
fixed-input comparison, provider/copy audit, and the three-lane matrix. The
removed high-load experiment is not an acceptance gate. A matrix shell `PASS`
only proves that the nine lane processes emitted JSON; promote a performance
baseline only after publishing each lane's three peak reports, fixed 10/30/60
rows, aggregation rule, and correctness comparison. Use the tolerance and
status semantics in [`results/README.md`](../results/README.md). Current AMD
result reports remain historical until this sequence completes on the selected
sibling and application revisions.
