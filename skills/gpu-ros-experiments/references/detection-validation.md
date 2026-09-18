# Detection validation

Use the installed `gpu_ros_detection_validation` commands after building and
sourcing the selected runtime. Start with the [AMD runtime contract](amd-runtime-contract.md)
or [NVIDIA method](phase2b-nvidia.md). Commands below are non-interactive runtime
payloads, not instructions to enter a separate shell. Keep bags and reports in
an external result directory and choose a new output name for every run.

Fixed-input capture checks detections. Copy audits check a specific transport
boundary. Neither is a throughput measurement. See
[result acceptance](result-acceptance.md) before interpreting a status.

## Fixed-input capture

Both versions of a migration comparison must use the same input/model bytes,
provider, image dimensions, size policy, parameters, and runtime. Record those
identities before graph startup. Disable profiling for ordinary captures.
The existing runners start the graph, wait for discovery, record detections,
replay the input, drain, and stop the graph. They reject existing output names.
Their message-count floor does not prove complete input coverage.

### AMD

Use the same environment for both models and both transports:

```bash
export CAPTURE_OUTPUT_ROOT="/workspaces/ovg-results/${RUN_ID}/bags"
export CAPTURE_PLAYBACK_RATE=0.25 CAPTURE_DRAIN_SECONDS=10
export CAPTURE_MIN_MESSAGES=20 CAPTURE_EXECUTION_PROVIDER=migraphx
export CAPTURE_RECORD=1 CAPTURE_WARMUP_ONLY=0
export CAPTURE_STOP_GRACE_SECONDS=60 CAPTURE_STOP_TERM_SECONDS=20
unset CAPTURE_ORT_PROFILE_PREFIX CAPTURE_BINDING_REPORT_PATH CAPTURE_NSYS_OUTPUT

CAPTURE_TRANSPORT=std ros2 run gpu_ros_detection_validation \
  run_amd_phase2a_fixed_input_capture.sh "rtdetr-std-${RUN_ID}"
CAPTURE_TRANSPORT=managed ros2 run gpu_ros_detection_validation \
  run_amd_phase2a_fixed_input_capture.sh "rtdetr-managed-${RUN_ID}"
CAPTURE_TRANSPORT=std ros2 run gpu_ros_detection_validation \
  run_amd_phase2a_fixed_input_capture.sh yolov8 "yolov8-std-${RUN_ID}"
CAPTURE_TRANSPORT=managed ros2 run gpu_ros_detection_validation \
  run_amd_phase2a_fixed_input_capture.sh yolov8 "yolov8-managed-${RUN_ID}"
```

A warmup-only invocation is a readiness diagnostic, not a recorded correctness
run. Do not silently substitute CPU execution when MIGraphX is unavailable.

### NVIDIA

The six lanes are `rtdetr-c`, `rtdetr-d`, `rtdetr-managed`, `yolov8-c`,
`yolov8-d`, and `yolov8-managed`. Five use `gpu_ros_nvidia_reference` launches;
`yolov8-d` retains its model-package launch.

```bash
export CAPTURE_OUTPUT_ROOT="/workspaces/ovg-results/${RUN_ID}/bags"
export CAPTURE_PLAYBACK_RATE=0.25 CAPTURE_DRAIN_SECONDS=30
export CAPTURE_MIN_MESSAGES=100
export CAPTURE_STOP_GRACE_SECONDS=60 CAPTURE_STOP_TERM_SECONDS=20
unset CAPTURE_ORT_PROFILE_PREFIX CAPTURE_BINDING_REPORT_PATH CAPTURE_NSYS_OUTPUT

ros2 run gpu_ros_detection_validation run_nvidia_fixed_input_capture.sh \
  rtdetr-c "rtdetr-c-${RUN_ID}"
ros2 run gpu_ros_detection_validation run_nvidia_fixed_input_capture.sh \
  rtdetr-managed "rtdetr-managed-${RUN_ID}"
```

Invoke the same command separately for each other requested lane. Do not run
capture lanes concurrently on one GPU.

## Compare detections

The comparator reads `vision_msgs/msg/Detection2DArray` bags. It selects the
only detection topic by default; provide `--reference-topic` and
`--candidate-topic` when a bag contains more than one.

For a migration, compare each lane with its own pre-change capture using the
[formal same-bag gate](result-acceptance.md#reproduction-acceptance):

```bash
ros2 run gpu_ros_detection_validation compare_detection2d_bags.py \
  --reference-bag "${BEFORE_BAG}" --candidate-bag "${AFTER_BAG}" \
  --match-policy stamp --min-score 0.0 --max-detections-per-frame 0 \
  --class-aware-matching --min-paired-frames 20 --min-class-match-rate 1.0 \
  --min-mean-iou 0.99 --min-pair-iou 0.99 \
  --max-mean-score-delta 0.001 --max-pair-score-delta 0.001 \
  --min-frame-pass-rate 1.0 --output-json "${COMPARISON_JSON}"
```

Unpaired frames count against the overall pass rate. Retain all class,
IoU, score, matched/unmatched detection, and frame-count fields. Compare AMD
std versus managed and NVIDIA C versus managed or D separately when requested;
a REPORT_ONLY cross-lane report does not replace the migration gate.

The tool also supports diagnostic options: `--match-policy index`, score or
detection-count filtering, `--max-frame-details`, and
`--ignore-unpaired-frames`. Index pairing requires an independently justified
message-order correspondence. Ignoring unmatched frames can be useful when
inspecting drop-allowing throughput sweeps. Neither option is permitted in the
strict migration comparison, and a filtered diagnostic cannot be promoted as
that gate.

## Transport audits

Run each model separately, with an independent `CAPTURE_AUDIT_ROOT`. Preserve
the capture parameters used by the baseline. NVIDIA audits use short ORT
profiles in addition to CUDA tracing:

```bash
export CAPTURE_AUDIT_ROOT="/workspaces/ovg-results/${RUN_ID}/audits"
export AUDIT_ORT_PROFILE_FRAMES=50
ros2 run gpu_ros_detection_validation run_nvidia_transport_audit.sh \
  rtdetr "rtdetr-audit-${RUN_ID}"
ros2 run gpu_ros_detection_validation run_nvidia_transport_audit.sh \
  yolov8 "yolov8-audit-${RUN_ID}"
```

The installed `run_nvidia_yolov8_transport_audit.sh` retains its YOLOv8 entry
point. `run_nvidia_c_vs_d_copy_audit.sh` is a separate C/D diagnostic, not an
extra requirement for managed-boundary closure.

For AMD, use the unified std/managed runner:

```bash
export CAPTURE_AUDIT_ROOT="/workspaces/ovg-results/${RUN_ID}/audits"
ros2 run gpu_ros_detection_validation run_amd_transport_audit.sh \
  rtdetr "rtdetr-audit-${RUN_ID}"
ros2 run gpu_ros_detection_validation run_amd_transport_audit.sh \
  yolov8 "yolov8-audit-${RUN_ID}"
```

The AMD runner warms up before attaching ROCprofiler to
`component_container_mt`. It requests memory-copy and kernel tracing with JSON
output, uses synchronous attach output when supported, and probes the duration
option for non-interactive detach. The capture waits for detach before stopping
the graph. Without synchronous output, the runner waits for stable non-empty
JSON. ROCprofiler-SDK 1.0 requires `ROCP_TOOL_ATTACH=1`; the runner scopes it to
each target. Attach/detach and missing-output errors are hard failures.

Archives contain capture manifests, lane reports, provider placement,
pointer/lifetime evidence, traces, copy comparisons, and detection comparisons.
Explicit memory-copy records are primary evidence. Kernel names containing
`copy`, `memcpy`, or `blit` are diagnostics, not proof of payload copies.
The ROCprofiler analyzer reads official `buffer_records` activity sections.
A requested but missing or empty memory-copy section is
`copy_domain_incomplete`, not evidence of zero-copy.

An explicit managed-only tensor-sized copy is FAIL. Unresolved kernel-only
payload risk or a copy without a byte count is INCONCLUSIVE. The direct AMD
production lane does not require adapter H2D/D2H records. The analyzer's
`--require-adapter-directions` option is for separately collected staged-control
evidence; direction and bytes alone do not prove adapter ownership. RT-DETR
CPU fallback remains provider-placement diagnostics and does not fail closure
when the requested provider has activity. These conclusions concern the
inference boundary, not all preprocessing, decoding, provider work, or ROS
serialization.

## HIP copy probe

On a new ROCm/device environment, use the executable from the standalone HIP
build to check profiler coverage:

```bash
ros2 run gpu_ros_detection_validation run_rocprof_hip_copy_probe.sh \
  "${HIP_COPY_TEST}" "${PROBE_OUTPUT}"
```

The probe requests memory-copy, kernel, and HIP runtime tracing in JSON and
CSV. JSON is authoritative for bytes; CSV cross-checks direction, agent, and
count. A D2H record with operation 3 proves D2H capture in that attach mode.
HIP runtime or `__amd_rocclr_copyBuffer` activity alone does not prove D2H
bytes. The formal attach runner remains JSON-only until the target setup has
established stable JSON/CSV and HIP runtime tracing.

## Performance and lifecycle evidence

The [AMD method](phase2b-managed.md#graph-proof-and-matrix) describes the native
three-round matrix. The [NVIDIA method](phase2b-nvidia.md#formal-throughput)
describes its four separate graphs. Both require an explicit benchmark request.
There is no extra high-load correctness gate: normal benchmark drops prevent
input/output counts alone from proving detection equivalence.

Preserve each command's exit code, graph/component logs, emitted-frame
coverage, reports, and post-exit process list. Capture runners can swallow a
child wait failure, and launch tests can pass after a component exits with
SIGSEGV. Check teardown separately. Never overwrite a failed run with a retry;
use a new output ID and keep both. A baseline waiver needs matching evidence
from the same lane, parameters, and failure stage. Unknown or new failures
block acceptance even when a copy parser reports PASS.
