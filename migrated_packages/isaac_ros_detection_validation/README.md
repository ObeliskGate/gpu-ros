# Isaac ROS Detection Validation

Offline numeric sanity checks for `vision_msgs/msg/Detection2DArray` output bags.

The main use case is comparing results generated on different machines, for
example NVIDIA reference output vs AMD candidate output. Each machine should run
the same input bag and record only the detection output topic. The two output
bags can then be copied to one machine and compared offline.

Record outputs on each machine:

```bash
ros2 bag record /detections_output -o nv_rtdetr_output
ros2 bag record /detections_output -o amd_rtdetr_output
```

Compare after copying both bags to the same machine:

```bash
ros2 run isaac_ros_detection_validation compare_detection2d_bags.py \
  --reference-bag ./nv_output_bag \
  --candidate-bag ./amd_output_bag \
  --match-policy index \
  --output-json /tmp/detection_nv_vs_amd.json
```

By default the script uses the only `Detection2DArray` topic in each bag. If a
bag has more than one detection topic, pass `--reference-topic` and
`--candidate-topic` explicitly.

Use `--match-policy stamp` when both outputs preserve the same input header
timestamps. Use `--match-policy index` when comparing bags recorded on different
machines where timestamps are not expected to match but message order is.

Useful options:

- `--min-score 0.3`: drop low-confidence detections before comparing.
- `--max-detections-per-frame 100`: keep only the top-scoring detections per frame.
- `--min-frame-pass-rate 0.7`: require this fraction of all evaluated frames to pass.
- `--max-frame-details 20`: write the worst per-frame comparisons into the JSON report.
- `--ignore-unpaired-frames`: ignore message-count differences and judge only paired frames.

Unpaired frames count against the overall frame pass rate. This means missing
messages on either side are treated as validation failures, even if all paired
frames look good.

For peak-throughput benchmark sweeps, different graph configurations may emit
different numbers of frames before the benchmark stops. Use
`--ignore-unpaired-frames` for that case. Leave it off for strict cross-machine
validation where both runs should cover the same input frames.

## NVIDIA fixed-input capture

The NVIDIA Phase 2B C/Managed comparison has a single-terminal capture runner.
It starts the selected graph, waits for ROS discovery, records
`/detections_output`, plays the complete `r2b_robotarm` input bag, shuts down
cleanly, and requires at least 20 recorded messages. Existing output bags and
logs are never overwritten.

From the Isaac ROS workspace root:

```bash
./src/amd_ros_object_detection/migrated_packages/isaac_ros_detection_validation/scripts/run_nvidia_fixed_input_capture.sh \
  rtdetr-c \
  rtdetr_config_c_20260801

./src/amd_ros_object_detection/migrated_packages/isaac_ros_detection_validation/scripts/run_nvidia_fixed_input_capture.sh \
  rtdetr-managed \
  rtdetr_managed_20260801
```

Set `CAPTURE_PLAYBACK_RATE` to change the default `0.25` playback rate. Set
`CAPTURE_DRAIN_SECONDS` or `CAPTURE_MIN_MESSAGES` only when diagnosing a run.

The runner also accepts `yolov8-c` and `yolov8-managed` lanes.

## NVIDIA transport audit

The unified audit runs Config C and Managed against the same fixed input for
either YOLOv8 or RT-DETR. Run it from the Isaac ROS workspace root after
building and sourcing the workspace:

```bash
./src/amd_ros_object_detection/migrated_packages/isaac_ros_detection_validation/scripts/run_nvidia_transport_audit.sh \
  yolov8 \
  yolov8_transport_audit_20260808

./src/amd_ros_object_detection/migrated_packages/isaac_ros_detection_validation/scripts/run_nvidia_transport_audit.sh \
  rtdetr \
  rtdetr_transport_audit_20260808
```

The historical YOLOv8 entry point remains compatible:

```bash
./src/amd_ros_object_detection/migrated_packages/isaac_ros_detection_validation/scripts/run_nvidia_yolov8_transport_audit.sh \
  yolov8_transport_audit_20260808
```

Each audit archives lane self reports, the paired copy report, ORT provider
placement, first-frame pointer/lifetime binding reports, Nsight traces, and
the fixed stamp-matched, report-only detection comparator JSON. Explicit `memory_copy`
records are the primary copy evidence. Kernel names containing `copy`,
`memcpy`, or `blit` are diagnostic and do not automatically mean a copy.
The ROCprofiler analyzer reads only the official `buffer_records` activity
sections. Each lane also archives a capture manifest recording the requested
tracing domains and output formats; a requested-but-missing or empty
`memory_copy` section is `copy_domain_incomplete`, not evidence of zero-copy.

The result status is `PASS`, `FAIL`, or `INCONCLUSIVE`: an explicit
Managed-only tensor-sized memory-copy record is `FAIL`, while an unresolved
kernel-only payload risk or a memory-copy record without a byte count is
`INCONCLUSIVE`. The direct AMD production lane does not require application
adapter H2D/D2H records; the staged-control lane can opt into that requirement
with `--require-adapter-directions`. H2D/D2H direction and bytes are reported
as staging-shaped evidence, not proof of adapter ownership.
RT-DETR CPU fallback is retained as ORT placement diagnostics and does not fail
closure. Config A is a manual sanity reference only when Config C is
inconclusive.

## AMD Phase 2B transport audit

Run the standard and Managed HIP lanes with the unified runner:

```bash
./src/amd_ros_object_detection/migrated_packages/isaac_ros_detection_validation/scripts/run_amd_transport_audit.sh \
  yolov8 \
  yolov8_amd_transport_audit_20260808

./src/amd_ros_object_detection/migrated_packages/isaac_ros_detection_validation/scripts/run_amd_transport_audit.sh \
  rtdetr \
  rtdetr_amd_transport_audit_20260808
```

The runner completes warm-up first, locates `component_container_mt`, and
attaches directly with:

```bash
rocprofv3 --attach <PID> \
  --memory-copy-trace \
  --kernel-trace \
  --output-format json \
  --attach-sync-output
```

The runner probes whether the installed ROCprofiler supports
`--attach-sync-output` and `--attach-duration-msec`. The duration option keeps
the attach non-interactive. After fixed-input playback and drain, the capture
runner waits for the audit runner to finish detach before stopping the target
graph. If synchronous output is unavailable, it waits for stable non-empty
JSON output before parsing. Only the subsequent fixed-input playback is
included in the rocprof trace. ROCprofiler-SDK 1.0 also requires the attached
target to opt in with `ROCP_TOOL_ATTACH=1`; the runner scopes that variable to
each capture lane and does not require a persistent shell export.
Attach/detach or missing-output errors are hard failures. Direct production
expects no application-level staging; staged-control H2D/D2H records are
reported separately as staging-shaped evidence, while an additional
tensor-sized inference-boundary copy fails the audit. This is not a claim that
the complete pipeline, adapters, decoders, provider kernels, or serialization
never copy.

The real-model benchmark matrix and high-load gate are separate commands:

```bash
ros2 run isaac_ros_detection_validation run_amd_phase2b_benchmark_matrix.sh \
  rtdetr rtdetr_phase2b_matrix_20260812

ros2 run isaac_ros_detection_validation run_amd_phase2b_managed_high_load.sh \
  rtdetr rtdetr_managed_high_load_20260812
```

The matrix runs three rounds with rotating lane order. The high-load command
enforces at least 600 seconds and 10,000 counted input images before writing a
PASS summary.

Before a formal audit on a new ROCm/MI350X environment, run the small HIP
probe with the executable that actually exists in the sibling checkout. The
current sibling provides `gpu_ros_managed_hip_copy_test`, but the probe keeps
the path explicit so it does not assume a target name or build layout:

```bash
ros2 run isaac_ros_detection_validation run_rocprof_hip_copy_probe.sh \
  /path/to/gpu_ros_managed_hip_copy_test \
  /tmp/rocprof-hip-copy-probe
```

The probe requests memory-copy, kernel, and HIP runtime tracing in JSON and
CSV. JSON is authoritative for bytes; CSV only cross-checks direction, agent,
and count. A D2H record with operation 3 proves that the profiler captured D2H
in that attach mode. HIP runtime or `__amd_rocclr_copyBuffer` activity alone
does not prove D2H bytes. The formal attach runner remains JSON-only until this
probe has confirmed stable JSON/CSV output and HIP runtime tracing on the target
ROCm/MI350X attach setup.
