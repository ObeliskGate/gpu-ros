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
the existing stamp-matched detection comparator JSON. Explicit `memory_copy`
records are the primary copy evidence. Kernel names containing `copy`,
`memcpy`, or `blit` are diagnostic and do not automatically mean a copy.

The result status is `PASS`, `FAIL`, or `INCONCLUSIVE`: an explicit
Managed-only tensor-sized memory-copy record is `FAIL`, while an unresolved
kernel-only payload risk is `INCONCLUSIVE`. RT-DETR CPU fallback is retained
as ORT placement diagnostics and does not fail closure. Config A is a manual
sanity reference only when Config C is inconclusive.

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
included in the rocprof trace.
Attach/detach or missing-output errors are hard failures. Managed-only H2D/D2H
records are separately reported as explicit adapter staging; an additional
tensor-sized inference-boundary copy fails the audit. This is not a claim that
the complete pipeline, adapters, decoders, provider kernels, or serialization
never copy.
