# Phase 2A AMD runbook

## Scope

Phase 2A is the standard ROS 2 TensorBundle path on AMD with ONNX Runtime
MIGraphX. It covers RT-DETRv2 R50 and user-provided YOLOv8. It does not resolve
the NVIDIA compatibility package and does not enable NITROS/CUDA.

The RT-DETR default is the formal `rtdetrv2_r50` profile. The exact ONNX SHA
and the model/checkpoint license evidence are in `config/model-profiles.json`.

## Environment and external ORT

Run in the approved AMD runtime environment. Select Docker or Apptainer explicitly
when the environment requires it, and provide the externally built ORT install:

```bash
export OVG_RUNTIME=docker                 # or apptainer
export GPU_ROS_MANAGED_DIR=/absolute/path/to/gpu_ros_managed
export OVG_STATE_ROOT=/absolute/path/to/phase2-state
export OVG_ORT_STATE_HOST=/absolute/path/to/phase2-state/ort
export OVG_ORT_ROOT=/absolute/path/to/ovg-ort/install/<ort-fingerprint>
export AMD_GPU_TARGETS=<runtime-target>    # use the target required by the runtime
unset APPTAINERENV_HOME                   # avoid HOME override warning
```

Site-specific host paths, SIF names, external-ORT fingerprints, and allocation
settings belong in `inner_docs/environment_conventions.md`, not in this
platform-neutral runbook. A launcher must bind its site asset/result roots to
the container paths represented by `OVG_ASSETS_ROOT` and `OVG_RESULTS_ROOT`.

For Apptainer, also set `OVG_APPTAINER_SIF` and the site-provided scheduler
variables. Slurm metadata mismatches warn by default; set
`OVG_REQUIRE_SLURM=1` only for a strict allocation gate. `/dev/kfd` and
`/dev/dri` are always required for an AMD GPU run.

Build the external ORT separately from a pristine recursive checkout at the
locked commit, then run the application environment gates:

```bash
./docker/phase2-amd.sh preflight
./docker/phase2-amd.sh bootstrap
./docker/phase2-amd.sh colcon
./docker/phase2-amd.sh verify
./docker/phase2-amd.sh shell
```

Inside the runtime, confirm the selected ORT install, MIGraphX provider, GPU
target, and asset root with:

```bash
phase2 env --verify
phase2 assets status
```

The AMD Apptainer SIF intentionally has no `uv` and no PyTorch RT-DETR export
environment. Do not run the RT-DETR exporter from this shell. If the formal
ONNX is missing, build it in the external export environment described in
`rtdetrv2-validation.md`, then import the resulting bytes here. The commands
below are the complete no-`uv` in-container asset gate:

```bash
phase2 assets status
phase2 assets \
  --model-profile rtdetrv2_r50 \
  --execution-provider migraphx
```

An X-AnyLabeling compatibility asset is not the formal `rtdetrv2_r50` model;
do not rename it or use it to satisfy the formal AMD gate. The formal source
checkout and checkpoint must be supplied in the external export environment
before RT-DETR AMD experiments can start. YOLOv8 Phase 2A/2B can proceed while
that formal asset is pending.

The AMD build profile must keep `BUILD_NITROS_TRANSPORT=OFF` and
`ORT_ENABLE_CUDA=OFF`; MIGraphX is explicitly enabled by the canonical AMD
profile.

## Assets

Import the formal R50 ONNX, the compatible YOLOv8 ONNX, and the R2B tree using
the commands in [the shared runbook](README.md). Inspect all installed assets,
then validate the exact RT-DETR/R2B profile selected by the provider:

```bash
phase2 assets status
phase2 assets --model-profile rtdetrv2_r50 --execution-provider migraphx
```

For the YOLO lane, first import the user asset with profile `yolov8`, then run
`phase2 assets status` and confirm the recorded digest is the expected
compatibility digest. The current helper has no `verify-yolov8` subcommand;
the real-model launch and capture fail before graph startup if the asset is
missing. The helper is offline and refuses missing or mismatched formal
RT-DETR assets.

## Tests and fixed-input capture

Run the complete AMD package set and inspect every test result:

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

launch_test migrated_packages/gpu_ros_rtdetr/test/gpu_ros_std_rtdetr_pol_test.py
launch_test migrated_packages/gpu_ros_yolov8/test/gpu_ros_yolov8_migraphx_pol_test.py
```

Run a warm-up-only probe before recording output, then capture both models:

```bash
CAPTURE_EXECUTION_PROVIDER=migraphx \
CAPTURE_WARMUP_ONLY=1 \
CAPTURE_FIRST_OUTPUT_TIMEOUT_SECONDS=900 \
ros2 run gpu_ros_detection_validation \
  run_amd_phase2a_fixed_input_capture.sh \
  "rtdetr_migraphx_probe_${RUN_ID}"

CAPTURE_EXECUTION_PROVIDER=migraphx \
CAPTURE_WARMUP_ONLY=1 \
CAPTURE_FIRST_OUTPUT_TIMEOUT_SECONDS=900 \
ros2 run gpu_ros_detection_validation \
  run_amd_phase2a_fixed_input_capture.sh \
  yolov8 "yolov8_migraphx_probe_${RUN_ID}"

CAPTURE_EXECUTION_PROVIDER=migraphx \
ros2 run gpu_ros_detection_validation \
  run_amd_phase2a_fixed_input_capture.sh \
  "rtdetr_phase2a_${RUN_ID}"

CAPTURE_EXECUTION_PROVIDER=migraphx \
ros2 run gpu_ros_detection_validation \
  run_amd_phase2a_fixed_input_capture.sh \
  yolov8 "yolov8_phase2a_${RUN_ID}"
```

Use the bag comparator with the formal thresholds appropriate to the
comparison. Record paired/unpaired frames, class equality, IoU, score deltas,
and frame pass rate; do not infer correctness from throughput alone.

## Provider and library audit

Inspect the running process and its ORT mappings. The selected external ORT
install must provide `libonnxruntime.so`, the shared provider library, and the
MIGraphX provider; `/opt/onnxruntime` is not a formal AMD runtime fallback.

A MIGraphX request fails only when no MIGraphX nodes/events are present. A
mixed MIGraphX+CPU graph is accepted when the report includes CPU node names,
count, and percentage. The historical five-node RT-DETR observation is not a
whitelist, and YOLO is not required to have zero CPU nodes.

## Formal Phase 2A benchmark

After tests, fixed-input comparison, provider audit, and library audit pass:

```bash
export R2B_RESULT_FILE="phase2a_amd_${RUN_ID}.json"
export MIGRAPHX_WARMUP_TIMEOUT_SEC=900
launch_test migrated_packages/benchmarks/gpu_ros_rtdetr_phase2a_amd_graph.py
launch_test migrated_packages/benchmarks/gpu_ros_yolov8_phase2a_amd_graph.py
```

Archive each JSON with the command, model/dataset hashes, ORT fingerprint,
runtime target, and repository-state manifest. The old RT-DETR Phase 2A numbers
belong to the pre-formal-export asset and remain historical until this lane is
closed again.
