# Phase 2A AMD runbook

## Scope

Phase 2A is the standard ROS 2 TensorBundle path on AMD with ONNX Runtime
MIGraphX. It covers RT-DETRv2 R50 and user-provided YOLOv8. It does not build
the NVIDIA compatibility package and does not enable NITROS/CUDA.

The RT-DETR default is the formal `rtdetrv2_r50` profile. Its exact ONNX SHA
and model/checkpoint license evidence are in `config/model-profiles.json`.

## Environment and external ORT

Use the public [AMD runtime contract](amd-runtime-contract.md). The one source
checkout is `/workspaces/gpu-ros` inside the runtime, and the launcher exports
both `GPU_ROS_REPO_ROOT` and `OVG_WORKSPACE_ROOT` to that path. Transport and
application packages are built from this checkout; no second transport source
or sibling state is required.

Select Docker or Apptainer explicitly and provide the externally built ORT
install:

```bash
export OVG_RUNTIME=docker                 # or apptainer
export OVG_STATE_ROOT=/absolute/path/to/phase2-state
export OVG_ORT_STATE_HOST=/absolute/path/to/phase2-state/ort
export OVG_ORT_ROOT=/workspaces/ovg-ort/install/<ort-fingerprint>
export AMD_GPU_TARGETS=<runtime-target>    # use the target required by runtime
# Apptainer only:
# export OVG_APPTAINER_SIF=/absolute/path/to/validated-runtime.sif
```

Scheduler variables are an optional site adapter: `OVG_REQUIRE_SLURM=1` makes
the launcher's allocation check fatal, while the default only warns.
`/dev/kfd` and `/dev/dri` are always required for an AMD GPU run. Do not place
scheduler setup, private host details, or deployment paths in this runbook.

Build the external ORT separately from a pristine recursive checkout at the
locked commit, then run the application environment gates from the repository
root:

```bash
./docker/phase2-amd.sh preflight
./docker/phase2-amd.sh up
./docker/phase2-amd.sh colcon
./docker/phase2-amd.sh verify
./docker/phase2-amd.sh shell
```

Inside the runtime, confirm the selected ORT install, MIGraphX provider, GPU
target, and asset root:

```bash
cd /workspaces/gpu-ros
phase2 env --verify
phase2 assets status
```

The AMD Apptainer SIF intentionally has no `uv` and no PyTorch RT-DETR export
environment. Do not run the RT-DETR exporter from this shell. If the formal
ONNX is missing, build it in the external export environment described in
[`rtdetrv2-validation.md`](rtdetrv2-validation.md), then import the resulting
bytes here. The complete no-`uv` in-container asset gate is:

```bash
phase2 assets status
phase2 assets prepare \
  --model-profile rtdetrv2_r50 \
  --execution-provider migraphx
```

An X-AnyLabeling compatibility asset is not the formal `rtdetrv2_r50` model;
do not rename it or use it to satisfy the formal AMD gate. The formal source
checkout and checkpoint must be supplied in the external export environment
before RT-DETR AMD experiments can start. YOLOv8 Phase 2A/2B can proceed while
that formal asset is pending.

The AMD build profile keeps `BUILD_NITROS_TRANSPORT=OFF` and
`ORT_ENABLE_CUDA=OFF`; MIGraphX is explicitly enabled by the canonical profile.

## Assets

Import the formal R50 ONNX, compatible YOLOv8 ONNX, and R2B tree using the
commands in the [shared runbook](README.md). Inspect installed assets, then
validate the exact RT-DETR/R2B profile selected by the provider:

```bash
phase2 assets status
phase2 assets prepare --model-profile rtdetrv2_r50 --execution-provider migraphx
```

For the YOLO lane, first import the user asset with profile `yolov8`, then run
`phase2 assets status` and confirm the recorded digest is the expected
compatibility digest. The helper has no `verify-yolov8` subcommand; the
real-model launch and capture fail before graph startup if the asset is missing.
The helper is offline and refuses missing or mismatched formal RT-DETR assets.

## Tests and fixed-input capture

Run the selected AMD package set and inspect every test result:

```bash
cd /workspaces/gpu-ros
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

For a managed capture, add `CAPTURE_TRANSPORT=managed`; standard Phase 2A
captures use the default standard transport. Use the bag comparator with the
formal thresholds appropriate to the comparison. Record paired/unpaired frames,
class equality, IoU, score deltas, and frame pass rate; do not infer
correctness from throughput alone.

## Provider and library audit

Inspect the running process and its ORT mappings. The selected external ORT
install must provide `libonnxruntime.so`, the shared provider library, and the
MIGraphX provider; `/opt/onnxruntime` is not a formal AMD runtime fallback.

A MIGraphX request fails only when no MIGraphX nodes/events are present. A
mixed MIGraphX plus CPU graph is accepted when the report includes CPU node
names, count, and percentage. The historical five-node RT-DETR observation is
not a whitelist, and YOLO is not required to have zero CPU nodes.

## Formal Phase 2A benchmark

After tests, fixed-input comparison, provider audit, and library audit pass:

```bash
cd /workspaces/gpu-ros
export R2B_RESULT_FILE="phase2a_amd_${RUN_ID}.json"
export MIGRAPHX_WARMUP_TIMEOUT_SEC=900
launch_test evaluation/benchmarks/gpu_ros_rtdetr_phase2a_amd_graph.py
launch_test evaluation/benchmarks/gpu_ros_yolov8_phase2a_amd_graph.py
```

Archive each JSON with the command, model/dataset hashes, ORT fingerprint,
runtime target, and monorepo-state manifest. Matrix manifests written by the
current tree use schema 3 and one `monorepo_revision`; the fixed-input capture
log is unversioned and uses `monorepo_revision` plus
`monorepo_worktree_diff_sha256`. Historical records are not rewritten.
