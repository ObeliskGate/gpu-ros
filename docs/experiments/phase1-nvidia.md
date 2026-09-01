# Phase 1 NVIDIA runbook

This is the NVIDIA reference campaign for the Isaac ROS 4.5-compatible
runtime. It is separate from the AMD/CPU runbooks and does not use Apptainer.
Keep source checkouts, model files, the R2B bag, external checkouts, plans,
traces, and reports outside the repository.

## Matrix and acceptance boundary

The formal throughput matrix is:

| Lane | RT-DETR | YOLOv8 |
| --- | --- | --- |
| A | TensorRT FP32 + NITROS | TensorRT FP16 + NITROS |
| B | TensorRT FP32 + standard ROS 2 boundary | TensorRT FP16 + standard ROS 2 boundary |
| C | ORT CUDA + NITROS | ORT CUDA + NITROS |
| D | ORT CUDA + standard ROS 2 | ORT CUDA + standard ROS 2 |
| Managed | Managed RT-DETR comparison with C | Not part of this Phase 1 matrix |

`rtdetrv2_r50` is not used for this historical NVIDIA/reference matrix;
CUDA `auto` selects `nvidia_synthetica`. The selected asset must exist and
pass its digest check. There is no silent model fallback.

The NVIDIA proof-of-life is the ONNX inference POL plus the transport probe.
The AMD/ROCm-only POL is not a Phase 1 gate.

### RT-DETR coordinate policy

`orig_target_sizes` is a small `1x2` `int64` input. It scales output boxes
into image coordinates, so both fixed-input lanes must use the same policy.
It is not a meaningful throughput knob; keep the official benchmark graph
unchanged for performance reproduction.

## Software identity

Record these values in the external archive for every run:

| Item | Policy |
| --- | --- |
| Runtime image | Use the pinned Isaac ROS 4.5-compatible image; store its immutable digest externally. |
| Application | Record the exact repository revision used by the runner. |
| Managed sibling | Record the exact selected sibling revision; no commit-pair allowlist. |
| External object detection | Use the revision pinned by `external/nvidia-isaac-ros.repos`. |
| External benchmark | Use the revision pinned by `external/nvidia-isaac-ros.repos`. |
| ORT | Record the selected 1.23.1 source/library identity. |

Do not put image, host, driver, or deployment identifiers in this document.

## Fresh runtime layout

Use a new run root and keep all external state out of the source tree. The
repository has no submodules, so the main clone is deliberately non-recursive.

```bash
export RUN_ROOT=/absolute/path/to/external/phase1-<run-id>
export APP_ROOT="${RUN_ROOT}/amd_ros_object_detection"
export MANAGED_ROOT="${RUN_ROOT}/gpu_ros_managed"
export NVIDIA_EXTERNAL_ROOT="${RUN_ROOT}/nvidia-external"
export NVIDIA_ASSETS_ROOT="${RUN_ROOT}/assets"
export NVIDIA_RESULTS_ROOT="${RUN_ROOT}/results"
export COMPOSE_PROJECT_NAME="ovg-phase1-<run-id>"

git clone --branch opensource-prep --single-branch <amd-repository-url> "${APP_ROOT}"
git clone --branch opensource-prep --single-branch <managed-repository-url> "${MANAGED_ROOT}"
git -C "${APP_ROOT}" rev-parse HEAD
git -C "${MANAGED_ROOT}" rev-parse HEAD
test -z "$(git -C "${APP_ROOT}" status --porcelain)"
test -z "$(git -C "${MANAGED_ROOT}" status --porcelain)"
```

The runner records committed revisions, staged/unstaged diff hashes,
untracked paths/content hashes, and sibling state. Dirty state is evidence,
not a benchmark gate.

## External sources and assets

The tracked `external/nvidia-isaac-ros.repos` file is a source manifest, not a
submodule. Bootstrap the pinned checkouts outside the application tree:

```bash
"${APP_ROOT}/tools/bootstrap-nvidia-external.sh" \
  --source-root "${NVIDIA_EXTERNAL_ROOT}"

export OVG_NVIDIA_EXTERNAL_SOURCE_ROOT="${NVIDIA_EXTERNAL_ROOT}"
export GPU_ROS_MANAGED_DIR="${MANAGED_ROOT}"
```

Acquire the Synthetica/YOLO assets and R2B bag with the offline import tools.
Record their hashes in the external archive. Keep the source and asset roots
separate from Git and never destroy an older asset volume to make room for a
new run.

## Build and package gates

```bash
cd "${APP_ROOT}"
./docker/phase1-nvidia.sh bootstrap
./docker/phase1-nvidia.sh colcon
./docker/phase1-nvidia.sh verify
```

Inside the runtime, source ROS and the workspace, then run the complete package
test suite and inspect every result. Run the CUDA proof and transport probe
before throughput tests:

```bash
colcon test --event-handlers console_direct+
colcon test-result --all --verbose
launch_test migrated_packages/gpu_ros_onnx_inference/test/gpu_ros_onnx_rtdetr_pol_test.py
launch_test migrated_packages/benchmarks/gpu_ros_rtdetr_transport_probe.py
```

The exact package counts and tool versions belong in the external archive.

## Formal throughput runs

Run each graph as a separate process with core dumps disabled. Preserve the
benchmark JSON even when a teardown fault occurs, but label its release status
`INCONCLUSIVE` until teardown is clean.

```bash
ulimit -c 0

launch_test migrated_packages/benchmarks/gpu_ros_rtdetr_config_a_fp32_graph.py
launch_test migrated_packages/benchmarks/gpu_ros_rtdetr_config_b_fp32_graph.py
launch_test migrated_packages/benchmarks/gpu_ros_rtdetr_config_c_graph.py
launch_test migrated_packages/benchmarks/gpu_ros_rtdetr_config_d_graph.py
launch_test migrated_packages/benchmarks/gpu_ros_rtdetr_managed_graph.py

launch_test migrated_packages/benchmarks/gpu_ros_yolov8_config_a_graph.py
launch_test migrated_packages/benchmarks/gpu_ros_yolov8_config_b_graph.py
launch_test migrated_packages/benchmarks/gpu_ros_yolov8_config_c_graph.py
launch_test migrated_packages/benchmarks/gpu_ros_yolov8_config_d_graph.py
```

Copy each emitted `r2b-log-*.json` into the immutable external result archive
after its graph exits. Record the command, runtime image digest, revisions,
asset hashes, throughput values, and teardown status there.

The B standard-transport boundary is intentionally explicit:

```text
std TensorBundle -> TensorBundleBridge (managed device buffer) ->
NITROS TensorRT -> NvidiaTensorListToTensorBundle -> std decoder
```

## Historical A reference

The unmodified Isaac ROS 4.5 RT-DETR reference reported a predicted peak of
`342.578 Hz`, mean peak output of `328.127 fps`, and zero misses at fixed
10/30/60 fps. The ordinary A wrapper reported `334.844 Hz`, `326.752 fps`,
and zero fixed-rate misses. These are historical observations, not a device
or release claim; retain the source JSON and environment metadata externally.

## Fixed-input and provider/copy evidence

Use fixed-input scripts only for detection equivalence. Capture the reference,
Managed candidate, and audit data with the same input bag, image dimensions,
model files, and RT-DETR size policy:

```bash
./migrated_packages/gpu_ros_detection_validation/scripts/run_nvidia_fixed_input_capture.sh \
  rtdetr-c rtdetr-c-<run-id>
./migrated_packages/gpu_ros_detection_validation/scripts/run_nvidia_fixed_input_capture.sh \
  rtdetr-managed rtdetr-managed-<run-id>
./migrated_packages/gpu_ros_detection_validation/scripts/run_nvidia_transport_audit.sh \
  rtdetr rtdetr-transport-<run-id>
./migrated_packages/gpu_ros_detection_validation/scripts/run_nvidia_c_vs_d_copy_audit.sh \
  rtdetr rtdetr-c-vs-d-<run-id>
```

The provider audit must contain CUDA activity. CPU shape/decoder bookkeeping
is reported, not rejected by a fixed node-count rule. A requested provider
with no provider activity is a failure. YOLO fixed-input parity is a separate
follow-up, not part of the formal Phase 1 table.

## Historical observations

The historical throughput rows below are retained as context only. “Peak” is
the benchmark's predicted maximum publisher rate; fixed trials show measured
rate and missed frames.

| Lane | Peak Hz | Mean fps | Peak misses | Fixed 10 / 30 / 60 fps |
| --- | ---: | ---: | ---: | --- |
| RT-DETR A_fp32 | 242.031 | 233.604 | 43/1210 | 10, 30, 60: 0 misses |
| RT-DETR B_fp32 | 149.219 | 142.629 | 35/746 | 10.204/0 · 30.248/0 · 60.263/0 |
| RT-DETR C | 102.813 | 100.485 | 4/514 | 10, 30, 60: 0 misses |
| RT-DETR D | 87.344 | 83.725 | 10.667/436 | 10, 30, 60: 0 misses |
| RT-DETR Managed | 102.813 | 99.113 | 10.667/514 | 10, 30, 60: 0 misses |
| YOLOv8 A | 141.484 | 136.294 | 20.333/707 | 10.206/0 · 30.209/0 · 60.207/0 |
| YOLOv8 B | 218.828 | 210.632 | 43/1094 | 10.207/0 · 30.240/0 · 60.328/0 |
| YOLOv8 C | 141.484 | 141.601 | 0/707 | 10.208/0 · 30.208/0 · 60.266/0 |
| YOLOv8 D | 141.484 | 139.303 | 4.667/707 | 10.216/0 · 30.209/0 · 60.187/0 |

The C/D/Managed historical processes wrote valid JSON but later failed during
teardown. Mark those rows `INCONCLUSIVE` for clean-release purposes. Do not
turn valid JSON into a clean pass solely because throughput was measured.

## Archive checklist

For each lane archive the raw report, command line, exit/teardown status,
application and Managed revisions/diff/untracked hashes, external commits,
runtime image identity, model/bag hashes, and provider/copy reports. Keep the
archive outside Git and keep sensitive deployment metadata out of these documents.
