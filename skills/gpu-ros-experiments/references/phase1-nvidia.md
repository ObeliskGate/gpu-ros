# Phase 1 NVIDIA runbook

This is the NVIDIA reference campaign for the Isaac ROS 4.5-compatible runtime.
It is separate from the AMD/CPU runbooks and does not use Apptainer. Keep
source checkouts, model files, the R2B bag, external checkouts, plans, traces,
and reports outside the repository.

## Workspace layout

NVIDIA deliberately keeps an outer Isaac ROS colcon workspace and this
repository below `src/gpu-ros`:

```text
/workspaces/isaac_ros-dev/
├── src/gpu-ros/                 this repository
├── src/nvidia_external/         pinned external Isaac ROS checkouts
├── build/                       outer-workspace build state
├── install/                     outer-workspace install state
└── log/                         outer-workspace logs
```

The repository is mounted once by Compose at
`/workspaces/isaac_ros-dev/src/gpu-ros`. `GPU_ROS_REPO_ROOT` points to that
path; `OVG_WORKSPACE_ROOT` is not redefined to mean the NVIDIA outer workspace.
The launcher defaults the repository path from
`${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}/src/gpu-ros` when the override is
not supplied.

## Matrix and acceptance boundary

The formal throughput matrix is:

| Lane | RT-DETR | YOLOv8 |
| --- | --- | --- |
| A | TensorRT FP32 + NITROS | TensorRT FP16 + NITROS |
| B | TensorRT FP32 + standard ROS 2 boundary | TensorRT FP16 + standard ROS 2 boundary |
| C | ORT CUDA + NITROS | ORT CUDA + NITROS |
| D | ORT CUDA + standard ROS 2 | ORT CUDA + standard ROS 2 |
| Managed | Managed RT-DETR comparison with C | Not part of this Phase 1 matrix |

This reference matrix uses `nvidia_synthetica`, not the AMD `rtdetrv2_r50`
profile. CUDA `auto` selects that NVIDIA profile. Verify actual model bytes and
independent provenance. If the profile has no fixed expected SHA, do not claim
a known-digest check passed. There is no model fallback.

The NVIDIA proof-of-life is the ONNX inference POL plus the transport probe.
The AMD/ROCm-only POL is not a Phase 1 gate.

### RT-DETR coordinate policy

`orig_target_sizes` is a small `1x2` `int64` input. It scales output boxes
into image coordinates, so both fixed-input lanes must use the same policy. It
is not a meaningful throughput knob; keep the official benchmark graph
unchanged for performance reproduction.

## Software identity

Record these values in the external archive for every run:

| Item | Policy |
| --- | --- |
| Runtime image | Use the pinned Isaac ROS 4.5-compatible image; store its immutable digest externally. |
| Monorepo | Record the exact `monorepo_revision`, tracked-diff hash, untracked paths/content hash, and dirty state. |
| External object detection | Use the revision pinned by `gpu_ros_object_detection/external/nvidia-isaac-ros.repos`. |
| External benchmark | Use the revision pinned by `gpu_ros_object_detection/external/nvidia-isaac-ros.repos`. |
| ORT | Record the selected 1.23.1 source/library identity. |
| Model/data | Record the selected profile and asset/dataset hashes. |

Do not put host, user, scheduler, device-instance, or deployment-path
identifiers in this document. Runtime image, driver, and provider versions are
scientific provenance and belong in the public result record when captured;
they are not deployment secrets.

## Fresh runtime layout

Use a new run root and keep all external state out of the source tree. The
repository has no submodules, so the main clone is deliberately non-recursive.
For a local checkout, the outer workspace and repository paths are:

```bash
export RUN_ROOT="/absolute/path/to/external/phase1-${RUN_ID}"
export NVIDIA_WORKSPACE_ROOT="${RUN_ROOT}/isaac_ros-dev"
export REPO_ROOT="${NVIDIA_WORKSPACE_ROOT}/src/gpu-ros"
export NVIDIA_EXTERNAL_ROOT="${NVIDIA_WORKSPACE_ROOT}/src/nvidia_external"
export NVIDIA_ASSETS_ROOT="${NVIDIA_WORKSPACE_ROOT}/assets"
export NVIDIA_RESULTS_ROOT="${NVIDIA_WORKSPACE_ROOT}/results"
export COMPOSE_PROJECT_NAME="gpu-ros-phase1-${RUN_ID}"

mkdir -p "${NVIDIA_WORKSPACE_ROOT}/src"
git clone --branch main --single-branch https://github.com/gpu-ros/gpu-ros.git \
  "${REPO_ROOT}"
cd "${REPO_ROOT}"
git rev-parse HEAD
```

For an already prepared outer workspace, set `REPO_ROOT` to
`/workspaces/isaac_ros-dev/src/gpu-ros` and do not create another checkout.
Dirty state is evidence, not a benchmark gate; preserve and record it.

## External sources and assets

The tracked `gpu_ros_object_detection/external/nvidia-isaac-ros.repos` is a
source manifest, not a submodule. Reuse matching checkouts; bootstrap only
when they are missing:

```bash
"${REPO_ROOT}/gpu_ros_object_detection/tools/bootstrap-nvidia-external.sh" \
  --source-root "${NVIDIA_EXTERNAL_ROOT}"

export OVG_NVIDIA_EXTERNAL_SOURCE_ROOT="${NVIDIA_EXTERNAL_ROOT}"
```

The Compose file mounts those checkouts read-only under
`/workspaces/isaac_ros-dev/src/nvidia_external`. Acquire Synthetica/YOLO assets
and the R2B bag with the offline import tools. Record their hashes in the
external archive. Keep source and asset roots separate from Git and never
destroy an older asset volume to make room for a new run.

## Build and package gates

Use the verified image, external assets, fresh state, and non-interactive
Compose execution described in the [NVIDIA managed method](phase2b-nvidia.md#reuse-the-dependency-image).
Keep its run-specific override on every command. Do not enter `shell` for
acceptance or source an old project overlay into a clean build.

Run that method's package tests. The additional Phase 1 CUDA proof and
transport probe are separate runtime payloads from the outer workspace:

```bash
launch_test \
  src/gpu-ros/gpu_ros_object_detection/gpu_ros_onnx_inference/test/gpu_ros_onnx_rtdetr_pol_test.py
launch_test \
  src/gpu-ros/gpu_ros_object_detection/benchmarks/gpu_ros_rtdetr_transport_probe.py
```

The exact package counts and tool versions belong in the external archive.

## Formal throughput runs

Run only when this performance scope is explicitly requested. Each graph is
a separate process with core dumps disabled. Preserve JSON after a teardown
fault, but keep its release status INCONCLUSIVE until teardown is clean:

```bash
cd /workspaces/isaac_ros-dev
ulimit -c 0

launch_test \
  src/gpu-ros/gpu_ros_object_detection/benchmarks/gpu_ros_rtdetr_config_a_fp32_graph.py
launch_test \
  src/gpu-ros/gpu_ros_object_detection/benchmarks/gpu_ros_rtdetr_config_b_fp32_graph.py
launch_test \
  src/gpu-ros/gpu_ros_object_detection/benchmarks/gpu_ros_rtdetr_config_c_graph.py
launch_test \
  src/gpu-ros/gpu_ros_object_detection/benchmarks/gpu_ros_rtdetr_config_d_graph.py
launch_test \
  src/gpu-ros/gpu_ros_object_detection/benchmarks/gpu_ros_rtdetr_managed_graph.py

launch_test \
  src/gpu-ros/gpu_ros_object_detection/benchmarks/gpu_ros_yolov8_config_a_graph.py
launch_test \
  src/gpu-ros/gpu_ros_object_detection/benchmarks/gpu_ros_yolov8_config_b_graph.py
launch_test \
  src/gpu-ros/gpu_ros_object_detection/benchmarks/gpu_ros_yolov8_config_c_graph.py
launch_test \
  src/gpu-ros/gpu_ros_object_detection/benchmarks/gpu_ros_yolov8_config_d_graph.py
```

Copy each emitted `r2b-log-*.json` into the immutable external result archive
after its graph exits. Record the command, runtime image digest, monorepo
revision/state, asset hashes, throughput values, and teardown status there.

The B standard-transport boundary is intentionally explicit:

```text
std TensorBundle -> TensorBundleBridge (managed device buffer) ->
NITROS TensorRT -> NvidiaTensorListToTensorBundle -> std decoder
```

## Fixed-input and provider/copy evidence

Use fixed-input scripts only for detection equivalence. Capture the reference,
managed candidate, and audit data with the same input bag, image dimensions,
model files, and RT-DETR size policy. Run from the NVIDIA outer workspace:

```bash
ros2 run gpu_ros_detection_validation run_nvidia_fixed_input_capture.sh \
  rtdetr-c "rtdetr-c-${RUN_ID}"
ros2 run gpu_ros_detection_validation run_nvidia_fixed_input_capture.sh \
  rtdetr-managed "rtdetr-managed-${RUN_ID}"
ros2 run gpu_ros_detection_validation run_nvidia_transport_audit.sh \
  rtdetr "rtdetr-transport-${RUN_ID}"
ros2 run gpu_ros_detection_validation run_nvidia_c_vs_d_copy_audit.sh \
  rtdetr "rtdetr-c-vs-d-${RUN_ID}"
```

Use the parameter, pairing, and lifecycle rules in
[detection validation](detection-validation.md). Full NVIDIA launch compositions
are owned by `gpu_ros_nvidia_reference`; conversion components remain in compat.

The provider audit must contain CUDA activity. CPU shape/decoder bookkeeping
is reported, not rejected by a fixed node-count rule. A requested provider
with no provider activity is a failure. YOLO fixed-input parity is a separate
follow-up, not part of the formal Phase 1 table.

## Historical results

Historical measurements are retained outside this source release. See the
[historical records](../../../docs/results/README.md#historical-records) section
for archive policy; dated commands and paths are not current build instructions.

## Archive checklist

For each lane archive the raw report, command line, exit/teardown status,
monorepo revision and diff/untracked hashes, external commits, runtime image
identity, model/bag hashes, and provider/copy reports. Keep the archive outside
Git and keep sensitive deployment metadata out of these documents. Matrix
manifests and capture logs follow the separate current-format rules in
[`../../../docs/results/README.md`](../../../docs/results/README.md).
