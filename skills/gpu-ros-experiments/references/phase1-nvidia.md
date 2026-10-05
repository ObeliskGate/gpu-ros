# Phase 1 NVIDIA runbook

This is the NVIDIA reference campaign for an official Isaac ROS 5.0/Lyrical image.
It is separate from the AMD/CPU runbooks and does not use Apptainer. Keep
source checkouts, model files, the R2B bag, external checkouts, plans, traces,
and reports outside the repository.

## Workspace layout

NVIDIA deliberately keeps an outer Isaac ROS colcon workspace and this
repository below `src/gpu-ros`:

```text
/workspaces/isaac_ros-dev/
├── src/gpu-ros/                 this repository
├── src/nvidia_external/         optional image-paired source overlay
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
| A | TensorRT FP32 + native TensorList | TensorRT FP16 + native TensorList |
| B | TensorRT FP32 + standard ROS 2 boundary | TensorRT FP16 + standard ROS 2 boundary |
| C | ORT CUDA + native TensorList | ORT CUDA + native TensorList |
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
| Runtime image | Use the verified official Isaac ROS 5.0 image; its digest and paired ROS packages are the version authority. |
| Monorepo | Record the exact `monorepo_revision`, tracked-diff hash, untracked paths/content hash, and dirty state. |
| External object detection | Prefer the image binary; any required source overlay must match the image's verifiable source provenance. |
| External benchmark | Prefer image-paired benchmark packages, not a separately selected upstream release head. |
| ORT | Record the selected 1.30.0 source/library identity and the locked CUDA 13 distribution hash. |
| Model/data | Record the selected profile and asset/dataset hashes. |

Do not put host, user, scheduler, device-instance, or deployment-path
identifiers in this document. Runtime image, driver, and provider versions are
scientific provenance and belong in the public result record when captured;
they are not deployment secrets.

Runtime acceptance must cite an exercised, identified target image and preserve
independent failure statuses; see the [current campaign](../../../docs/results/README.md#isaac-ros-50-migration-campaign-2026-10-05).
A historical 4.5/Jazzy result retains its original identity and does not validate
the new stack. Missing image identity blocks runtime verification, not
source-only checks. Set `GPU_ROS_NVIDIA_PROFILE=1` before colcon discovery.

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

Use image-installed binary packages first. The tracked
`gpu_ros_object_detection/external/nvidia-isaac-ros.repos` is an optional
standard vcstool source manifest, not a submodule or a verified 5.0 lock.
Its empty mapping blocks bootstrap until image-paired exact revisions are
supplied. If a source overlay is necessary:

```bash
export ISAAC_ROS_BASE_IMAGE="${VERIFIED_ISAAC_ROS_5_IMAGE_WITH_DIGEST:?required}"
"${REPO_ROOT}/gpu_ros_object_detection/tools/bootstrap-nvidia-external.sh" \
  --source-root "${NVIDIA_EXTERNAL_ROOT}"
```

The tracked Compose file does not bind old external source trees. Add any
verified image-paired overlay explicitly in the run-specific override, under
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
native TensorList TensorRT -> NvidiaTensorListToTensorBundle -> std decoder
```

## Fixed-input and provider/copy evidence

Use fixed-input scripts to collect detection-comparison observations. Capture
the reference, managed candidate, and audit data with the same input bag, image
dimensions, model files, and RT-DETR size policy. Run from the NVIDIA outer workspace:

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
with no provider activity is a failure. YOLO fixed-input detection reporting is
a separate observation, not part of the formal Phase 1 table.

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
