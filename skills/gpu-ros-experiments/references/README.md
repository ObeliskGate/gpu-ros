# Experiment runbooks

These are the experiment methods for the checked-out GPU ROS revision. They
can be read directly without installing the [agent skill](../SKILL.md).
GPU commands require the selected runtime; a source-only checkout can run the
[standalone managed build](../../../gpu_ros_managed/README.md#build-the-standalone-library)
and CPU checks.

AMD defaults to Docker. The [runtime contract](amd-runtime-contract.md) also
defines an explicit Apptainer branch for an authorized compute allocation.
NVIDIA retains its outer Isaac ROS workspace. Reuse verified dependencies and
assets, then run cheap checks before a requested benchmark.

## Run order

1. Read the [AMD runtime contract](amd-runtime-contract.md) for an AMD lane, or
   the NVIDIA runbook for the outer-workspace and external-source contract.
2. Inspect and reuse external sources, assets, and the exact ORT install.
   Prepare separate fresh build/install/log/result state.
3. Run environment preflight, a clean project build, package checks, and the
   applicable static checks.
4. Validate the formal RT-DETRv2 export/provider contract when that profile is
   selected.
5. Capture fixed-input outputs and complete the requested numeric,
   provider, or copy audits.
6. Run a formal performance matrix only when explicitly requested.
7. Write the result archive and update a result summary without moving raw
   artifacts into Git.

| Runbook | Runtime/backend | Formal lanes |
| --- | --- | --- |
| [Phase 1 NVIDIA](phase1-nvidia.md) | NVIDIA, Isaac ROS 4.5-compatible image | RT-DETR A/B/C/D; YOLOv8 A/B/C/D |
| [Phase 2B NVIDIA](phase2b-nvidia.md) | NVIDIA CUDA + Isaac ROS 4.5 | RT-DETR and YOLOv8 Config C versus managed transport |
| [Phase 2A AMD](phase2a-amd.md) | AMD ROCm + MIGraphX, standard ROS 2 | RT-DETRv2 and YOLOv8 standard paths |
| [Phase 2B Managed](phase2b-managed.md) | AMD ROCm + managed HIP | RT-DETRv2 and YOLOv8 direct, staged-control, and matrix lanes |
| [RT-DETRv2 validation](rtdetrv2-validation.md) | External export environment plus provider runtime | export reproducibility, CPU parity, provider parity, same-bag, optional COCO |
| [Detection validation](detection-validation.md) | Prepared AMD or NVIDIA runtime | capture, strict comparison, audit, HIP probe, and lifecycle evidence |
| [Result acceptance](result-acceptance.md) | All lanes | identity, correctness, reproduction, and promotion rules |

## Shared preparation

Set paths to directories outside the repository. Do not put model bytes,
checkpoints, bags, ORT installs, or result output under the Git worktree:

```bash
export REPO_ROOT=/absolute/path/to/gpu-ros
export ASSETS_ROOT=/absolute/path/to/external-assets
export RESULTS_ROOT="/absolute/path/to/external-results/${RUN_ID}"
```

The AMD runtime mounts this checkout at `/workspaces/gpu-ros` and uses that path
for both `GPU_ROS_REPO_ROOT` and `OVG_WORKSPACE_ROOT`. The NVIDIA runtime keeps
`/workspaces/isaac_ros-dev` as its outer colcon workspace and mounts this
checkout at `/workspaces/isaac_ros-dev/src/gpu-ros`. Do not create a second
transport checkout or source bind.

### NVIDIA external sources

The manifest is tracked at `${REPO_ROOT}/gpu_ros_object_detection/external/nvidia-isaac-ros.repos`.
Bootstrap its pinned checkouts outside the repository and under the NVIDIA
outer workspace's `src/nvidia_external` when running that lane:

```bash
"${REPO_ROOT}/gpu_ros_object_detection/tools/bootstrap-nvidia-external.sh" \
  --source-root /absolute/path/to/nvidia-external
```

The bootstrap is required only when the pinned external sources are not already
available. A fresh clone of this repository does not need `--recursive`.

### Assets

The asset helper is offline and never downloads a model or dataset:

```bash
export OVG_ASSETS_ROOT="${ASSETS_ROOT}"
"${REPO_ROOT}/gpu_ros_object_detection/tools/phase2-assets" status
"${REPO_ROOT}/gpu_ros_object_detection/tools/phase2-assets" import-model \
  --profile rtdetrv2_r50 \
  --source /absolute/path/to/rtdetrv2_r50.onnx
"${REPO_ROOT}/gpu_ros_object_detection/tools/phase2-assets" import-model \
  --profile nvidia_synthetica \
  --source /absolute/path/to/sdetr_grasp.onnx
"${REPO_ROOT}/gpu_ros_object_detection/tools/phase2-assets" import-model \
  --profile yolov8 \
  --source /absolute/path/to/yolov8s.onnx
"${REPO_ROOT}/gpu_ros_object_detection/tools/phase2-assets" import-r2b \
  --source /absolute/path/to/r2b_robotarm
```

Only import profiles required by the selected lane. The formal RT-DETRv2
import checks its known SHA-256. User-provided YOLOv8 assets are checked for
compatibility and provenance, not granted a redistribution license. The exact
RT-DETR export procedure is in [rtdetrv2-validation.md](rtdetrv2-validation.md).

### AMD external ORT

AMD formal work requires a pristine recursive ORT checkout at the exact commit
in `gpu_ros_object_detection/config/onnxruntime.lock` and a completed install produced by
`gpu_ros_object_detection/tools/build-phase2a-external-ort.sh`. Set `OVG_ORT_ROOT` to the resulting
fingerprinted install before `phase2-amd.sh preflight`, `colcon`, capture, or
benchmark commands. The public runtime contract documents the mounts and
device interfaces needed to make that install visible; a scheduler-specific
environment file is not a prerequisite.

The ORT build is intentionally separate from the application build. Record the
install fingerprint and `build-info.txt` in the external result archive.

## Common software gates

Run the source checks from the repository root, using the existing CPU CI
selection rather than collecting ROS-dependent tests:

```bash
uv run --no-project --isolated --python 3.12 \
  python tools/test_open_source_namespace_boundaries.py
ROCPROF_RUN_FULL_TRACE_REGRESSION=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONDONTWRITEBYTECODE=1 uv run --no-project --isolated --python 3.12 \
  --with pytest==9.0.3 python -m pytest -q -p no:cacheprovider \
  gpu_ros_object_detection/tools/test_edit_isaac_ros_tensor_list_interfaces_package.py \
  gpu_ros_object_detection/gpu_ros_onnx_inference/test/test_summarize_ort_profile.py \
  gpu_ros_object_detection/gpu_ros_onnx_inference/test/test_compare_nsys_cuda_traces.py \
  gpu_ros_object_detection/gpu_ros_onnx_inference/test/test_compare_nvidia_copy_traces.py \
  gpu_ros_object_detection/gpu_ros_onnx_inference/test/test_analyze_rocprof_traces.py \
  gpu_ros_object_detection/gpu_ros_detection_validation/test/test_capture_runner.py
```

These `uv` commands are host/source-environment checks, not commands for the AMD
Apptainer SIF. Inside the SIF, run package tests with `colcon` from the phase
runbook and run the standard-library boundary check directly when needed:

```bash
python tools/test_open_source_namespace_boundaries.py
```

RT-DETR export and CPU/MIGraphX Python parity use explicitly supplied external
interpreters; see `rtdetrv2-validation.md`. Never substitute a CPU-only Python
ORT wheel for the MIGraphX provider gate.

On a built ROS environment, run the package-specific `colcon test` commands
from the phase runbook and inspect the complete result with:

```bash
colcon test-result --all --verbose
```

These checks cover managed lifecycle/orphan behavior, device selection, staging
limits, compatibility boundaries, YOLO class-aware NMS/float boxes, result
comparison, capture topology, and provenance metadata. They do not replace a
real-model benchmark or a fixed-input smoke.

## Result discipline

Use a new output name for every run. Never overwrite a bag, profile, trace,
manifest, or benchmark JSON. Keep raw outputs in `${RESULTS_ROOT}` and copy only
the small summary/provenance record into a result folder.

For an incomplete profiler or copy record, report `INCONCLUSIVE`; do not
convert unresolved evidence into a zero-copy claim. Mixed MIGraphX plus CPU
placement is allowed when MIGraphX kernels are present and the CPU node list,
count, and share are reported.

Record wrapper exit, component lifecycle, numeric comparison, copy evidence,
and throughput separately. A zero wrapper exit can hide SIGSEGV during
teardown. A report may lack a requested fixed-rate field. Record the crash or
missing field directly, without changing it to PASS or zero. Follow
[result acceptance](result-acceptance.md), and show measured values and
historical comparability before updating a result summary.

A new matrix manifest is schema 3 and records one `monorepo_revision` plus the
existing monorepo diff, untracked-path/content, and dirty-state fields. A new
fixed-input capture remains an unversioned command log with
`monorepo_revision` and `monorepo_worktree_diff_sha256`; it does not gain a
matrix schema marker. New promoted summaries use `monorepo_revision` instead
of separate application/managed heads and also remain unversioned. See
[`../../../docs/results/README.md`](../../../docs/results/README.md) for legacy v2/capture handling.

Public summaries must include hardware model, GPU target, ROS/ROCm/provider/ORT
identities, container recipe or image/SIF digest, model/data hashes, command,
status, and the monorepo revision. Hostnames, usernames, scheduler IDs, and
deployment paths remain private.
