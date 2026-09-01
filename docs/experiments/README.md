# Experiment runbooks

These runbooks are for the current remediation tip. Run them in the appropriate
runtime environment, not in the WSL source-only checkout. The public
[AMD runtime contract](amd-runtime-contract.md) is the canonical definition of
that environment; scheduler and deployment wrappers are optional adapters.
The runbooks are ordered so that a failed cheap gate does not waste a long
benchmark allocation.

## Run order

1. Read the [AMD runtime contract](amd-runtime-contract.md) when running an
   AMD lane; prepare external sources, assets, and the exact external ORT
   install.
2. Run environment preflight, clean build, package tests, and static tests.
3. Validate the formal RT-DETRv2 export and provider parity.
4. Capture fixed-input outputs and complete numeric/provider/copy audits.
5. Run the formal performance matrices.
6. Write the result archive and update the result report.

| Runbook | Runtime/backend | Formal lanes |
| --- | --- | --- |
| [Phase 1 NVIDIA](phase1-nvidia.md) | NVIDIA, Isaac ROS 4.5-compatible image | RT-DETR A_fp32/B_fp32/C/D; YOLOv8 A/B/C/D |
| [Phase 2B NVIDIA](phase2b-nvidia.md) | NVIDIA CUDA + Isaac ROS 4.5 | RT-DETR and YOLOv8 Config C versus Managed transport |
| [Phase 2A AMD](phase2a-amd.md) | AMD ROCm + MIGraphX, standard ROS 2 | RT-DETR and YOLOv8 standard paths |
| [Phase 2B Managed](phase2b-managed.md) | ROCm + Managed HIP | RT-DETR and YOLOv8 direct, staged-control, and matrix lanes |
| [RT-DETRv2 validation](rtdetrv2-validation.md) | CPU export environment plus provider runtime | export reproducibility, CPU parity, provider parity, same-bag, optional COCO |

## Shared preparation

Set paths to directories outside the repository. Do not put model bytes,
checkpoints, bags, ORT installs, or result output under the Git worktree.

```bash
export APP_ROOT=/absolute/path/to/amd_ros_object_detection
export MANAGED_ROOT=/absolute/path/to/gpu_ros_managed
export ASSETS_ROOT=/absolute/path/to/external-assets
export RESULTS_ROOT=/absolute/path/to/external-results/<run-id>
```

### NVIDIA external sources

The manifest is tracked at `${APP_ROOT}/external/nvidia-isaac-ros.repos`.
Import it into a directory outside `${APP_ROOT}` and verify the two pinned
commits:

```bash
"${APP_ROOT}/tools/bootstrap-nvidia-external.sh" \
  --source-root /absolute/path/to/nvidia-external
```

The bootstrap is required only for the NVIDIA lane. A fresh clone of this
repository does not need `--recursive`.

### Assets

The asset helper is offline and never downloads a model or dataset:

```bash
export OVG_ASSETS_ROOT="${ASSETS_ROOT}"
"${APP_ROOT}/tools/phase2-assets" status
"${APP_ROOT}/tools/phase2-assets" import-model \
  --profile rtdetrv2_r50 \
  --source /absolute/path/to/rtdetrv2_r50.onnx
"${APP_ROOT}/tools/phase2-assets" import-model \
  --profile nvidia_synthetica \
  --source /absolute/path/to/sdetr_grasp.onnx
"${APP_ROOT}/tools/phase2-assets" import-model \
  --profile xanylabeling_rtdetrv2_r50 \
  --source /absolute/path/to/rtdetrv2_r50vd_6x_coco.onnx
"${APP_ROOT}/tools/phase2-assets" import-model \
  --profile yolov8 \
  --source /absolute/path/to/yolov8s.onnx
"${APP_ROOT}/tools/phase2-assets" import-r2b \
  --source /absolute/path/to/r2b_robotarm
```

Only import the profiles required by the selected lane. The formal R50 and
X-AnyLabeling imports check their known SHA-256 values. The user-provided YOLO
asset is checked for compatibility and provenance, not granted a
redistribution license.

For a formal R50 rebuild, use the command in
[rtdetrv2-validation.md](rtdetrv2-validation.md). Do not use `prepare` as a
download command: it only verifies the selected model and R2B tree.

### AMD external ORT

AMD formal work requires a pristine recursive ORT checkout at the exact commit
in `config/onnxruntime.lock` and a completed install produced by
`tools/build-phase2a-external-ort.sh`. Set `OVG_ORT_ROOT` to the resulting
fingerprinted install before `phase2-amd.sh preflight`, `colcon`, capture, or
benchmark commands. The public runtime contract documents the mounts and
device interfaces needed to make that install visible; a scheduler-specific
environment file is not a prerequisite.

The ORT build is intentionally separate from the application build. Record
the install fingerprint and `build-info.txt` in the result archive.

## Common software gates

Run the repository's static and Python checks from the application root before
requesting hardware:

```bash
git diff --check
uv run --with pytest --no-project python -m pytest -q \
  migrated_packages/gpu_ros_detection_validation/test
uv run --with pytest --no-project python -m pytest -q \
  tools/test_open_source_namespace_boundaries.py
```

The two `uv` commands above are host/source-environment checks, not commands
for the AMD Apptainer SIF. That SIF intentionally has no `uv`. Inside it, run
the package tests with `colcon test` from the phase runbook and run the
stdlib-only boundary check directly when needed:

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

The tests cover managed lifecycle/orphan behavior, device selection, staging
limits, NITROS exception boundaries, YOLO class-aware NMS/float boxes, result
comparison, capture topology, and manifest metadata. They do not replace a
real-model benchmark.

## Result discipline

Use a new output name for every run. Never overwrite a bag, profile, trace,
manifest, or benchmark JSON. Keep raw outputs in `${RESULTS_ROOT}` and copy
only the small summary/provenance record into the result folder.

For an incomplete profiler record, report `INCONCLUSIVE`; do not convert an
unresolved copy record into a zero-copy claim. Mixed MIGraphX+CPU provider
placement is allowed when MIGraphX kernels are present and the CPU node list,
count, and share are reported.

The result archive is expected to retain both sensitive deployment metadata
(privately) and public scientific/software provenance. Public summaries must
include the hardware model, GPU target, ROS/ROCm/MIGraphX/ORT identities,
container recipe or image digest, model/data hashes, and application/sibling
revisions. Hostnames, user names, scheduler IDs, and deployment paths remain
private.
