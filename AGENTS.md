# AGENTS.md

## Project scope

This repository contains the migrated ROS 2 object-detection application. The
maintained implementations are RT-DETRv2 and YOLOv8 under
`migrated_packages/`. Standard ROS 2 transport uses the project-owned
`gpu_ros_tensor_bundle_msgs/msg/TensorBundle`; the managed device-buffer
runtime is maintained in the sibling `gpu_ros_managed` repository.

The current validation target is ROS 2 Jazzy on Ubuntu 24.04 with ROCm 7.1.1,
ONNX Runtime 1.23.1 at the revision in `config/onnxruntime.lock`, and the
selected MIGraphX GPU target. Other ROS, ROCm, GPU, and ORT combinations are
not verified. The upstream reference is
[NVIDIA-ISAAC-ROS/isaac_ros_object_detection](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_object_detection).
Pinned external NVIDIA sources are described by
`external/nvidia-isaac-ros.repos` and the reference manifests.

If `inner_docs/environment_conventions.md` exists, read it for machine-local
paths, host roles, and operational commands. It is intentionally untracked
and may refine environment details, but it does not override this document's
architecture, security, scope, or quality rules.

`gpu_ros_rtdetr` and `gpu_ros_yolov8` are implemented here. Grounding DINO and
DetectNet remain deferred upstream scope; their names in historical Phase 0/1
records do not mean that this repository can build or launch them.

## Architecture and dependency boundaries

The canonical AMD standard path is:

```text
Image -> encoder/preprocessor -> TensorBundle -> ONNX Runtime/MIGraphX
      -> TensorBundle -> decoder -> Detection2DArray
```

The managed path runs the graph in one component process and passes owned HIP
buffers through the sibling `gpu_ros_managed` transport. It does not promise
that the complete ROS graph or model execution performs no copies.

`gpu_ros_nvidia_tensor_bundle_compat` is the only package allowed to depend on
`isaac_ros_tensor_list_interfaces`. It exists for NVIDIA/reference comparison
and is excluded from AMD build profiles. Do not add NVIDIA TensorList or GXF
fields to the project TensorBundle. AMD builds disable NITROS and CUDA and
enable MIGraphX. The project-built ORT installation, selected for the target
GPU, is required; `/opt/onnxruntime` in an image is legacy build content, not
an AMD runtime fallback.
The application owns provider selection, I/O binding, synchronization, and
staging policy. The sibling transport does not own model logic, preprocessing,
decoding, or provider policy. Phase 2A `transport=std` remains independently
supported.

Discover ORT through `ONNXRUNTIME_ROOT`, or set
`ONNXRUNTIME_INCLUDE_DIR` and `ONNXRUNTIME_LIBRARY` explicitly. The canonical
AMD configuration uses `-DBUILD_NITROS_TRANSPORT=OFF`,
`-DORT_ENABLE_CUDA=OFF`, `-DORT_ENABLE_ROCM=OFF`,
`-DORT_ENABLE_MIGRAPHX=ON`, and `-DBUILD_MIGRAPHX_POL_TEST=ON`. A MIGraphX
request without `ORT_ENABLE_MIGRAPHX=ON` must fail clearly rather than fall
back to CPU.

The RT-DETRv2 R50 contract uses `images` and `orig_target_sizes` inputs and
produces `labels`, `boxes`, and `scores`. Keep these names and the configured
shape/type contract aligned with `config/model-profiles.json`.


When changing the owning `ITensorBundleIO` contract, update the sibling
managed repository and this repository's callers together. Keep model-specific
preprocessing and decoders here; do not move them into the transport layer.

## Models and data

The repository does not redistribute model weights, checkpoints, ONNX files,
engines, datasets, bags, traces, profiles, logs, or SIF images. The asset tool
works offline and imports user-held files:

```bash
export OVG_ASSETS_ROOT=/absolute/path/to/external-assets
./tools/phase2-assets status
./tools/phase2-assets import-model --profile rtdetrv2_r50 \
  --source /absolute/path/to/rtdetrv2_r50.onnx
```

Use the model contracts and hashes in `config/model-profiles.json` and
[RT-DETRv2 validation](docs/experiments/rtdetrv2-validation.md). A hash is a
compatibility check, not a license grant. Apache-2.0 applies to identified
upstream source code, not automatically to checkpoint or ONNX bytes.
SyntheticaDETR is a historical NVIDIA/reference profile and remains subject to
its model terms. Do not add download or export steps for YOLOv8 weights.

## Phase history and runbooks

Phase 0 ran the in-scope NVIDIA RT-DETR and Grounding DINO benchmarks and
recorded a local A100 baseline. This is not a same-hardware reproduction of
NVIDIA's published RTX 5090 performance. The historical measurements, package
and asset tables, and commands are retained in
[Phase 1 NVIDIA](docs/experiments/phase1-nvidia.md#phase-0-published-baseline).
Use [the experiment runbooks](docs/experiments/README.md) for the current
procedures. Private machine-readable
result JSONs are not published with this repository.

Phase 1's NVIDIA A/B/C/D matrix and detailed profiling remain historical
reference work. Phase 2A measures AMD standard ROS 2 plus MIGraphX for RT-DETR
and YOLOv8. Phase 2B covers managed transport in the sibling repository. Do
not compare A100, RTX 5090, and MI350X values as if they isolate one backend;
retain the hardware and software identities. A ±15% criterion applies only to
same-hardware comparisons or to a documented normalization.

Use these entry points rather than duplicating historical tables here:

- [Phase 2A AMD](docs/experiments/phase2a-amd.md)
- [Phase 2B managed](docs/experiments/phase2b-managed.md)
- [Phase 1 NVIDIA](docs/experiments/phase1-nvidia.md)
- [Published result summaries](docs/results/)

## Development and verification

Keep external checkouts, assets, and result output outside the worktree. For
AMD work, follow [the runtime contract](docs/experiments/amd-runtime-contract.md)
and use the launcher or the explicit README build command. The direct build
must retain explicit `--base-paths migrated_packages ../gpu_ros_managed` and
select the AMD package set; do not include the NVIDIA compatibility package.

Before requesting hardware, use the static and Python checks in the experiment
index. In a built ROS workspace, run the package-specific `colcon test` command
and inspect it with `colcon test-result --all --verbose`. For an AMD provider
run, verify MIGraphX kernel events and record the external ORT fingerprint. A
MIGraphX run fails the provider audit only when no MIGraphX kernel events exist.
Mixed MIGraphX/CPU placement is allowed, but the report must list CPU node
names, counts, and proportions. The historical five-CPU-node RT-DETR result is
an observation, not an allowlist, and YOLO is not required to have zero CPU
nodes. Compare detections with
`migrated_packages/gpu_ros_detection_validation/scripts/compare_detection2d_bags.py`.

Use stamp matching when source timestamps are preserved. Use index matching
only when input order is confirmed and timestamps are not comparable. Do not
silently fall back from MIGraphX to CPU. Never treat an incomplete profiler
record as a zero-copy result; report it as `INCONCLUSIVE`.

## Security and data handling

Do not commit credentials, private paths, hostnames, scheduler identifiers,
model bytes, bags, traces, or deployment metadata. Keep private result details
outside Git and publish only the provenance needed to interpret a result.
Read `CONTRIBUTING.md` and `SECURITY.md` for repository reporting and disclosure
rules. Do not weaken provider checks, digest checks, permission checks, or
asset-license boundaries to make a run pass.

## Pull requests and changes

Keep changes scoped to the package or runbook being updated. Preserve recorded
experiment numbers and clearly label historical NVIDIA reference results.
Document hardware, GPU target, ROS/ROCm/MIGraphX/ORT identities, model/data
hashes, container identity, and application/sibling revisions in result
archives. Do not claim support for Grounding DINO or DetectNet from historical
references alone.
