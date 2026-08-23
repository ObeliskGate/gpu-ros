# amd_ros_object_detection

Maintainer: Boshen Chen (@ObeliskGate)

This repository contains ROS 2 object-detection experiments and reusable
application code for RT-DETR and YOLOv8. It compares standard ROS 2,
NVIDIA/NITROS, and AMD/HIP plus ONNX Runtime/MIGraphX paths while keeping the
model and benchmark assets outside the source tree.

This is an independent project, not an NVIDIA official project. Some migrated
files are derived from file-level Apache-2.0 NVIDIA Isaac ROS sources; the
upstream copyright and local modification notices are retained where that
provenance is established. NVIDIA package and submodule licenses do not become
the license of this repository as a whole.

## Scope and architecture

The public application packages are under `migrated_packages/`:

- `gpu_ros_detection_common` contains image preprocessing shared by the
  standard and Managed HIP paths.
- `gpu_ros_onnx_inference` owns the transport-neutral ONNX Runtime session,
  provider selection, I/O Binding, tensor contracts, and optional diagnostics.
- `gpu_ros_rtdetr` and `gpu_ros_yolov8` provide standard ROS 2
  encoders/decoders and the direct Managed HIP nodes.
- `gpu_ros_tensor_bundle_msgs` defines the project-owned, contiguous tensor
  wire format used by the canonical standard ROS 2 paths.
- `gpu_ros_nvidia_tensor_bundle_compat` is an optional NVIDIA-only boundary
  package. It is the only project package that depends on
  `isaac_ros_tensor_list_interfaces` and is not part of an AMD build.
- `gpu_ros_detection_validation` provides source-level tests and offline
  comparison/profiling tools.
- The package and namespace migration table is in
  [`docs/open-source-migration.md`](docs/open-source-migration.md). It lists
  supported names only; no old project-name aliases are installed.
- The reusable `gpu_ros_managed_tensor_bundle` package is maintained in the
  sibling `gpu_ros_managed` repository and must be checked out explicitly.
- `migrated_packages/benchmarks/` contains the A/B/C/D, Phase 2A, and Phase 2B
  graph definitions. It does not contain benchmark result artifacts.

The Phase 2B AMD production topology is direct and same-process:

```text
Image -> Managed HIP preprocessing -> Managed ORT/MIGraphX
      -> Managed HIP decoder -> Detection2DArray
```

The staged-control graphs deliberately add standard TensorBundle materialization
around ORT so that adapter cost can be measured separately. NVIDIA comparison
graphs use the external NITROS components and are not silently relabeled as
AMD results.

## TensorBundle interface and NVIDIA boundary

`gpu_ros_tensor_bundle_msgs/msg/TensorBundle` is the canonical application
interface. Its tensors carry a project-owned dtype value, an `int64[]` shape,
and contiguous C-order payload bytes. It intentionally does not reproduce the
wire schema of NVIDIA's TensorList messages, including their rank, stride, or
GXF enum fields.

The compatibility package converts between the two schemas at an explicit
boundary, validates contiguous layouts, and preserves the old rank/shape/stride
semantics only while crossing that boundary. NVIDIA reference launches may use
that adapter; AMD profiles and the standard project launches must not resolve
or install it. There are no compatibility aliases retaining the old project
package names.

## Compatibility

The current experiment baseline is ROS 2 Jazzy with Isaac ROS 4.5-compatible
interfaces. AMD inference uses ROCm/HIP and ONNX Runtime `v1.23.1` with
MIGraphX; the exact ORT source commit used by the patch workflow is
`d9b2048791efb5804fe3d53a04b4971256addebf`. NVIDIA comparison paths use CUDA,
NITROS, and the pinned external benchmark components. `gpu_ros_managed` is a
separate sibling repository used by the Managed transport packages; build both
repositories from compatible, explicitly recorded commits in one workspace.

The source does not promise ABI compatibility across arbitrary ROS, CUDA,
ROCm, driver, or ONNX Runtime versions. The active container profiles and
patches are the compatibility contract for this experiment.

## Build and tests

The container and Apptainer entry points are in `docker/`, `apptainer/`, and
`docker-compose.phase2a-amd.yaml`. A prepared ROS workspace can build the
application with the repository's colcon defaults:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --merge-install
colcon test --merge-install --event-handlers console_direct+
colcon test-result --all --verbose
```

For a source-only review, the useful checks are XML/editor tests, Python unit
tests, shell syntax checks, graph import/compile checks, and documentation
link checks. GPU benchmark execution is intentionally outside this
remediation and requires the external runtime and assets described below.

The namespace and dependency boundary audit can be run without ROS 2:

```bash
uv run --isolated --no-project python -B \
  tools/test_open_source_namespace_boundaries.py
```

## Models and data

The repository does not vendor or redistribute ONNX weights,
TensorRT/MIGraphX engines, R2B data, ROS bags, traces, profiles, logs, SIF
images, or raw benchmark JSON. The AMD default is the Apache-2.0 RT-DETRv2
R50 COCO export at
`models/rtdetrv2_r50vd_6x_coco/rtdetrv2_r50vd_6x_coco.onnx`, pinned to the
upstream source revision recorded in the Phase 2A runbook. The NVIDIA
Synthetica DETR asset remains only in the historical NVIDIA/reference profile.
The RT-DETRv2 image encoder follows that export's validation preprocessing and
converts uint8 RGB values to float32 in `[0, 1]`; the decoder and tensor
contract remain unchanged.
YOLOv8 uses a user-provided YOLOv8 ONNX asset. Model terms and dataset terms
are independent of this source-code license. A recorded hash is only a
compatibility check, not permission to redistribute an asset.

Generated output belongs under an explicit out-of-tree result directory such
as `OVG_RESULTS_ROOT`; the source tree is not an experiment database.

## License and third parties

The root `LICENSE` covers project-authored source and the intended public
migrated code subject to the component boundaries in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). ORT patches are covered by
the copied MIT notice in `LICENSES/ONNXRUNTIME-MIT.txt` when applied to the
external ORT source. The `isaac_ros_object_detection` and
`isaac_ros_benchmark` directories are pinned external gitlinks and retain
their own upstream licenses; they are not re-licensed by this root repository.

## Known limitations

- Direct AMD Managed HIP results require an external ORT build, compatible
  ROCm/MIGraphX runtime, and external model/data assets.
- The AMD RT-DETRv2 asset must satisfy the documented `images`,
  `orig_target_sizes` -> `labels`, `boxes`, `scores` contract; a raw
  Transformers RT-DETR export is not interchangeable with this path.
- RT-DETR has a documented MIGraphX CPU placement for part of its
  postprocessing; provider placement must be audited separately from
  throughput.
- Standard and Managed paths are complete pipeline configurations, not a
  decoder-neutral single-variable benchmark.
- A source checkout is not a sanitized public snapshot. Existing private
  history contains older experimental metadata; future publication requires a
  manual sanitized snapshot as described in the remediation report.
