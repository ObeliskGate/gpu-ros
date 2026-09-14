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

`rtdetrv2_r50` is not used for this historical NVIDIA/reference matrix;
CUDA `auto` selects `nvidia_synthetica`. The selected asset must exist and
pass its digest check. There is no silent model fallback.

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
| External object detection | Use the revision pinned by `external/nvidia-isaac-ros.repos`. |
| External benchmark | Use the revision pinned by `external/nvidia-isaac-ros.repos`. |
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
export RUN_ROOT=/absolute/path/to/external/phase1-<run-id>
export NVIDIA_WORKSPACE_ROOT="${RUN_ROOT}/isaac_ros-dev"
export REPO_ROOT="${NVIDIA_WORKSPACE_ROOT}/src/gpu-ros"
export NVIDIA_EXTERNAL_ROOT="${NVIDIA_WORKSPACE_ROOT}/src/nvidia_external"
export NVIDIA_ASSETS_ROOT="${NVIDIA_WORKSPACE_ROOT}/assets"
export NVIDIA_RESULTS_ROOT="${NVIDIA_WORKSPACE_ROOT}/results"
export COMPOSE_PROJECT_NAME="gpu-ros-phase1-<run-id>"

mkdir -p "${NVIDIA_WORKSPACE_ROOT}/src"
git clone --branch main --single-branch https://github.com/gpu-ros/gpu-ros.git \
  "${REPO_ROOT}"
cd "${REPO_ROOT}"
git rev-parse HEAD
test -z "$(git status --porcelain)"
```

For an already prepared outer workspace, set `REPO_ROOT` to
`/workspaces/isaac_ros-dev/src/gpu-ros` and do not create another checkout.
Dirty state is evidence, not a benchmark gate; preserve and record it.

## External sources and assets

The tracked `external/nvidia-isaac-ros.repos` file is a source manifest, not a
submodule. Bootstrap pinned checkouts outside the repository:

```bash
"${REPO_ROOT}/tools/bootstrap-nvidia-external.sh" \
  --source-root "${NVIDIA_EXTERNAL_ROOT}"

export OVG_NVIDIA_EXTERNAL_SOURCE_ROOT="${NVIDIA_EXTERNAL_ROOT}"
```

The Compose file mounts those checkouts read-only under
`/workspaces/isaac_ros-dev/src/nvidia_external`. Acquire Synthetica/YOLO assets
and the R2B bag with the offline import tools. Record their hashes in the
external archive. Keep source and asset roots separate from Git and never
destroy an older asset volume to make room for a new run.

## Build and package gates

Run these commands from the repository path. Compose keeps the container's
working directory at `/workspaces/isaac_ros-dev`:

```bash
cd "${REPO_ROOT}"
docker compose -f docker-compose.yaml config
./docker/phase1-nvidia.sh up
./docker/phase1-nvidia.sh colcon
./docker/phase1-nvidia.sh verify
./docker/phase1-nvidia.sh shell
```

Inside the runtime, source ROS and the outer workspace, then run the complete
package test suite and inspect every result. Run the CUDA proof and transport
probe before throughput tests:

```bash
cd /workspaces/isaac_ros-dev
colcon test --event-handlers console_direct+
colcon test-result --all --verbose
launch_test src/gpu-ros/perception/gpu_ros_onnx_inference/test/gpu_ros_onnx_rtdetr_pol_test.py
launch_test src/gpu-ros/evaluation/benchmarks/gpu_ros_rtdetr_transport_probe.py
```

The exact package counts and tool versions belong in the external archive.

## Formal throughput runs

Run each graph as a separate process with core dumps disabled. Preserve the
benchmark JSON even when a teardown fault occurs, but label its release status
`INCONCLUSIVE` until teardown is clean:

```bash
cd /workspaces/isaac_ros-dev
ulimit -c 0

launch_test src/gpu-ros/evaluation/benchmarks/gpu_ros_rtdetr_config_a_fp32_graph.py
launch_test src/gpu-ros/evaluation/benchmarks/gpu_ros_rtdetr_config_b_fp32_graph.py
launch_test src/gpu-ros/evaluation/benchmarks/gpu_ros_rtdetr_config_c_graph.py
launch_test src/gpu-ros/evaluation/benchmarks/gpu_ros_rtdetr_config_d_graph.py
launch_test src/gpu-ros/evaluation/benchmarks/gpu_ros_rtdetr_managed_graph.py

launch_test src/gpu-ros/evaluation/benchmarks/gpu_ros_yolov8_config_a_graph.py
launch_test src/gpu-ros/evaluation/benchmarks/gpu_ros_yolov8_config_b_graph.py
launch_test src/gpu-ros/evaluation/benchmarks/gpu_ros_yolov8_config_c_graph.py
launch_test src/gpu-ros/evaluation/benchmarks/gpu_ros_yolov8_config_d_graph.py
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
cd /workspaces/isaac_ros-dev
src/gpu-ros/evaluation/gpu_ros_detection_validation/scripts/run_nvidia_fixed_input_capture.sh \
  rtdetr-c rtdetr-c-<run-id>
src/gpu-ros/evaluation/gpu_ros_detection_validation/scripts/run_nvidia_fixed_input_capture.sh \
  rtdetr-managed rtdetr-managed-<run-id>
src/gpu-ros/evaluation/gpu_ros_detection_validation/scripts/run_nvidia_transport_audit.sh \
  rtdetr rtdetr-transport-<run-id>
src/gpu-ros/evaluation/gpu_ros_detection_validation/scripts/run_nvidia_c_vs_d_copy_audit.sh \
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

These are historical observations, not a current support or reproduction
claim. New runs use the one-checkout provenance contract and must not be
compared across different model, hardware, provider, or image identities.

## Phase 0 reference context

Phase 0 package, asset, and performance details are retained as historical
NVIDIA reference context only. Their commands require the historical NVIDIA
workspace and are not AMD or current GPU ROS build instructions. Do not infer
support for Grounding DINO or DetectNet from those records.

The local A100 run-through measured RT-DETR at 251.03 fps / 13.18 ms and
Grounding DINO at 70.59 fps / 26.77 ms at 30 Hz. NVIDIA's published RTX 5090
and AGX Thor T5000 values were:

| Graph | Input | RTX 5090 | AGX Thor T5000 |
| --- | --- | --- | --- |
| DetectNet (PeopleNet) | 544p | 227 fps / 18 ms | 143 fps / 18 ms |
| RT-DETR (SyntheticaDETR) | 720p | 444 fps / 11 ms | 188 fps / 12 ms |
| Grounding DINO | 544p | 130 fps / 15 ms | 23.4 fps / 50 ms |

Phase 0 package and asset details are NVIDIA benchmark inputs, not migrated
package support. The release-4.4 apt packages were:

```text
ros-jazzy-isaac-ros-rtdetr-benchmark
ros-jazzy-isaac-ros-grounding-dino-benchmark
ros-jazzy-isaac-ros-detectnet-benchmark
```

The first two pulled `isaac_ros_tensor_rt`, `isaac_ros_dnn_image_encoder`,
`isaac_ros_image_proc`, `isaac_ros_tensor_proc`, `isaac_ros_benchmark`,
`NitrosPlaybackNode`, and the NITROS/GXF runtime. DetectNet used
`isaac_ros_triton` rather than TensorRT.

| Benchmark | Model asset | Dataset asset |
| --- | --- | --- |
| RT-DETR | `nvidia/isaac/synthetica_detr:1.0.0_onnx` -> `models/sdetr/sdetr_grasp.onnx` | `nvidia/isaac/r2bdataset2024:1` -> `datasets/r2b_dataset/r2b_robotarm` |
| Grounding DINO | `nvidia/tao/grounding_dino:grounding_dino_swin_tiny_commercial_deployable_v1.0` -> `models/grounding_dino/grounding_dino_model.onnx` | `r2b_robotarm` |
| DetectNet | `nvidia/tao/peoplenet` `deployable_quantized_onnx_v2.6.3`: `resnet34_peoplenet.onnx`, `resnet34_peoplenet_int8.txt`, `config.pbtxt`, `labels.txt` | `nvidia/isaac/r2bdataset2023:2` -> `datasets/r2b_dataset/r2b_hallway` |

The local DetectNet run was not measured. The historical commands were:

```bash
launch_test src/isaac_ros_benchmark/benchmarks/isaac_ros_rtdetr_benchmark/scripts/isaac_ros_rtdetr_graph.py
launch_test src/isaac_ros_benchmark/benchmarks/isaac_ros_detectnet_benchmark/scripts/isaac_ros_detectnet_graph.py
launch_test src/isaac_ros_benchmark/benchmarks/isaac_ros_grounding_dino_benchmark/scripts/isaac_ros_grounding_dino_graph.py
```

The source was NVIDIA's [performance page](https://nvidia-isaac-ros.github.io/performance/index.html)
and the [release-4.4 benchmark scripts](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/tree/release-4.4).
Local result JSONs were not committed; the archived names were
`rt-detr-baseline.json` and `grounding-dino-baseline.json`. These commands
require the historical NVIDIA workspace and are not AMD or current GPU ROS
instructions.

## Archive checklist

For each lane archive the raw report, command line, exit/teardown status,
monorepo revision and diff/untracked hashes, external commits, runtime image
identity, model/bag hashes, and provider/copy reports. Keep the archive outside
Git and keep sensitive deployment metadata out of these documents. Matrix
manifests and capture logs follow the separate current-format rules in
[`../results/README.md`](../results/README.md).
