# AGENTS.md

## Project Overview

This project migrates NVIDIA Isaac ROS object detection packages from the NVIDIA-proprietary stack (TensorRT + NITROS + GXF) to a vendor-neutral runtime (ONNX Runtime) that supports both NVIDIA (CUDA EP) and AMD (ROCm/MIGraphX EP) GPUs.

The active upstream source is `isaac_ros_object_detection` v4.5.0, which
contains detection pipelines for YOLOv8, RT-DETR, Grounding DINO, and
DetectNet. Phase 0 and Phase 1 measurements remain historical 4.4.0 results
unless they are explicitly rerun and labeled as 4.5.0.

## Optional Local Environment

If `inner_docs/environment_conventions.md` exists, read it for machine-local paths,
remote-host roles, and operational commands. The file is intentionally untracked.
It may refine environment-specific details, but the architecture, security, scope,
and quality requirements in this `AGENTS.md` remain authoritative.

## Long-term Goals

0. **Phase 0 (completed)**: Reproduce NVIDIA's official benchmark numbers on our NVIDIA hardware to establish a verified baseline before making any code changes.
1. **Phase 1 (completed)**: Replace `isaac_ros_tensor_rt` with an ONNX Runtime inference node using the project-owned `gpu_ros_tensor_bundle_msgs/msg/TensorBundle` interface. Remove NITROS dependency from decoder nodes. Validate on NVIDIA GPU and profile four configurations (2x2 matrix of inference backend and transport). The historical NVIDIA/reference lane may cross the old interface through `gpu_ros_nvidia_tensor_bundle_compat`:
   - (A) TensorRT + NITROS (baseline)
   - (B) TensorRT + standard ROS 2 interfaces (isolate NITROS overhead)
   - (C) ONNX Runtime + NITROS (isolate inference backend overhead)
   - (D) ONNX Runtime + standard ROS 2 interfaces (target)
2. **Phase 2A (completed)**: Validate the AMD standard ROS 2 paths for RT-DETR and YOLOv8 with ONNX Runtime MIGraphX EP. These paths are the accepted reference implementations for later managed-transport work.
3. **Phase 2B (active, separate sibling repository)**: Build the reusable `gpu_ros_managed` device-buffer TensorBundle transport/runtime layer and integrate it through the owning `ITensorBundleIO` contract in this repository. Phase 2B uses the completed Phase 2A standard ROS 2 paths as its AMD reference.
4. **Phase 3**: Extend the migrated AMD/std ROS 2 pattern to Grounding DINO and other deferred object detection pipelines.

The final Phase 1 NVIDIA results, correctness checks, provider-placement audit,
and interpretation limits are recorded in
[`docs/phase1-results.md`](docs/phase1-results.md). The normalized machine-readable
summary is
[`migrated_packages/benchmark_results/phase1_final_summary_20260722.json`](migrated_packages/benchmark_results/phase1_final_summary_20260722.json).

## Phase 0: Reproduce NVIDIA Official Benchmarks

### Status

**Completed for the two graphs in scope of Phase 1 migration.** Baseline reproduced on NVIDIA A100-SXM4-40GB:

| Graph | Ours (A100) | NVIDIA published (RTX 5090) | Status |
|-------|-------------|------------------------------|--------|
| RT-DETR | 251.03 fps / 13.18 ms @ 30Hz | 444 fps / 11 ms @ 30Hz | ✅ Reproduced |
| Grounding DINO | 70.59 fps / 26.77 ms @ 30Hz | 130 fps / 15 ms @ 30Hz | ✅ Reproduced |
| DetectNet | Not measured | 227 fps / 18 ms @ 30Hz | ⏭️ Skipped (deferred from Phase 1, see Subpackage Status) |

Result JSONs are saved locally (not committed; archived as `rt-detr-baseline.json` / `grounding-dino-baseline.json`).

### Goal

Run the official `isaac_ros_benchmark` test suite for the object detection graphs in Phase 1 migration scope and confirm our hardware produces numbers in the same ballpark as NVIDIA's published data.

### NVIDIA Published Baseline (historical release-4.4)

| Graph | Input | x86_64 + RTX 5090 | AGX Thor T5000 |
|-------|-------|-------------------|----------------|
| DetectNet (PeopleNet) | 544p | 227 fps / 18ms @ 30Hz | 143 fps / 18ms |
| RT-DETR (SyntheticaDETR) | 720p | 444 fps / 11ms @ 30Hz | 188 fps / 12ms |
| Grounding DINO | 544p | 130 fps / 15ms @ 30Hz | 23.4 fps / 50ms |

Source: https://nvidia-isaac-ros.github.io/performance/index.html

Benchmark scripts & result JSONs: https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/tree/release-4.4

### Required Debian Packages (provided by Isaac ROS apt repo)

```
ros-jazzy-isaac-ros-rtdetr-benchmark           # RT-DETR benchmark + isaac_ros_rtdetr graph deps
ros-jazzy-isaac-ros-grounding-dino-benchmark   # Grounding DINO benchmark + isaac_ros_grounding_dino graph deps
ros-jazzy-isaac-ros-detectnet-benchmark        # DetectNet uses Triton, not TensorRT. Install only if running DetectNet.
```

Each `*-benchmark` package transitively pulls in the corresponding graph implementation
(`isaac_ros_rtdetr` etc.), the inference backend (`isaac_ros_tensor_rt` for RT-DETR/GDino,
`isaac_ros_triton` for DetectNet), preprocessing nodes (`isaac_ros_dnn_image_encoder`,
`isaac_ros_image_proc`, `isaac_ros_tensor_proc`), the `isaac_ros_benchmark` framework with
`NitrosPlaybackNode`, and the underlying NITROS / GXF runtime.

### Required Models & Datasets

| Benchmark | External model asset | External dataset asset |
|-----------|-------------|---------------|
| RT-DETR | `nvidia/isaac/synthetica_detr:1.0.0_onnx` → `models/sdetr/sdetr_grasp.onnx` (FP16 TRT engine generated on first run) | `nvidia/isaac/r2bdataset2024:1` → `datasets/r2b_dataset/r2b_robotarm` |
| Grounding DINO | `nvidia/tao/grounding_dino:grounding_dino_swin_tiny_commercial_deployable_v1.0` → rename to `models/grounding_dino/grounding_dino_model.onnx` | (reuses `r2b_robotarm`) |
| DetectNet | `nvidia/tao/peoplenet` (`deployable_quantized_onnx_v2.6.3`): `resnet34_peoplenet.onnx` + `resnet34_peoplenet_int8.txt` + `config.pbtxt` + `labels.txt` | `nvidia/isaac/r2bdataset2023:2` → `datasets/r2b_dataset/r2b_hallway` |

### Reproduction Procedure

See [`docs/phase0-benchmark-reproduction.md`](docs/phase0-benchmark-reproduction.md)
for the actual commands and gotchas.

### Running Benchmarks

```bash
# RT-DETR benchmark
launch_test src/isaac_ros_benchmark/benchmarks/isaac_ros_rtdetr_benchmark/scripts/isaac_ros_rtdetr_graph.py

# DetectNet benchmark
launch_test src/isaac_ros_benchmark/benchmarks/isaac_ros_detectnet_benchmark/scripts/isaac_ros_detectnet_graph.py

# Grounding DINO benchmark
launch_test src/isaac_ros_benchmark/benchmarks/isaac_ros_grounding_dino_benchmark/scripts/isaac_ros_grounding_dino_graph.py
```

### Success Criteria

- Both in-scope benchmarks, RT-DETR and Grounding DINO, run to completion without errors
- Measured FPS is within ±15% of NVIDIA's published numbers (accounting for GPU hardware differences)
- Results JSON files are saved for future comparison against our ported pipeline

### Notes

- NVIDIA does **not** publish YOLOv8 benchmark results. No official baseline exists for that pipeline.
- The benchmark framework uses `NitrosPlaybackNode` to feed data and auto-tunes publisher rate to find peak throughput.
- TRT engine files are generated on first run via `trtexec`; first run will be slow
- Our hardware GPU model will differ from NVIDIA's test rigs; document the delta

## Phase 2A: AMD Standard ROS 2 Paths

### Scope

Phase 2A is complete in this repository. It covers two accepted AMD standard ROS 2 paths:

RT-DETR:

```
Image -> RtDetrImageEncoderNode (std ROS 2)
      -> RtDetrPreprocessorNode (project-owned TensorBundle)
      -> OnnxInferenceNode (transport=std, execution_provider=migraphx)
      -> RtDetrDecoderNode (std ROS 2)
      -> Detection2DArray
```

YOLOv8:

```
Image -> YoloV8ImageEncoderNode (std ROS 2)
      -> OnnxInferenceNode (transport=std, execution_provider=migraphx)
      -> YoloV8DecoderNode (std ROS 2)
      -> Detection2DArray
```

Phase 2A does not reproduce the Phase 1 A/B/C/D matrix on AMD. AMD benchmarking measures the standard ROS 2 plus ONNX Runtime MIGraphX paths above. See [`docs/phase2a-results.md`](docs/phase2a-results.md) for the accepted implementation and results.

### Requirements

- Use `gpu_ros_tensor_bundle_msgs/msg/TensorBundle` on every canonical project
  path. Do not add NVIDIA's `TensorShape`, rank, stride, or GXF enum fields to
  the project message.
- `gpu_ros_nvidia_tensor_bundle_compat` is the only project package allowed to
  depend on `isaac_ros_tensor_list_interfaces`. It is an explicit NVIDIA-only
  boundary and is excluded from AMD colcon profiles. NVIDIA reference launches
  must show the adapter at the old/new message boundary; AMD launches must not
  resolve it.
- ONNX Runtime must be discoverable through `ONNXRUNTIME_ROOT`, or through explicit `ONNXRUNTIME_INCLUDE_DIR` and `ONNXRUNTIME_LIBRARY` CMake parameters.
- Every AMD build, validation, capture, and benchmark run must use the project-built external ONNX Runtime for its selected GPU target. The SIF copy at `/opt/onnxruntime` is legacy build-only content and is never a formal AMD runtime fallback.
- Build the AMD Phase 2A profile with `-DBUILD_NITROS_TRANSPORT=OFF`, `-DORT_ENABLE_CUDA=OFF`, `-DORT_ENABLE_ROCM=OFF`, `-DORT_ENABLE_MIGRAPHX=ON`, and `-DBUILD_MIGRAPHX_POL_TEST=ON`.
- Build AMD/CPU standard ROS 2 paths with `-DBUILD_NITROS_TRANSPORT=OFF` when Isaac ROS NITROS/CUDA packages are not present. If `execution_provider:=migraphx` is requested without `-DORT_ENABLE_MIGRAPHX=ON`, the node must fail clearly instead of silently falling back to CPU.
- The AMD RT-DETR default is the fixed-revision Apache-2.0 RT-DETRv2 R50
  ONNX export documented in `docs/phase2a-experiment.md`; SyntheticaDETR is
  retained only for the historical NVIDIA/reference profile.
- The repository does not contain, download, or export YOLOv8 weights. YOLOv8 is an optional asset supplied by the user through the explicit local import command. Compatibility is determined by the canonical ONNX SHA-256 recorded in the runbook.

### Benchmark And Validation

- Run the RT-DETR and YOLOv8 Phase 2A AMD benchmark scripts for the standard ROS 2 plus MIGraphX graphs and archive JSON results outside Git.
- Compare AMD `Detection2DArray` output against the corresponding NVIDIA reference bag using `migrated_packages/gpu_ros_detection_validation/scripts/compare_detection2d_bags.py`. Prefer stamp matching when source timestamps are preserved. Use index matching only when the input order is confirmed and timestamps are not comparable.
- Audit ORT profiles for MIGraphX kernel events, CPU fallback placement, and the selected external ORT libraries. A MIGraphX run fails the provider audit only when no MIGraphX kernel events exist. Mixed MIGraphX/CPU placement is accepted and must report CPU node names, counts, and proportions.
- The five CPU postprocessor nodes in one MI350X RT-DETR closure profile and the no-CPU YOLOv8 profile are observations for those tested ORT/model combinations, not correctness contracts or node allowlists.

## Phase 2B: Managed Device TensorBundle Transport Runtime

Phase 2B is the active follow-on phase and is implemented in the independent sibling `gpu_ros_managed` repository. This application consumes the manually selected sibling checkout through colcon and owns only ONNX Runtime provider, binding, synchronization, and staging policy. The sibling provides:

- AMD device-buffer TensorBundle transport.
- Same-process zero-copy transport for component containers.
- Future cross-process transport once the device-memory ownership and synchronization contract is defined.
- Component/container helper APIs for wiring AMD TensorBundle publishers and subscribers.

Phase 2B does not contain object detection model logic, RT-DETR/YOLO/Grounding DINO decoders, detection postprocessing, model-specific preprocessing, or a GXF graph runtime replacement. The existing Phase 2A `transport=std` paths remain independently supported.

## Architecture

### Original pipeline (NVIDIA-only)
```
Image → dnn_image_encoder (NITROS) → TensorRTNode (NITROS) → DecoderNode (NITROS) → Detection2DArray
```

### Target pipeline (vendor-neutral)
```
Image → ImageEncoder (std ROS 2) → OnnxInferenceNode (std ROS 2 TensorBundle) → DecoderNode (std ROS 2 TensorBundle) → Detection2DArray
```

### Phase 2A AMD target pipelines

RT-DETR:

```
Image → RtDetrImageEncoderNode → RtDetrPreprocessorNode → OnnxInferenceNode(transport=std, EP=MIGraphX) → RtDetrDecoderNode → Detection2DArray
```

YOLOv8:

```
Image → YoloV8ImageEncoderNode → OnnxInferenceNode(transport=std, EP=MIGraphX) → YoloV8DecoderNode → Detection2DArray
```

## Subpackage Status

| Package | Status | Follow-on |
|---------|----------------------|-------------------|
| `gpu_ros_rtdetr` | AMD Phase 2A completed | Use as the standard ROS 2 reference for Phase 2B |
| `gpu_ros_yolov8` | AMD Phase 2A completed | Use as the standard ROS 2 reference for Phase 2B |
| `isaac_ros_grounding_dino` | Future extension | Reuse the AMD standard ROS 2 pattern after Phase 2B |
| `isaac_ros_detectnet` | Deferred | Full GXF graph remains outside the current migration scope |
| `gxf_isaac_detectnet` | Deferred | Pure GXF component remains outside the current migration scope |

## Replacements Used by the Migrated Paths

- `isaac_ros_nitros` / `isaac_ros_managed_nitros`: standard ROS 2 subscriber
- `isaac_ros_nitros_tensor_list_type`: external NVIDIA/NITROS TensorList types,
  used only by the NVIDIA reference lane and compatibility boundary
- `isaac_ros_tensor_rt` (TensorRTNode): ONNX Runtime inference node
- `isaac_ros_dnn_image_encoder` (NITROS-based): simple OpenCV-based image encoder node
- `cudaMemcpyAsync` in decoders: host-memory tensor access (data already on host in std msg)
- `gpu_ros_tensor_bundle_msgs/msg/TensorBundle`: canonical project message;
  `gpu_ros_nvidia_tensor_bundle_compat` converts it to/from the external
  NVIDIA `isaac_ros_tensor_list_interfaces/msg/TensorList` only at the
  NVIDIA boundary.

## Build System

- ROS 2 (Humble or later)
- `ament_cmake` build type
- C++17
- Dependencies: `onnxruntime`, `OpenCV`, `rclcpp`, `vision_msgs`, and
  `gpu_ros_tensor_bundle_msgs`. The old NVIDIA interface is not a general
  dependency; it is isolated to the compatibility package.

## Development Notes

- The upstream `isaac_ros_object_detection` repo is kept as-is in `./isaac_ros_object_detection/` for reference.
- YOLOv8 model files are user-provided external assets. Do not instruct maintainers to download or export them. The canonical ONNX identity and SHA-256 are documented in `docs/phase2a-experiment.md`.

## Profiling Plan

Phase 1 compares four configurations on the same NVIDIA GPU (2×2 matrix):

| Config | Inference Backend | Transport | Purpose |
|--------|------------------|-----------|---------|
| (A) | TensorRT | NITROS | Baseline: original performance |
| (B) | TensorRT | Standard ROS 2 TensorBundle | Isolate NITROS overhead |
| (C) | ONNX Runtime (CUDA EP) | NITROS | Isolate inference backend overhead |
| (D) | ONNX Runtime (CUDA EP) | Standard ROS 2 TensorBundle | Target vendor-neutral stack |

Metrics:
- End-to-end latency (image in → detections out)
- Inference-only latency (tensor in → tensor out)
- Throughput (FPS at sustained load)
- GPU memory usage

Phase 2A AMD profiling measured the RT-DETR and YOLOv8 target paths: ONNX Runtime MIGraphX plus standard ROS 2 TensorBundle. Report those results beside the historical Phase 1 NVIDIA configurations, with the hardware and backend differences stated explicitly. Do not interpret the cross-GPU numbers as a single-variable backend comparison, and do not create an AMD A/B/C/D matrix in this repository.
