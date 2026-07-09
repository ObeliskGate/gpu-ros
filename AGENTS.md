# AGENTS.md

## Project Overview

This project migrates NVIDIA Isaac ROS object detection packages from the NVIDIA-proprietary stack (TensorRT + NITROS + GXF) to a vendor-neutral runtime (ONNX Runtime) that supports both NVIDIA (CUDA EP) and AMD (ROCm/MIGraphX EP) GPUs.

The upstream source is `isaac_ros_object_detection` (v4.4.0) which contains detection pipelines for YOLOv8, RT-DETR, Grounding DINO, and DetectNet.

## Long-term Goals

0. **Phase 0 (completed)**: Reproduce NVIDIA's official benchmark numbers on our NVIDIA hardware to establish a verified baseline before making any code changes.
1. **Phase 1**: Replace `isaac_ros_tensor_rt` with an ONNX Runtime inference node using standard ROS2 `isaac_ros_tensor_list_interfaces` messages. Remove NITROS dependency from decoder nodes. Validate on NVIDIA GPU and profile four configurations (2×2 matrix of inference backend × transport):
   - (A) TensorRT + NITROS (baseline)
   - (B) TensorRT + standard ROS2 interfaces (isolate NITROS overhead)
   - (C) ONNX Runtime + NITROS (isolate inference backend overhead)
   - (D) ONNX Runtime + standard ROS2 interfaces (target)
2. **Phase 2a (current)**: Port the RT-DETR target pipeline to AMD using only standard ROS2 transport and ONNX Runtime MIGraphX EP:
   `Image -> std image encoder -> official TensorList -> ONNX Runtime MIGraphX -> std decoder -> Detection2DArray`.
3. **Phase 2b (future, separate repository)**: Build a reusable AMD TensorList transport/runtime layer similar to the transport subset of NITROS. This repository documents the boundary only; it does not implement or depend on Phase 2b.
4. **Phase 3**: Extend the migrated AMD/std ROS2 pattern to Grounding DINO and other deferred object detection pipelines.

## Phase 0: Reproduce NVIDIA Official Benchmarks

### Status

**Completed for the two graphs in scope of Phase 1 migration.** Baseline reproduced on NVIDIA A100-SXM4-40GB (Jetstream2):

| Graph | Ours (A100) | NVIDIA published (RTX 5090) | Status |
|-------|-------------|------------------------------|--------|
| RT-DETR | 251.03 fps / 13.18 ms @ 30Hz | 444 fps / 11 ms @ 30Hz | ✅ Reproduced |
| Grounding DINO | 70.59 fps / 26.77 ms @ 30Hz | 130 fps / 15 ms @ 30Hz | ✅ Reproduced |
| DetectNet | — | 227 fps / 18 ms @ 30Hz | ⏭️ Skipped (deferred from Phase 1, see Subpackage Status) |

Result JSONs are saved locally (not committed; archived as `rt-detr-baseline.json` / `grounding-dino-baseline.json`).

### Goal

Run the official `isaac_ros_benchmark` test suite for the object detection graphs in Phase 1 migration scope and confirm our hardware produces numbers in the same ballpark as NVIDIA's published data.

### NVIDIA Published Baseline (release-4.4)

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
ros-jazzy-isaac-ros-detectnet-benchmark        # DetectNet (uses Triton, not TensorRT) — install only if running DetectNet
```

Each `*-benchmark` package transitively pulls in the corresponding graph implementation
(`isaac_ros_rtdetr` etc.), the inference backend (`isaac_ros_tensor_rt` for RT-DETR/GDino,
`isaac_ros_triton` for DetectNet), preprocessing nodes (`isaac_ros_dnn_image_encoder`,
`isaac_ros_image_proc`, `isaac_ros_tensor_proc`), the `isaac_ros_benchmark` framework with
`NitrosPlaybackNode`, and the underlying NITROS / GXF runtime.

### Required Models & Datasets

| Benchmark | Model (NGC) | Dataset (NGC) |
|-----------|-------------|---------------|
| RT-DETR | `nvidia/isaac/synthetica_detr:1.0.0_onnx` → `models/sdetr/sdetr_grasp.onnx` (FP16 TRT engine generated on first run) | `nvidia/isaac/r2bdataset2024:1` → `datasets/r2b_dataset/r2b_robotarm` |
| Grounding DINO | `nvidia/tao/grounding_dino:grounding_dino_swin_tiny_commercial_deployable_v1.0` → rename to `models/grounding_dino/grounding_dino_model.onnx` | (reuses `r2b_robotarm`) |
| DetectNet | `nvidia/tao/peoplenet` (`deployable_quantized_onnx_v2.6.3`): `resnet34_peoplenet.onnx` + `resnet34_peoplenet_int8.txt` + `config.pbtxt` + `labels.txt` | `nvidia/isaac/r2bdataset2023:2` → `datasets/r2b_dataset/r2b_hallway` |

### Reproduction Procedure

See [`docs/run-benchmark.md`](docs/run-benchmark.md) for the actual commands and gotchas.

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

- All three benchmarks run to completion without errors
- Measured FPS is within ±15% of NVIDIA's published numbers (accounting for GPU hardware differences)
- Results JSON files are saved for future comparison against our ported pipeline

### Notes

- NVIDIA does **not** publish YOLOv8 benchmark results — no official baseline exists for that pipeline
- The benchmark framework uses `NitrosPlaybackNode` to feed data and auto-tunes publisher rate to find peak throughput
- TRT engine files are generated on first run via `trtexec`; first run will be slow
- Our hardware GPU model will differ from NVIDIA's test rigs; document the delta

## Phase 2a: AMD Standard ROS2 RT-DETR Port

### Scope

Phase 2a is implemented in this repository. It covers only the production target path for AMD:

```
Image -> RtDetrImageEncoderNode (std ROS2)
      -> RtDetrPreprocessorNode (official TensorList)
      -> OnnxInferenceNode (transport=std, execution_provider=migraphx)
      -> RtDetrDecoderNode (std ROS2)
      -> Detection2DArray
```

Phase 2a does **not** reproduce the Phase 1 A/B/C/D matrix on AMD. AMD benchmarking only measures the target standard ROS2 + ONNX Runtime MIGraphX path. NVIDIA A100 Phase 1 results remain the reference for performance and offline numeric comparison.

### Requirements

- Use the existing official `isaac_ros_tensor_list_interfaces/msg/TensorList`; do not add a custom TensorList message in this repository.
- ONNX Runtime must be discoverable through `ONNXRUNTIME_ROOT`, or through explicit `ONNXRUNTIME_INCLUDE_DIR` and `ONNXRUNTIME_LIBRARY` CMake parameters.
- Build AMD/CPU standard ROS2 paths with `-DBUILD_NITROS_TRANSPORT=OFF` when Isaac ROS NITROS/CUDA packages are not present.
- Build MIGraphX support explicitly with `-DORT_ENABLE_MIGRAPHX=ON`. If `execution_provider:=migraphx` is requested without that build flag, the node must fail clearly instead of silently falling back to CPU.

### Benchmark And Validation

- Run the Phase 2a AMD benchmark script for the standard ROS2 + MIGraphX graph and archive JSON results under `migrated_packages/benchmark_results/`.
- Compare AMD `Detection2DArray` output against the existing A100 Phase 1 baseline bag using `migrated_packages/isaac_ros_detection_validation/scripts/compare_detection2d_bags.py`.
- Prefer strict same-input validation. If cross-machine timestamps differ, use index-based matching and document that choice in the result notes.

## Phase 2b: Future AMD TensorList Transport Runtime

Phase 2b is intentionally out of scope for this repository and should be developed as a future independent repository. It may provide:

- AMD device-buffer TensorList transport.
- Same-process zero-copy transport for component containers.
- Future cross-process transport once the device-memory ownership and synchronization contract is defined.
- Component/container helper APIs for wiring AMD TensorList publishers and subscribers.

Phase 2b must not contain object detection model logic. It must not implement RT-DETR/YOLO/Grounding DINO decoders, detection postprocessing, model-specific preprocessing, or a GXF graph runtime replacement. Phase 2a must not wait for Phase 2b.

## Architecture

### Original pipeline (NVIDIA-only)
```
Image → dnn_image_encoder (NITROS) → TensorRTNode (NITROS) → DecoderNode (NITROS) → Detection2DArray
```

### Target pipeline (vendor-neutral)
```
Image → ImageEncoder (std ROS2) → OnnxInferenceNode (std ROS2 TensorList) → DecoderNode (std ROS2 TensorList) → Detection2DArray
```

### Phase 2a AMD target pipeline
```
Image → RtDetrImageEncoderNode → RtDetrPreprocessorNode → OnnxInferenceNode(transport=std, EP=MIGraphX) → RtDetrDecoderNode → Detection2DArray
```

## Subpackage Status

| Package | NITROS/GXF Dependency | Migration Priority |
|---------|----------------------|-------------------|
| `isaac_ros_yolov8` | ManagedNitrosSubscriber + CUDA memcpy | **First** — simplest decoder |
| `isaac_ros_rtdetr` | ManagedNitrosSubscriber + CUDA memcpy | Second |
| `isaac_ros_grounding_dino` | ManagedNitrosSubscriber + CUDA memcpy + multi-input | Third |
| `isaac_ros_detectnet` | Full GXF graph (NitrosNode + gxf_isaac_detectnet .so) | Deferred — heavy GXF dependency |
| `gxf_isaac_detectnet` | Pure GXF component | Deferred |

## Key Dependencies to Replace

- `isaac_ros_nitros` / `isaac_ros_managed_nitros` → standard ROS2 subscriber
- `isaac_ros_nitros_tensor_list_type` → `isaac_ros_tensor_list_interfaces/msg/TensorList`
- `isaac_ros_tensor_rt` (TensorRTNode) → new ONNX Runtime inference node
- `isaac_ros_dnn_image_encoder` (NITROS-based) → simple OpenCV-based image encoder node
- `cudaMemcpyAsync` in decoders → host-memory tensor access (data already on host in std msg)
- Custom TensorList messages are out of scope; use `isaac_ros_tensor_list_interfaces/msg/TensorList`.

## Build System

- ROS2 (Humble or later)
- `ament_cmake` build type
- C++17
- Dependencies: `onnxruntime`, `OpenCV`, `rclcpp`, `vision_msgs`, `isaac_ros_tensor_list_interfaces`

## Development Notes

- The upstream `isaac_ros_object_detection` repo is kept as-is in `./isaac_ros_object_detection/` for reference.
- New migrated packages will be created alongside or in a separate directory.
- Use FP32 for initial ONNX Runtime validation to ensure correctness; add FP16 later for fair perf comparison with TensorRT.
- ONNX model files: export from ultralytics YOLOv8 (`yolov8s.onnx` etc.) with opset 17+.

## Profiling Plan

Phase 1 compares four configurations on the same NVIDIA GPU (2×2 matrix):

| Config | Inference Backend | Transport | Purpose |
|--------|------------------|-----------|---------|
| (A) | TensorRT | NITROS | Baseline — original performance |
| (B) | TensorRT | Standard ROS2 TensorList | Isolate NITROS overhead |
| (C) | ONNX Runtime (CUDA EP) | NITROS | Isolate inference backend overhead |
| (D) | ONNX Runtime (CUDA EP) | Standard ROS2 TensorList | Target vendor-neutral stack |

Metrics:
- End-to-end latency (image in → detections out)
- Inference-only latency (tensor in → tensor out)
- Throughput (FPS at sustained load)
- GPU memory usage

Phase 2a AMD profiling measures only the target path: ONNX Runtime MIGraphX + standard ROS2 TensorList. Report it beside the Phase 1 A100 TensorRT+NITROS baseline and A100 ORT+std target; do not create an AMD A/B/C/D matrix in this repository.
