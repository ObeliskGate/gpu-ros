# AGENTS.md

## Project Overview

This project migrates NVIDIA Isaac ROS object detection packages from the NVIDIA-proprietary stack (TensorRT + NITROS + GXF) to a vendor-neutral runtime (ONNX Runtime) that supports both NVIDIA (CUDA EP) and AMD (ROCm/MIGraphX EP) GPUs.

The upstream source is `isaac_ros_object_detection` (v4.4.0) which contains detection pipelines for YOLOv8, RT-DETR, Grounding DINO, and DetectNet.

## Long-term Goals

1. **Phase 1 (current)**: Replace `isaac_ros_tensor_rt` with an ONNX Runtime inference node using standard ROS2 `isaac_ros_tensor_list_interfaces` messages. Remove NITROS dependency from decoder nodes. Validate on NVIDIA GPU and profile four configurations (2×2 matrix of inference backend × transport):
   - (A) TensorRT + NITROS (baseline)
   - (B) TensorRT + standard ROS2 interfaces (isolate NITROS overhead)
   - (C) ONNX Runtime + NITROS (isolate inference backend overhead)
   - (D) ONNX Runtime + standard ROS2 interfaces (target)
2. **Phase 2**: Port the pipeline to AMD GPU using ONNX Runtime ROCm EP or MIGraphX EP.
3. **Phase 3**: Extend to RT-DETR and Grounding DINO pipelines.

## Architecture

### Original pipeline (NVIDIA-only)
```
Image → dnn_image_encoder (NITROS) → TensorRTNode (NITROS) → DecoderNode (NITROS) → Detection2DArray
```

### Target pipeline (vendor-neutral)
```
Image → ImageEncoder (std ROS2) → OnnxInferenceNode (std ROS2 TensorList) → DecoderNode (std ROS2 TensorList) → Detection2DArray
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

Compare four configurations on the same NVIDIA GPU (2×2 matrix):

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
