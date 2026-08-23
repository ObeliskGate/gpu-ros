# Phase 1: NVIDIA Backend and Transport Experiment

## Scope

Phase 1 compares inference backend and TensorBundle transport on the same
NVIDIA environment. These are complete pipeline configurations, not strictly
additive two-factor measurements.

RT-DETR uses the precision-aligned matrix:

| Config | Backend | Transport | Precision |
| --- | --- | --- | --- |
| A_fp32 | TensorRT | NITROS | FP32 |
| B_fp32 | TensorRT | standard ROS 2 bridge | FP32 |
| C | ONNX Runtime CUDA EP | NITROS | FP32 |
| D | ONNX Runtime CUDA EP | standard ROS 2 | FP32 |

The ordinary A/B RT-DETR graphs are TensorRT FP16 and must not be confused
with A_fp32/B_fp32. YOLOv8 A/B use TensorRT FP16; C/D use ORT FP32. YOLOv8
A/C therefore is not a pure backend comparison.

## Environment and build

Use the Phase 0 NVIDIA environment:

~~~bash
./docker/phase1-nvidia.sh bootstrap
./docker/phase1-nvidia.sh colcon
./docker/phase1-nvidia.sh shell
~~~

The helper enforces Release builds. Before a formal benchmark, run the
package tests and inspect the complete result:

~~~bash
colcon test \
  --merge-install \
  --packages-select \
    gpu_ros_managed_core \
    gpu_ros_managed_cuda \
    gpu_ros_managed_ros \
    gpu_ros_managed_tensor_bundle \
    gpu_ros_tensor_bundle_msgs \
    gpu_ros_nvidia_tensor_bundle_compat \
    gpu_ros_onnx_inference \
    gpu_ros_rtdetr \
    gpu_ros_yolov8 \
    gpu_ros_detection_validation \
  --event-handlers console_direct+

colcon test-result --all --verbose
~~~

Run the relevant proof-of-life tests and the RT-DETR transport probe before
benchmarking. A test failure stops the experiment.

For the RT-DETR lanes, the minimum checks are:

~~~bash
launch_test \
  src/amd_ros_object_detection/migrated_packages/gpu_ros_onnx_inference/test/gpu_ros_onnx_rtdetr_pol_test.py
launch_test \
  src/amd_ros_object_detection/migrated_packages/gpu_ros_rtdetr/test/gpu_ros_std_rtdetr_pol_test.py
launch_test \
  src/amd_ros_object_detection/migrated_packages/benchmarks/gpu_ros_rtdetr_transport_probe.py
~~~

The first A_fp32/B_fp32 run may create /tmp/sdetr_grasp_fp32.plan. That file is
a temporary TensorRT artifact, not a result archive.

## Configuration C device boundary

Configuration C keeps tensor payloads on the CUDA device across the
NITROS/ORT boundary. The adapter borrows the input device pointer, uses ORT
CUDA I/O Binding, and wraps ORT-owned output allocations with a release
callback. Metadata may be copied; the adapter must not add D2H or H2D payload
copies.

Provider placement is a separate question. CPU shape, bookkeeping, and
ORT-optimization nodes may be valid. A provider audit must still show CUDA
kernel events and must not show whole-model CPU fallback. Run profiling only
as a separate diagnostic.

## Benchmark commands

~~~bash
launch_test \
  src/amd_ros_object_detection/migrated_packages/benchmarks/gpu_ros_rtdetr_config_a_fp32_graph.py
launch_test \
  src/amd_ros_object_detection/migrated_packages/benchmarks/gpu_ros_rtdetr_config_b_fp32_graph.py
launch_test \
  src/amd_ros_object_detection/migrated_packages/benchmarks/gpu_ros_rtdetr_config_c_graph.py
launch_test \
  src/amd_ros_object_detection/migrated_packages/benchmarks/gpu_ros_rtdetr_config_d_graph.py
~~~

~~~bash
launch_test \
  src/amd_ros_object_detection/migrated_packages/benchmarks/gpu_ros_yolov8_config_a_graph.py
launch_test \
  src/amd_ros_object_detection/migrated_packages/benchmarks/gpu_ros_yolov8_config_b_graph.py
launch_test \
  src/amd_ros_object_detection/migrated_packages/benchmarks/gpu_ros_yolov8_config_c_graph.py
launch_test \
  src/amd_ros_object_detection/migrated_packages/benchmarks/gpu_ros_yolov8_config_d_graph.py
~~~

Start each graph once. The benchmark framework owns warm-up, measured
iterations, throughput search, and fixed-rate trials.

## Numeric validation and diagnostics

Performance benchmarks and output validation are separate. Use the fixed-input
capture tooling in
migrated_packages/gpu_ros_detection_validation/README.md where the lane is
supported, then compare bags with:

~~~bash
ros2 run gpu_ros_detection_validation \
  compare_detection2d_bags.py \
  --reference-bag <reference-bag> \
  --candidate-bag <candidate-bag> \
  --match-policy stamp \
  --output-json <comparison-result.json>
~~~

Use stamp matching when source header timestamps are preserved. Use index
matching only after input order and content are known to be identical.

ORT profiling, bridge timing, Nsight Systems tracing, and extra pointer logging
must remain separate diagnostic runs. Save benchmark, numeric comparison,
provider placement, pointer, copy-count, and timing evidence separately.

The final NVIDIA results and interpretation limits are recorded in
phase1-results.md.
