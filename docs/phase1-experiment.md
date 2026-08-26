# Phase 1: NVIDIA Backend and Pipeline-Variant Experiment

## Scope

Phase 1 compares complete pipeline configurations on the same NVIDIA
environment. Backend and application-interface variants provide useful matrix
labels, but the configurations are not strictly additive two-factor
measurements: changing the variant also changes the model-specific pre/post
nodes and any required compatibility boundaries.

RT-DETR uses the precision-aligned matrix:

| Config | Backend | Application/interface variant | Precision |
| --- | --- | --- | --- |
| A_fp32 | TensorRT | NVIDIA/NITROS native | FP32 |
| B_fp32 | TensorRT | standard ROS 2-compatible migrated | FP32 |
| C | ONNX Runtime CUDA EP | NVIDIA/NITROS native | FP32 |
| D | ONNX Runtime CUDA EP | standard ROS 2 migrated | FP32 |

The ordinary A/B RT-DETR graphs are TensorRT FP16 and must not be confused
with A_fp32/B_fp32. YOLOv8 A/B use TensorRT FP16; C/D use ORT FP32. YOLOv8
A/C therefore is not a pure backend comparison.

For RT-DETR, the shared six-node image/tensor preprocessing chain remains the
NVIDIA/NITROS part in all four configurations. A_fp32 and C retain the
NVIDIA/NITROS RT-DETR preprocessor and decoder. B_fp32 and D use the
project-owned standard RT-DETR preprocessor and decoder. B_fp32 keeps the
TensorRT node, so explicit NVIDIA TensorList-to-project-TensorBundle adapters
and a standard-to-NITROS bridge surround that node; D uses ORT with standard
TensorBundle transport after its input boundary. Thus B_fp32 is the
standard-compatible TRT configuration, not a transport-only ablation, while D
is the migrated standard ORT target configuration.

The open-source remediation also changed the concrete standard interface from
the earlier NVIDIA TensorList-compatible path to
`gpu_ros_tensor_bundle_msgs/msg/TensorBundle`, with explicit conversion only
inside `gpu_ros_nvidia_tensor_bundle_compat`. This preserves the experiment's
native-versus-migrated roles but changes adapter and ownership costs. Historical
4.4 B/D numbers are therefore reference results for earlier implementations,
not direct regression baselines for the current 4.5-compatible source.

YOLOv8 follows the same A/B/C/D roles without the RT-DETR-specific
preprocessor: A/C retain the NVIDIA/NITROS decoder, while B/D use the project
standard decoder. B still needs the explicit TensorBundle/NITROS bridge around
TensorRT; D stays on the standard TensorBundle inference path after its input
boundary.

### RT-DETR original-size input

RT-DETR's `orig_target_sizes` is a real `1x2` `int64` model input. It controls
the coordinate scale of the returned boxes; it is not merely transport
metadata. The tensor is tiny and does not add a material image or inference
workload, so it is not expected to change the throughput conclusions of the
Phase 1 benchmark. It does, however, change the coordinate units in
`Detection2DArray`. Every fixed-input numeric comparison must therefore record
and use the same `orig_target_sizes` policy across the compared lanes. The
official benchmark graph and the numerical-capture recipe are separate: the
former remains the performance reference, while the latter makes the
coordinate policy explicit for output comparison.

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

# Managed device-buffer comparison: official NITROS pre/post-processing with
# ORT CUDA between NitrosToManagedTensorBundle and ManagedToNitrosTensorBundle.
launch_test \
  src/amd_ros_object_detection/migrated_packages/benchmarks/gpu_ros_rtdetr_managed_graph.py
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

The Managed RT-DETR run is a separate transport check against Config C, not a
replacement for Config D. It uses the same old NVIDIA SyntheticaDETR asset and
the same NITROS preprocessor/decoder as Config C. Compare its peak prediction,
mean output at peak, fixed-rate endpoint latencies, and missed frames with C;
the managed device-buffer path must show no material throughput or latency
regression. Keep the adapter's optional timing diagnostics disabled during the
formal benchmark.

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
