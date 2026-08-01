# Phase 2B managed transport

The application consumes the independent `gpu_ros_managed` sibling
repository. NVIDIA Isaac ROS NITROS `v4.5-0`, commit
`82310fce298d3d9db26945a3a988d5c471d14973`, is the fixed behavior reference.

Ownership is:

```text
shared ROS/NITROS/Managed message
  -> ManagedTensorListView
  -> OnnxInferenceCore
  -> ReadHandle or BlockingReadyLease
  -> Ort::Value
  -> synchronous Run/output synchronization
  -> lease destruction
  -> message/buffer release
```

Existing `std` and `nitros` topic names, components, parameters, and Phase
A/B/C/D launch/test entry points are unchanged. Select the new path with
`transport:=managed`.

The NVIDIA RT-DETR graph is deliberately a transport comparison, not a
replacement for the vendor graph:

```text
official NITROS preprocessor
  -> NitrosToManagedTensorListNode
  -> OnnxInferenceNode(transport=managed, execution_provider=cuda)
  -> ManagedToNitrosTensorListNode
  -> official NITROS decoder
```

Each boundary waits for the upstream completion contract and transfers the
existing device allocation owner. It makes zero tensor-payload copies. Set
`nitros_to_managed.enable_timing:=true` and
`managed_to_nitros.enable_timing:=true` to record their callback/readiness
cost independently; each report states `payload copies=0`.

From the workspace `src` directory, import the pinned sibling with:

```bash
vcs import < amd_ros_object_detection/dependencies/gpu_ros_managed.repos
```

The NVIDIA and AMD compose files mount the sibling automatically. Set
`GPU_ROS_MANAGED_DIR` if it is not located next to this repository.

## Current validation status

- backend-neutral core: standalone compile and fake-backend lifecycle tests;
- CUDA backend: compile-verified against the local CUDA SDK;
- NITROS C lane: source-adapted to the fixed 4.5 API; container compile/runtime
  remains pending on an NVIDIA Isaac ROS environment;
- HIP/MIGraphX: device input uses explicit D2H staging inside the ORT adapter
  until the ORT 1.23.1 external HIP pointer probe is completed;
- native Managed HIP output is host-backed in this revision.

## NVIDIA validation order

Inside the Phase 2B NVIDIA container, build and run tests first:

```bash
colcon build
source install/setup.bash
colcon test --packages-select gpu_ros_managed_core gpu_ros_managed_cuda gpu_ros_managed_tensor_list --event-handlers console_direct+
colcon test-result --verbose
colcon test --packages-select isaac_ros_onnx_inference --event-handlers console_direct+
colcon test-result --verbose
launch_test src/amd_ros_object_detection/migrated_packages/isaac_ros_onnx_inference/test/isaac_ros_onnx_rtdetr_pol_test.py
launch_test src/amd_ros_object_detection/migrated_packages/isaac_ros_onnx_inference/test/isaac_ros_onnx_rtdetr_managed_pol_test.py
launch_test src/amd_ros_object_detection/migrated_packages/benchmarks/isaac_ros_rtdetr_transport_probe.py
```

Only after the probe and POL tests pass, run the precision-aligned A, C, and M
graphs once each. The benchmark framework owns its five measured iterations,
warm-up, throughput search, and fixed-rate trial:

```bash
launch_test src/amd_ros_object_detection/migrated_packages/benchmarks/isaac_ros_rtdetr_config_a_fp32_graph.py
launch_test src/amd_ros_object_detection/migrated_packages/benchmarks/isaac_ros_rtdetr_config_c_graph.py
launch_test src/amd_ros_object_detection/migrated_packages/benchmarks/isaac_ros_rtdetr_managed_graph.py
```

For a fixed-input C/M comparison, record both Detection2DArray topics to bags
and require stamp matching, at least 20 pairs, no unpaired frames, mean IoU
at least 0.999, score delta at most `1e-4`, and class match rate 1.0:

```bash
./src/amd_ros_object_detection/migrated_packages/isaac_ros_detection_validation/scripts/run_nvidia_fixed_input_capture.sh \
  rtdetr-c \
  rtdetr_config_c_20260801

./src/amd_ros_object_detection/migrated_packages/isaac_ros_detection_validation/scripts/run_nvidia_fixed_input_capture.sh \
  rtdetr-managed \
  rtdetr_managed_20260801

ros2 run \
  isaac_ros_detection_validation \
  compare_detection2d_bags.py \
  --reference-bag <config-c-bag> \
  --candidate-bag <managed-bag> \
  --output-json migrated_packages/benchmark_results/rtdetr_c_vs_managed_validation.json \
  --match-policy stamp \
  --min-mean-iou 0.999 \
  --max-mean-score-delta 0.0001 \
  --min-frame-pass-rate 1.0 \
  --min-paired-frames 20 \
  --min-class-match-rate 1.0
```

Historical Phase 1 results do not validate the revised C or Managed paths.
Zero-copy claims require current pointer-identity, copy-count, stream-ordering,
and detection-validation results.
