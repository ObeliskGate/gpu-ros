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

- backend-neutral core: fake-backend lifecycle, event-failure, safe-orphan,
  multi-reader, pool-destruction and pending-cleanup tests pass;
- CUDA backend: non-default stream, device mismatch, external owner and session
  lifetime tests pass on an NVIDIA A100;
- NITROS C and Managed lanes: compile, POL, component loading, pointer identity,
  fixed-input comparison and benchmark validation pass with Isaac ROS 4.5;
- Managed bridge runtime audit: Nsight Systems reports no Managed-only memcpy
  signature and no increase in input/output payload-copy rate. NITROS,
  Managed and round-trip NITROS payload pointers are identical in the adapter
  GTest;
- HIP/MIGraphX: device input uses explicit D2H staging inside the ORT adapter
  until the ORT 1.23.1 external HIP pointer probe is completed;
- native Managed HIP output is host-backed in this revision.

## NVIDIA validation result (2026-08-01)

The validated host was `boshen`, with an NVIDIA A100-SXM4-40GB, ROS 2 Jazzy,
Isaac ROS 4.5 and a Release build. The fixed input was `r2b_robotarm`, hash
`8eee68848ee1a95e21b1cd44d5d6ba71`.

The final YOLOv8 bridge audit used Config C as the reference and Managed as the
candidate. Config C produced 388 recorded detections and Managed produced 390,
so memory-operation counts were normalized by the number of processed output
frames. The byte-precision Nsight trace showed:

- identical Device-to-Device memcpy count and bytes across the complete traces;
- 388 versus 390 copies of the 2,822,400-byte Device-to-Host decoder output,
  exactly one per recorded output frame;
- 396 versus 396 copies of the 4,915,200-byte Device-to-Device input payload;
- no Managed-only memcpy signature;
- no increase in normalized payload-copy rate;
- `bridge_zero_copy_pass: true`.

The Device-to-Host output is an existing official YOLOv8 decoder behavior in
both lanes. It is not a Managed bridge copy. The ORT profiles were retained as
a control: both lanes assigned the same 175 nodes to CUDA, recorded 8,750 node
events over the bounded profile, and observed no CPU fallback.

The Managed benchmark logs also preserve callback/readiness timing. Across
equal 500-frame reporting windows, the mean boundary costs were:

- RT-DETR NITROS-to-Managed: 0.160 ms; Managed-to-NITROS: 0.059 ms;
- YOLOv8 NITROS-to-Managed: 0.064 ms; Managed-to-NITROS: 0.050 ms.

These timing values include callback, readiness and publish work. They are not
tensor-copy timings.

Two non-blocking limitations remain documented: fixed-input captures can
differ by one or two boundary frames, and the ORT C/M component container can
exit `-11` during post-report shutdown. Paired-frame numeric results and all
benchmark reports are written before that shutdown failure. Neither limitation
changes the pointer-identity or byte-precision zero-copy result.

## NVIDIA validation order

Inside the Phase 2B NVIDIA container, build and run tests first:

```bash
colcon build --merge-install
source install/setup.bash
colcon test --merge-install --packages-select gpu_ros_managed_core gpu_ros_managed_cuda gpu_ros_managed_tensor_list --event-handlers console_direct+
colcon test-result --verbose
colcon test --merge-install --packages-select isaac_ros_onnx_inference --event-handlers console_direct+
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
