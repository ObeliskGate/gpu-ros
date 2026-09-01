# Phase 2B NVIDIA Managed transport runbook

This runbook validates the CUDA Managed TensorBundle path against the NVIDIA
NITROS reference in the Isaac ROS 4.5-compatible runtime. It is separate from
the Phase 1 A/B/C/D matrix and does not redefine those configurations.

The two formal comparisons are:

| Model | Reference C | Managed candidate |
| --- | --- | --- |
| RT-DETR | NVIDIA/NITROS preprocessing and decoder, ORT CUDA + NITROS | The same pipeline with the two Managed TensorBundle bridges around ORT CUDA |
| YOLOv8 | NVIDIA/NITROS preprocessing and decoder, ORT CUDA + NITROS | The same pipeline with the two Managed TensorBundle bridges around ORT CUDA |

Managed is a transport comparison, not a replacement for the reference graph
and not a standard-ROS B/D lane. The boundary must preserve pointer identity
and add no tensor-payload copy.

## Environment and source identity

Use the pinned Isaac ROS 4.5-compatible runtime, Release build, ORT 1.23.1,
and the external checkouts pinned by `external/nvidia-isaac-ros.repos`.
Models and the R2B bag remain outside Git. Record the exact application and
Managed revisions, runtime-image digest, model/dataset hashes, and result root
in the external archive; do not put deployment or machine identity in this
document.

If a runtime already contains the pinned image, external sources, assets, and
workspace, update the application checkout without discarding dirty diagnostic
files:

```bash
cd /absolute/path/to/amd_ros_object_detection
git pull --ff-only origin opensource-prep
```

Set a host-backed result root and a unique run id in the local environment:

```bash
export RUN_ID=phase2b-nvidia-<date>-<suffix>
export RESULTS_ROOT=/absolute/path/to/external-results/${RUN_ID}
export CONTAINER=<running-runtime-name>
```

`git pull` updates only committed source. Preserve and record any dirty state;
never discard it with a destructive reset during an experiment.

## Build and runtime tests

Run the incremental build inside the existing runtime. It does not rebuild
runtime layers:

```bash
docker exec "${CONTAINER}" bash -lc \
  'source /opt/ros/jazzy/setup.bash; \
   source /workspace/install/setup.bash; \
   cd /workspace; \
   colcon build --event-handlers console_cohesion+'
```

Run the required C++/CUDA and package gates, then archive the complete output:

```bash
docker exec "${CONTAINER}" bash -lc \
  'source /opt/ros/jazzy/setup.bash; \
   cd /workspace; source install/setup.bash; \
   for p in \
     gpu_ros_managed_core gpu_ros_managed_cuda gpu_ros_managed_tensor_bundle \
     gpu_ros_onnx_inference gpu_ros_nvidia_tensor_bundle_compat \
     gpu_ros_rtdetr gpu_ros_yolov8; do \
     ctest --test-dir "build/${p}" --output-on-failure \
       -E "copyright|cppcheck|cpplint|flake8|lint_cmake|pep257|uncrustify|xmllint|test_summarize_ort_profile|test_compare_nsys_cuda_traces|test_compare_nvidia_copy_traces|test_analyze_rocprof_traces"; \
   done'
```

The gates cover lifecycle/orphan handling, stream and provider selection,
pointer identity, output ownership, staging limits, adapters, decoder
behavior, and CUDA proof-of-life.

## Fixed-input correctness

Captures are for detection equivalence, not throughput. Use the same input
bag, image dimensions, model files, and RT-DETR `orig_target_sizes` policy for
both lanes. Keep capture output under the external result root:

```bash
docker exec "${CONTAINER}" bash -lc \
  'set -e; ulimit -c 0; \
   source /opt/ros/jazzy/setup.bash; source /workspace/install/setup.bash; \
   cd /workspace; \
   export CAPTURE_OUTPUT_ROOT=/results/'"${RUN_ID}"'/bags; \
   export CAPTURE_PLAYBACK_RATE=0.25 CAPTURE_DRAIN_SECONDS=30 CAPTURE_MIN_MESSAGES=100; \
   ./migrated_packages/gpu_ros_detection_validation/scripts/run_nvidia_fixed_input_capture.sh rtdetr-c rtdetr-c; \
   ./migrated_packages/gpu_ros_detection_validation/scripts/run_nvidia_fixed_input_capture.sh rtdetr-managed rtdetr-managed; \
   ./migrated_packages/gpu_ros_detection_validation/scripts/run_nvidia_fixed_input_capture.sh yolov8-c yolov8-c; \
   ./migrated_packages/gpu_ros_detection_validation/scripts/run_nvidia_fixed_input_capture.sh yolov8-managed yolov8-managed'
```

Compare stamp-paired outputs in report-only mode. Exact source stamps are the
formal pairing method; index pairing is diagnostic only when replay boundaries
create a one-frame skew:

```bash
docker exec "${CONTAINER}" bash -lc \
  'source /opt/ros/jazzy/setup.bash; source /workspace/install/setup.bash; \
   ros2 run gpu_ros_detection_validation compare_detection2d_bags.py \
     --reference-bag /results/'"${RUN_ID}"'/bags/rtdetr-c \
     --candidate-bag /results/'"${RUN_ID}"'/bags/rtdetr-managed \
     --match-policy stamp --report-only \
     --output-json /results/'"${RUN_ID}"'/rtdetr-c-vs-managed.json'
```

Repeat for YOLOv8. Record paired/unpaired frames, class match rate, IoU,
score deltas, and the comparator source revision. A throughput pass is not a
numeric pass.

## Formal throughput

Run each graph separately with core dumps, bridge timing, Nsight, ORT
profiling, and debug instrumentation disabled. Both YOLOv8 Managed bridge
timing parameters must be `False`.

```bash
docker exec "${CONTAINER}" bash -lc \
  'set -o pipefail; ulimit -c 0; \
   source /opt/ros/jazzy/setup.bash; source /workspace/install/setup.bash; \
   cd /workspace; \
   launch_test migrated_packages/benchmarks/gpu_ros_rtdetr_config_c_graph.py; \
   launch_test migrated_packages/benchmarks/gpu_ros_rtdetr_managed_graph.py; \
   launch_test migrated_packages/benchmarks/gpu_ros_yolov8_config_c_graph.py; \
   launch_test migrated_packages/benchmarks/gpu_ros_yolov8_managed_graph.py'
```

Copy each newly created `r2b-log-*.json` into the external result root after
its graph exits. Record predicted peak, measured output rate, 10/30/60 Hz
means and misses, latency endpoints, command, and teardown status. Valid JSON
with an unclean teardown is retained but marked `INCONCLUSIVE`.

## Provider, pointer, and copy audit

Run the unified audit separately for each model. It reruns fixed-input lanes
with short ORT profiles and CUDA traces, then writes provider,
pointer/lifetime, copy, and detection reports:

```bash
docker exec "${CONTAINER}" bash -lc \
  'set -o pipefail; ulimit -c 0; \
   source /opt/ros/jazzy/setup.bash; source /workspace/install/setup.bash; \
   cd /workspace; \
   export CAPTURE_AUDIT_ROOT=/results/'"${RUN_ID}"'/audits; \
   export AUDIT_ORT_PROFILE_FRAMES=50; \
   ./migrated_packages/gpu_ros_detection_validation/scripts/run_nvidia_transport_audit.sh rtdetr rtdetr-c-vs-managed; \
   ./migrated_packages/gpu_ros_detection_validation/scripts/run_nvidia_transport_audit.sh yolov8 yolov8-c-vs-managed'
```

Run one model at a time and check external free space. A `PASS` requires
complete copy records, CUDA provider activity, pointer/lifetime evidence, no
Managed-only tensor-payload copy signature, and a passing stamp-paired
detection report. CPU bookkeeping in RT-DETR is diagnostic and is not a
failure by itself.

## Historical campaign notes

The pre-fix campaign wrote valid throughput JSONs but the component runtime
later exited with `SIGSEGV (-11)` during teardown; retain those rows as
historical `INCONCLUSIVE` observations. The subsequent lifecycle-fix rerun
completed all four formal graphs with zero launch exit codes and no remaining
benchmark process. The concise records are kept in the results documents; full
bags, profiles, traces, and logs remain in the external result archive.
