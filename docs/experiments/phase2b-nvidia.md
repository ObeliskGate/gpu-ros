# Phase 2B NVIDIA managed transport runbook

This runbook validates the CUDA managed TensorBundle path against the NVIDIA
NITROS reference in the Isaac ROS 4.5-compatible runtime. It is separate from
the Phase 1 A/B/C/D matrix and does not redefine those configurations.

The two formal comparisons are:

| Model | Reference C | Managed candidate |
| --- | --- | --- |
| RT-DETR | NVIDIA/NITROS preprocessing and decoder, ORT CUDA + NITROS | The same pipeline with the two managed TensorBundle bridges around ORT CUDA |
| YOLOv8 | NVIDIA/NITROS preprocessing and decoder, ORT CUDA + NITROS | The same pipeline with the two managed TensorBundle bridges around ORT CUDA |

Managed is a transport comparison, not a replacement for the reference graph
and not a standard-ROS B/D lane. The boundary must preserve pointer identity
and add no tensor-payload copy.

## Workspace and source identity

The NVIDIA runtime retains this outer workspace layout:

```text
/workspaces/isaac_ros-dev/
├── src/gpu-ros/                 this monorepo
├── src/nvidia_external/         pinned external checkouts
├── build/ and install/          outer-workspace build/install state
└── log/                         outer-workspace logs
```

Compose mounts the repository once at `src/gpu-ros`, keeps the outer workspace
as its working directory, and exports `GPU_ROS_REPO_ROOT` accordingly. Use the
pinned Isaac ROS 4.5-compatible runtime, Release build, ORT 1.23.1, and the
external checkouts pinned by `external/nvidia-isaac-ros.repos`. Models and the
R2B bag remain outside Git. Record the exact `monorepo_revision`, dirty-state
hashes, runtime-image digest, model/dataset hashes, and external result root in
the archive; do not put deployment or machine identity in this document.

If a runtime already contains the pinned image, external sources, assets, and
workspace, update only the monorepo checkout without discarding dirty
diagnostic files:

```bash
cd /workspaces/isaac_ros-dev/src/gpu-ros
git status --short
git pull --ff-only origin main
```

Set a host-backed result root and a unique run id in the local environment:

```bash
export RUN_ID=phase2b-nvidia-<date>-<suffix>
export RESULTS_ROOT=/absolute/path/to/external-results/${RUN_ID}
export CONTAINER=<running-runtime-name>
```

The existing runtime setup must make this host-backed result root available at
`/results` for the commands below. This is an external result mapping, not a
second repository bind or a new Compose source mount; keep it outside the
monorepo.

A pull updates only committed source. Preserve and record any dirty state; do
not discard it with a destructive reset during an experiment.

## Build and runtime tests

Resolve the Compose file from the repository path, then run the existing
The container working directory remains `/workspaces/isaac_ros-dev`:

```bash
cd /workspaces/isaac_ros-dev/src/gpu-ros
docker compose -f docker-compose.yaml config
./docker/phase1-nvidia.sh up
./docker/phase1-nvidia.sh colcon
./docker/phase1-nvidia.sh verify
./docker/phase1-nvidia.sh shell
```

Run the incremental build inside the existing runtime. It does not rebuild
runtime layers:

```bash
docker exec "${CONTAINER}" bash -lc \
  'source /opt/ros/jazzy/setup.bash; \
   source /workspaces/isaac_ros-dev/install/setup.bash; \
   cd /workspaces/isaac_ros-dev; \
   colcon build --event-handlers console_cohesion+'
```

Run the required C++/CUDA and package gates, then archive the complete output:

```bash
docker exec "${CONTAINER}" bash -lc \
  'source /opt/ros/jazzy/setup.bash; \
   cd /workspaces/isaac_ros-dev; source install/setup.bash; \
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

Captures are for detection equivalence, not throughput. Use the same input bag,
image dimensions, model files, and RT-DETR `orig_target_sizes` policy for both
lanes. Keep capture output under the external result root:

```bash
docker exec "${CONTAINER}" bash -lc \
  'set -e; ulimit -c 0; \
   source /opt/ros/jazzy/setup.bash; source /workspaces/isaac_ros-dev/install/setup.bash; \
   cd /workspaces/isaac_ros-dev; \
   export CAPTURE_OUTPUT_ROOT=/results/'"${RUN_ID}"'/bags; \
   export CAPTURE_PLAYBACK_RATE=0.25 CAPTURE_DRAIN_SECONDS=30 CAPTURE_MIN_MESSAGES=100; \
   src/gpu-ros/evaluation/gpu_ros_detection_validation/scripts/run_nvidia_fixed_input_capture.sh rtdetr-c rtdetr-c; \
   src/gpu-ros/evaluation/gpu_ros_detection_validation/scripts/run_nvidia_fixed_input_capture.sh rtdetr-managed rtdetr-managed; \
   src/gpu-ros/evaluation/gpu_ros_detection_validation/scripts/run_nvidia_fixed_input_capture.sh yolov8-c yolov8-c; \
   src/gpu-ros/evaluation/gpu_ros_detection_validation/scripts/run_nvidia_fixed_input_capture.sh yolov8-managed yolov8-managed'
```

Compare stamp-paired outputs in report-only mode. Exact source stamps are the
formal pairing method; index pairing is diagnostic only when replay boundaries
create a one-frame skew:

```bash
docker exec "${CONTAINER}" bash -lc \
  'source /opt/ros/jazzy/setup.bash; source /workspaces/isaac_ros-dev/install/setup.bash; \
   cd /workspaces/isaac_ros-dev; \
   ros2 run gpu_ros_detection_validation compare_detection2d_bags.py \
     --reference-bag /results/'"${RUN_ID}"'/bags/rtdetr-c \
     --candidate-bag /results/'"${RUN_ID}"'/bags/rtdetr-managed \
     --match-policy stamp --report-only \
     --output-json /results/'"${RUN_ID}"'/rtdetr-c-vs-managed.json'
```

Repeat for YOLOv8. Record paired/unpaired frames, class match rate, IoU, score
deltas, and the comparator source revision. A throughput pass is not a
numeric pass.

## Formal throughput

Run each graph separately with core dumps, bridge timing, Nsight, ORT
profiling, and debug instrumentation disabled. Both YOLOv8 managed bridge
timing parameters must be `False`:

```bash
docker exec "${CONTAINER}" bash -lc \
  'set -o pipefail; ulimit -c 0; \
   source /opt/ros/jazzy/setup.bash; source /workspaces/isaac_ros-dev/install/setup.bash; \
   cd /workspaces/isaac_ros-dev; \
   launch_test src/gpu-ros/evaluation/benchmarks/gpu_ros_rtdetr_config_c_graph.py; \
   launch_test src/gpu-ros/evaluation/benchmarks/gpu_ros_rtdetr_managed_graph.py; \
   launch_test src/gpu-ros/evaluation/benchmarks/gpu_ros_yolov8_config_c_graph.py; \
   launch_test src/gpu-ros/evaluation/benchmarks/gpu_ros_yolov8_managed_graph.py'
```

Copy each newly created `r2b-log-*.json` into the external result root after
its graph exits. Record predicted peak, measured output rate, 10/30/60 Hz
means and misses, latency endpoints, command, monorepo revision, and teardown
status. Valid JSON with an unclean teardown is retained but marked
`INCONCLUSIVE`.

## Provider, pointer, and copy audit

Run the unified audit separately for each model. It reruns fixed-input lanes
with short ORT profiles and CUDA traces, then writes provider,
pointer/lifetime, copy, and detection reports:

```bash
docker exec "${CONTAINER}" bash -lc \
  'set -o pipefail; ulimit -c 0; \
   source /opt/ros/jazzy/setup.bash; source /workspaces/isaac_ros-dev/install/setup.bash; \
   cd /workspaces/isaac_ros-dev; \
   export CAPTURE_AUDIT_ROOT=/results/'"${RUN_ID}"'/audits; \
   export AUDIT_ORT_PROFILE_FRAMES=50; \
   src/gpu-ros/evaluation/gpu_ros_detection_validation/scripts/run_nvidia_transport_audit.sh rtdetr rtdetr-c-vs-managed; \
   src/gpu-ros/evaluation/gpu_ros_detection_validation/scripts/run_nvidia_transport_audit.sh yolov8 yolov8-c-vs-managed'
```

Run one model at a time and check external free space. A `PASS` requires
complete copy records, CUDA provider activity, pointer/lifetime evidence, no
managed-only tensor-payload copy signature, and a passing stamp-paired
detection report. CPU bookkeeping in RT-DETR is diagnostic and is not a failure
by itself.

## Historical campaign notes

The pre-fix campaign wrote valid throughput JSONs but the component runtime
later exited with `SIGSEGV (-11)` during teardown; retain those rows as
historical `INCONCLUSIVE` observations. The subsequent lifecycle-fix rerun
completed all four formal graphs with zero launch exit codes and no remaining
benchmark process. The concise records are kept in the results documents; full
bags, profiles, traces, and logs remain in the external result archive.

New matrix manifests use schema 3 and one `monorepo_revision`; fixed-input
capture logs remain unversioned and use the monorepo revision/diff fields. Do
not rewrite the historical result archives.
