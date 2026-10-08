# Phase 2B NVIDIA managed transport

This method compares the Isaac ROS 5.0 native TensorList Config C reference with
managed TensorBundle bridges around ORT CUDA for RT-DETR and YOLOv8. It is separate
from the [Phase 1 A/B/C/D matrix](phase1-nvidia.md). Managed transport must
preserve pointer ownership without an extra tensor-payload copy at the
inference boundary; it does not make the complete pipeline copy-free.

## Workspace and identity

The outer workspace is `/workspaces/isaac_ros-dev`. The repository is mounted
once at `src/gpu-ros`; image-matched source overlays, when needed, are mounted
read-only at `src/nvidia_external`. `gpu_ros_object_detection/` owns detection
facilities, not the outer workspace or monorepo identity.

Use an official Isaac ROS 5.0 Docker image with Lyrical. Its immutable digest,
paired package versions and actual installed headers/libraries define the ROS
baseline. Prefer image binaries over duplicate source overlays. Resolve any
required external source commit from the image's official provenance, not a
current upstream release head. Missing image identity or source provenance
blocks that runtime step; it does not authorize upgrading the image.

Before running anything, verify GPU model/architecture and driver, the image
identity, ROS package versions/prefixes, any external source revisions, loaded
ORT library identity, source revision and dirty/untracked hashes, model bytes, and each
R2B file. Preserve original asset volumes. A model profile without a fixed
expected SHA needs independent provenance and a recorded actual digest; it has
not passed a known-digest check. Do not treat the framework's 32-character
input hash as SHA-256.

For migration verification, retain the actual pre-change worktree in an
independent checkout with real Git metadata. Preserve model/data bytes and use
separate empty build/install/log and result directories. A 4.5/Jazzy baseline
and a 5.0/Lyrical candidate have different runtime identities: compare them as
migration observations, not same-stack performance reproduction. If no usable
old runtime or raw baseline exists, report the before/after comparison blocked.
Do not source an old project overlay to fill a missing package.

## Reuse the dependency image

Set `REPO_ROOT` and `VERIFY_ROOT` to the verified checkout and external run
state. Prefer image-installed binary dependencies. Only a required,
image-matched source overlay belongs under `src/nvidia_external`; add its
read-only bind explicitly to the run-specific override.
Set `GPU_ROS_NVIDIA_PROFILE=1` before colcon package discovery. Exercise this
method on the identified target image and retain independent failure statuses;
the [current campaign](../../../docs/results/README.md#isaac-ros-50-migration-campaign-2026-10-05)
records its accepted SDK shutdown exception separately. Use the supplied existing
environment entrypoint; do not start another container when that would bypass its
deployment contract.
When Compose is the selected entrypoint, create an external `NV_OVERRIDE` with:

- the literal verified existing image ID for service `dev`;
- a unique Compose project and fresh binds for outer-workspace build, install,
  and log directories;
- a new results bind at `/workspaces/ovg-results`;
- the existing `assets` volume declared `external: true` with its verified
  name, rather than a new empty volume.

Keep the NVIDIA runtime hook, GPU reservation, host networking, and container
working directory from the tracked Compose file. It no longer mounts old 4.5
source trees. Resolve the combined configuration before starting it:

```bash
NV_COMPOSE="${REPO_ROOT}/gpu_ros_object_detection/docker/docker-compose.yaml"
NV_OVERRIDE="${VERIFY_ROOT}/nvidia.override.yaml"
docker compose -f "${NV_COMPOSE}" -f "${NV_OVERRIDE}" config
docker compose -f "${NV_COMPOSE}" -f "${NV_OVERRIDE}" up -d --no-build dev
```

Use both files for every subsequent command. Do not call launcher `up` or
`colcon` branches that omit the override. Reusing an image verifies the existing
dependency runtime plus the current checkout, not a fresh Dockerfile build.

## Build and runtime tests

Build from the fresh outer workspace without sourcing an old project install:

```bash
docker compose -f "${NV_COMPOSE}" -f "${NV_OVERRIDE}" exec -T dev \
  bash -lc 'set -e; export GPU_ROS_NVIDIA_PROFILE=1; source /opt/gpu-ros/setup.bash; colcon build'
```

Run the selected colcon tests and the packages not selected by the NVIDIA test
defaults. Keep every command failure visible:

```bash
docker compose -f "${NV_COMPOSE}" -f "${NV_OVERRIDE}" exec -T dev \
  bash -lc 'set -e
    export GPU_ROS_NVIDIA_PROFILE=1
    source /opt/gpu-ros/setup.bash
    source install/setup.bash
    colcon test --event-handlers console_direct+
    colcon test-result --all --verbose
    ctest --test-dir build/gpu_ros_nvidia_tensor_bundle_compat \
      --tests-regex "^(test_tensor_list_native|test_tensor_list_buffer_adapter|test_tensor_bundle_conversion)$" \
      --output-on-failure --no-tests=error
    ctest --test-dir build/gpu_ros_onnx_inference \
      --tests-regex "^(test_native_onnx_executor|test_onnx_inference_core)$" \
      --output-on-failure --no-tests=error
    for package in gpu_ros_detection_common gpu_ros_rtdetr \
      gpu_ros_yolov8 gpu_ros_detection_validation; do
      ctest --test-dir "build/${package}" --output-on-failure
    done'
```

Check Release configuration, package prefixes, CMake source/install paths, and
the actual loaded ORT libraries. The defaults discover twelve project packages
including `gpu_ros_nvidia_reference`, excluding HIP; external packages are
counted separately. Inspect the five installed reference launches with
`ros2 launch gpu_ros_nvidia_reference <file> --show-args`, and the compatibility
boundary launch in its own package. Old owners must not supply duplicate launch
files. A skipped device test is not a PASS.

Source-only implementation checks are not ROS/CUDA acceptance. Without the
selected SDK, run the method index's CPU checks, standalone core plus installed
consumer, and syntax checks; record the native C++ build and GPU behavior tests
as pending. Deployment and GPU verification require a separately authorized
runtime stage. Historical pre-separation captures do not establish that the new
native component has passed.

After a clean SDK build, exercise an installed consumer linking only the native
component/session targets. Require actual CUDA Buffer inference, direct and
dynamic-fallback Add payloads, and output readback after node/session destruction,
not only a successful include/link step. Run both original-model native and
Managed proof-of-life/capture paths before copy acceptance.

## Non-interactive runtime payloads

Capture, comparison, audit, and benchmark commands below run through the same
Compose prefix. Set `PAYLOAD` to the requested existing command sequence; do not
enter an interactive shell or introduce another graph runner:

```bash
docker compose -f "${NV_COMPOSE}" -f "${NV_OVERRIDE}" exec -T dev \
  bash -lc "set -e; ulimit -c 0
    export GPU_ROS_NVIDIA_PROFILE=1
    source /opt/gpu-ros/setup.bash
    source /workspaces/isaac_ros-dev/install/setup.bash
    ${PAYLOAD}"
```

If intentionally collecting evidence after a known failure, record each exit
code explicitly and return the accumulated failure status. `set -e` cannot
reveal a child error swallowed by a capture or launch-test wrapper; inspect
component logs and post-exit processes too.

## Fixed-input detection observations
Use [detection validation](detection-validation.md#nvidia) for the six existing
capture lanes and parameters. Preserve model/input bytes, provider, image
geometry, RT-DETR size policy, and source timestamps. Use unique output names,
profiling disabled, playback 0.25, drain 30 seconds, at least 100 messages, and
stop grace/termination windows of 60/20 seconds when matching this protocol.

Compare each migration lane with its own pre-change bag to collect a
detection-comparison observation; do not treat metric differences or unmatched
coverage as an acceptance verdict. C versus managed and C versus D are separate
cross-lane observations. Use the complete default report: exact timestamp
pairing, FIFO for duplicate stamps, and deterministic class-unconstrained
IoU-greedy detection matching. Record coverage and numeric observations.
Optional index or filtering diagnostics retain their method/options and do not
replace the full-coverage default report.

## Formal throughput

Run only when explicitly requested. Keep core dumps, bridge timing, Nsight,
ORT profiling, and debug instrumentation disabled. Both managed YOLOv8 bridge
timing parameters remain `False`. Do not change graph search, warmup, frequency,
executor, or QoS settings.

From `/workspaces/isaac_ros-dev`, run each command as a separate payload and
archive its newly emitted `r2b-log-*.json` and full log before the next command:

```bash
launch_test \
  src/gpu-ros/gpu_ros_object_detection/benchmarks/gpu_ros_rtdetr_config_c_graph.py
launch_test \
  src/gpu-ros/gpu_ros_object_detection/benchmarks/gpu_ros_rtdetr_managed_graph.py
launch_test \
  src/gpu-ros/gpu_ros_object_detection/benchmarks/gpu_ros_yolov8_config_c_graph.py
launch_test \
  src/gpu-ros/gpu_ros_object_detection/benchmarks/gpu_ros_yolov8_managed_graph.py
```

Inventory existing reports before each graph so an old JSON cannot satisfy the
run. Record peak prediction, mean output, peak misses/sent, the actual fixed-rate
fields, latency endpoints, command status, component teardown, and remaining
processes. Do not invent a NVIDIA three-round runner or fill absent fields with
zero. Valid JSON with unclean teardown is retained as INCONCLUSIVE evidence,
not promoted as a clean benchmark. NVIDIA has no declared multi-round tolerance;
report historical deltas without inventing one.

## Provider, pointer, and copy audit

Use `run_nvidia_transport_audit.sh` separately for each model with
`AUDIT_ORT_PROFILE_FRAMES=50` and a new `CAPTURE_AUDIT_ROOT`; see
[transport audits](detection-validation.md#transport-audits). Capture parameters
must match the corresponding baseline. Keep provider activity, pointer/lifetime
binding reports, CUDA traces, copy and detection reports, and lifecycle logs.

The native transport token is `tensor_list`; its independent
`gpu_ros::onnx_inference::NativeOnnxInferenceNode` uses `output_contracts`.
The `managed` lane retains `OnnxInferenceNode` and `managed_output_contracts`.
Both executors share the pure ORT session/output policy, but the native component
does not instantiate the Managed core or link the aggregate compatibility target.
Both CUDA lanes bind formal model outputs directly to native Buffer allocations.
Managed outputs retain a typed envelope attachment so the egress bridge can
reuse the original TensorList. Validate ORT/native pointer identity for C and
ORT/native/Managed pointer identity for the Managed lane within the allocating
process; mapped addresses in another process need not be equal.

For the native C captures, retain the recursive ELF dependency/symbol evidence,
compile include dependencies, and the actual component-container maps while the
model is loaded and frames are playing. They must exclude project Managed
core/CUDA/HIP/ROS/TensorBundle libraries, the old inference core/node, and the
aggregate compatibility library. Package-level discovery of Managed dependencies
is not the runtime test; do not reject unrelated system libraries merely because
their names contain `managed`.

Binding reports retain schema 1: the native ROS `output_contracts` parameter is
serialized under historical `managed_output_contracts`. Native reports use
`managed_io_contract=compat`, empty `managed_input_contracts`, and zero historical
pool-capacity/wait fields because there is no Managed pool. Direct outputs retain
the native writer/synchronization lifetime claim; dynamic outputs explicitly
report their D2D copy and `pointer_identity=false`.

The original NVIDIA Synthetica asset uses a postprocessor TopK of 300 despite
dynamic ONNX output dimensions. Its batch-one output contracts are
`labels=int64[1,300]`, `boxes=float32[1,300,4]`, and `scores=float32[1,300]`.
The corresponding output payloads are 2400, 4800, and 1200 bytes. These shapes
must match across native/managed launches, POL tests, benchmark graphs, and
copy-audit classifiers; a 100-query preallocation drops frames rather than
changing the model to 100 queries.

Pointer evidence must contain positive numeric addresses (integers or decimal/
hexadecimal strings); equal placeholders such as `unknown` are not identity
evidence. An explicitly incomplete profiler capture or a caller-identified
payload-risk kernel keeps the audit INCONCLUSIVE even with valid binding reports.

Immutable native-message republishing disables intra-process optimization and
uses `publish(*const_message)` without an intermediate message copy. In the
selected Lyrical implementation, the disabled-IPC const-reference path directly
publishes through RMW; the IPC branch clones the message. Check the installed
publisher implementation and actual consumer/trace behavior rather than
assuming another SDK overload or an unconditional zero-copy path.
Unsupported IPC topology may legitimately materialize CPU data; record that
fallback rather than claiming an unconditional inter-process zero-copy path.
Generic dynamic models without output contracts and Managed buffers without
a reusable envelope use explicit copy fallback; they do not satisfy the formal
C/managed inference-boundary no-copy criterion. Standard TensorBundle
conversion remains a host boundary.

Before each audit's captures, execute the pure `test_tensor_list_native` and
retained Managed `test_tensor_list_buffer_adapter` in
`gpu_ros_nvidia_tensor_bundle_compat`, plus pure `test_native_onnx_executor` and
retained `test_onnx_inference_core` in `gpu_ros_onnx_inference`. The audit invokes
each exact CTest name separately with `--no-tests=error` and requires one passing
entry per invocation; a skipped device test cannot satisfy it. The older
Managed-bridge transport probe does not load or validate the native ORT node.
Compare new C trace observations with the preserved same-lane baseline as well
as C versus Managed: a copy added to both paths must not be hidden by their
difference alone.

Complete copy records and CUDA activity are required. CPU shape/decoder
bookkeeping is diagnostic rather than a fixed forbidden-node count. A copy
parser PASS does not establish clean teardown, and numeric detection
observations do not replace copy or lifecycle evidence. Compare suspected
regressions with the preserved source using the same native command, not a
different benchmark or failure stage.

## Historical results and reporting

Historical NVIDIA records remain outside this source release. The pre-fix and
clean-throughput observations remain separate in the archive; the latter did not
rerun fixed-input correctness and copy audits. Do not backfill missing identities
from a current machine.

Follow [result acceptance](result-acceptance.md) and
[result formats](../../../docs/results/README.md). Record the source that was
executed, not the later documentation state. Public records omit private
users, hosts, scheduler/device-instance identifiers, and deployment paths.
