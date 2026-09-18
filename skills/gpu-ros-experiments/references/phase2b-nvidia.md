# Phase 2B NVIDIA managed transport

This method compares the NVIDIA/NITROS Config C reference with managed
TensorBundle bridges around ORT CUDA for RT-DETR and YOLOv8. It is separate
from the [Phase 1 A/B/C/D matrix](phase1-nvidia.md). Managed transport must
preserve pointer ownership without an extra tensor-payload copy at the
inference boundary; it does not make the complete pipeline copy-free.

## Workspace and identity

The outer workspace is `/workspaces/isaac_ros-dev`. The repository is mounted
once at `src/gpu-ros`, and pinned external sources are mounted read-only at
`src/nvidia_external`. `gpu_ros_object_detection/` owns detection facilities,
not the outer workspace or monorepo identity.

Before running anything, verify GPU model/architecture and driver, the existing
image ID, external commits against
`gpu_ros_object_detection/external/nvidia-isaac-ros.repos`, loaded ORT library
identity, source revision and dirty/untracked hashes, model bytes, and each
R2B file. Preserve original asset volumes. A model profile without a fixed
expected SHA needs independent provenance and a recorded actual digest; it has
not passed a known-digest check. Do not treat the framework's 32-character
input hash as SHA-256.

For migration verification, retain the actual pre-change worktree in an
independent checkout with real Git metadata. Both versions use the same verified
image, external sources, and asset bytes, but separate empty build/install/log
and result directories. Do not pull over a dirty experiment checkout or source
an old project overlay to fill a missing package.

## Reuse the dependency image

Set `REPO_ROOT`, `VERIFY_ROOT`, and `OVG_NVIDIA_EXTERNAL_SOURCE_ROOT` to the
verified checkout, external run state, and pinned external-source directory.
Create a run-specific override outside Git at `NV_OVERRIDE` with:

- the literal verified existing image ID for service `dev`;
- a unique Compose project and fresh binds for outer-workspace build, install,
  and log directories;
- a new results bind at `/workspaces/ovg-results`;
- the existing `assets` volume declared `external: true` with its verified
  name, rather than a new empty volume.

Keep the NVIDIA runtime hook, GPU reservation, host networking, external-source
binds, and container working directory from the tracked Compose file. Resolve
the combined configuration before starting it:

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
  bash -lc 'set -e; source /opt/ros/jazzy/setup.bash; colcon build'
```

Run the selected colcon tests and the packages not selected by the NVIDIA test
defaults. Keep every command failure visible:

```bash
docker compose -f "${NV_COMPOSE}" -f "${NV_OVERRIDE}" exec -T dev \
  bash -lc 'set -e
    source /opt/ros/jazzy/setup.bash
    source install/setup.bash
    colcon test --event-handlers console_direct+
    colcon test-result --all --verbose
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

## Non-interactive runtime payloads

Capture, comparison, audit, and benchmark commands below run through the same
Compose prefix. Set `PAYLOAD` to the requested existing command sequence; do not
enter an interactive shell or introduce another graph runner:

```bash
docker compose -f "${NV_COMPOSE}" -f "${NV_OVERRIDE}" exec -T dev \
  bash -lc "set -e; ulimit -c 0
    source /opt/ros/jazzy/setup.bash
    source /workspaces/isaac_ros-dev/install/setup.bash
    ${PAYLOAD}"
```

If intentionally collecting evidence after a known failure, record each exit
code explicitly and return the accumulated failure status. `set -e` cannot
reveal a child error swallowed by a capture or launch-test wrapper; inspect
component logs and post-exit processes too.

## Fixed-input correctness

Use [detection validation](detection-validation.md#nvidia) for the six existing
capture lanes and parameters. Preserve model/input bytes, provider, image
geometry, RT-DETR size policy, and source timestamps. Use unique output names,
profiling disabled, playback 0.25, drain 30 seconds, at least 100 messages, and
stop grace/termination windows of 60/20 seconds when matching this protocol.

Each migration lane is compared only with its own pre-change bag using the
[strict acceptance command](detection-validation.md#compare-detections).
C versus managed and C versus D are separate cross-lane observations. Keep
REPORT_ONLY reports as such; index pairing and ignoring unmatched frames do not
satisfy the migration gate. Record coverage as well as numeric differences.

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

Complete copy records and CUDA activity are required. CPU shape/decoder
bookkeeping is diagnostic rather than a fixed forbidden-node count. A copy
parser PASS does not waive failed strict numeric comparison or a new teardown
fault. Compare suspected regressions with the preserved source using the same
native command, not a different benchmark or failure stage.

## Historical results and reporting

Historical NVIDIA records remain outside this source release. The pre-fix and
clean-throughput observations remain separate in the archive; the latter did not
rerun fixed-input correctness and copy audits. Do not backfill missing identities
from a current machine.

Follow [result acceptance](result-acceptance.md) and
[result formats](../../../docs/results/README.md). Record the source that was
executed, not the later documentation state. Public records omit private
users, hosts, scheduler/device-instance identifiers, and deployment paths.
