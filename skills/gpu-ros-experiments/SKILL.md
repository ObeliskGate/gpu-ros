---
name: gpu-ros-experiments
description: Use when asked to build, test, capture, audit, reproduce, or explicitly benchmark GPU ROS. Selects the checked-out standalone managed, AMD Docker or Slurm Apptainer, and NVIDIA reference methods; preserves runtime and asset identity and distinguishes wrapper success from numeric, copy, and teardown evidence.
compatibility: Requires a checked-out GPU ROS repository and the tools, authorized hardware, runtime, and external assets needed by the selected method.
---

# GPU ROS experiments

Use the checked-out revision's methods, not an installed skill's assumed paths.
This skill and its references are Apache-2.0 under the repository license.
Users can read them without installing a skill.

## Establish scope and source

Locate the repository before resolving relative paths. It contains
`gpu_ros_managed/` and `gpu_ros_object_detection/`; neither is an aggregate ROS
package. The installed skill directory is not the checkout. Read
[the method index](references/README.md) and choose one of the three scenarios
below.

Identify the requested work: library build, package verification, fixed-input
comparison, transport audit, or performance experiment. `launch_test` on a
benchmark graph runs a performance measurement. A request to verify a change
does not by itself authorize performance matrices, Phase 1 TensorRT sweeps,
COCO, model export, or provider-parity work.

For a migration, preserve the actual pre-change worktree, including dirty and
untracked source, in an independent checkout with real Git metadata. Record
the revision and fingerprints, not HEAD alone. Use separate empty build,
install, log, and result roots for each source version. Keep inputs and the
verified dependency runtime unchanged; no old project overlay may satisfy a
new package or launch path.

## Scenario 1: standalone managed library

Read [the managed library](../../gpu_ros_managed/README.md) and the local
managed agent guide. A source-only machine can build the backend-neutral core
without ROS or a GPU SDK. From the repository root, select an external build
directory:

```bash
cmake -S gpu_ros_managed -B "${VERIFY_ROOT}/managed-core" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_DISABLE_FIND_PACKAGE_ament_cmake=TRUE \
  -DBUILD_TESTING=ON -DGPU_ROS_MANAGED_BUILD_TESTING=ON \
  -DGPU_ROS_MANAGED_BUILD_CUDA=OFF \
  -DGPU_ROS_MANAGED_BUILD_HIP=OFF
cmake --build "${VERIFY_ROOT}/managed-core" --parallel 2
ctest --test-dir "${VERIFY_ROOT}/managed-core" \
  --output-on-failure --no-tests=error
```

Enable a GPU backend only when it is part of the requested change and its SDK
and device are available. A CTest skip is not a device PASS. ROS builds use
individual package directories or the tracked runtime defaults, not the
managed parent CMake project. Use the index's existing CPU check selection for
namespace and Python-tool changes; do not collect ROS-dependent tests in a
source-only environment.

## Scenario 2: AMD detection

Read [the AMD runtime contract](references/amd-runtime-contract.md), then
[Phase 2A](references/phase2a-amd.md) for standard transport or
[Phase 2B](references/phase2b-managed.md) for managed/staged graphs.

Select `OVG_RUNTIME=docker` explicitly by default. Inspect the real GPU target,
ROCm/provider versions, existing image, external ORT fingerprint and loaded
libraries, available space, mounts, and assets before building. The AMD
workspace remains `/workspaces/gpu-ros`, not the detection subdirectory.
`OVG_ORT_ROOT` must select the complete external install; image
`/opt/onnxruntime` is not a fallback.

For Docker, create an external run-specific override using the verified image
ID, fresh project state, original assets, and the existing device/runtime
settings. Follow the contract's `config`, `up -d --no-build`, and `exec -T`
sequence. Keep the override on every command. Do not switch to a launcher
branch that silently omits it or rebuild the dependency image for a path change.

If an authorized Slurm compute allocation has no Docker, select Apptainer
explicitly. Use `OVG_APPTAINER_INSTANCE=0`, `OVG_REQUIRE_SLURM=1`, and
`OVG_PREPARE_ASSETS=0`; clear `APPTAINERENV_HOME`. Use the existing launcher for
preflight, colcon, and verification, then the contract's exact non-interactive
bind/environment table for further payloads. Execute the mounted checkout
entrypoint, not an old copy inside the SIF. Never run GPU work on a login node
or add site submission commands to public methods.

Reuse existing formal model and data bytes. Validate the formal RT-DETRv2 SHA
and ONNX I/O contract; inspect the actual type, source, and consumer of any
compatibility asset instead of trusting its filename. Missing export tools in
the AMD SIF do not justify installing another provider or regenerating models.
Use [RT-DETRv2 validation](references/rtdetrv2-validation.md) only for the
requested asset work. The offline helper is
`gpu_ros_object_detection/tools/phase2-assets`.

Run native package tests and inspect `colcon test-result`. For captures and
audits, use [detection validation](references/detection-validation.md) without
changing model, provider, thresholds, geometry, or shutdown parameters. An
explicitly requested full AMD matrix uses the existing
`run_amd_phase2b_benchmark_matrix.sh` for each requested model, preserving its
three-round rotation and fixed-rate configuration. Do not create a temporary
graph or another player.

The layout campaign exercised AMD through Apptainer. AMD Docker GPU execution
and a fresh dependency-image build were not tested by that campaign; do not
claim they were.

## Scenario 3: NVIDIA reference

Read [the NVIDIA managed method](references/phase2b-nvidia.md). Use
[Phase 1](references/phase1-nvidia.md) only when that wider reference scope is
requested. The outer workspace is `/workspaces/isaac_ros-dev`, the checkout is
`src/gpu-ros`, and pinned external sources are under `src/nvidia_external`.

Inspect GPU/driver identity, the existing image ID, external commits, actual
ORT libraries, model bytes, and per-file input hashes. Preserve
`OVG_NVIDIA_EXTERNAL_SOURCE_ROOT` and the existing external assets volume. A
profile without a known expected SHA cannot be described as having passed a
fixed-digest check. A historical 32-character framework hash is not SHA-256.

Use the method's external Compose override, empty project build/install/log
binds, new results, and `up -d --no-build`. Keep `exec -T` and the same override
for builds, tests, and experiment payloads. Do not source an old project
overlay before a fresh build. Run the selected colcon tests and the explicit
CTest packages omitted by the defaults. Check Release, loaded library identity,
package prefixes, and all five installed `gpu_ros_nvidia_reference` launches;
old owners must not provide duplicate launch files. Generic conversion
components and the boundary launch still belong to compat.

Use the six existing capture lanes when the requested comparison needs them.
Run each lane and audit serially on one GPU. For explicitly requested Phase 2B
performance, run the four existing benchmark graphs separately, in the order
specified by the method. Inventory and archive each newly emitted native JSON
before starting the next graph. There is no NVIDIA three-round runner to infer
from the AMD method.

## Interpret and report evidence

Read [result acceptance](references/result-acceptance.md) before declaring a
PASS. Preserve command exit, component startup/runtime/teardown, numeric
comparison, copy evidence, and performance statistics separately. Native
capture and launch-test wrappers can return zero after a component SIGSEGV.
A copy parser PASS does not establish clean shutdown. Inspect component logs
and post-exit process lists alongside wrapper return codes and JSON.

Compare each migration lane with its own baseline using the strict stamp-based
command. Keep cross-lane REPORT_ONLY evidence separate. Do not ignore unmatched
frames, use index pairing, filter away score differences, or loosen thresholds
to pass a migration. An unchanged-source control investigates repeatability;
it does not erase a failed comparison. A known fault requires matching raw
same-lane, same-parameter, same-stage evidence. New or unexplained faults block
acceptance even if another graph has a similar historical crash.

For performance, retain per-round peak/mean, missed/sent, actual fixed-rate
fields, latency, aggregation, and native lifecycle status. Missing fields stay
MISSING. Compare hardware/model/provider/runtime/source identities before
using a historical number. Report missing historical identities and tolerance
inconsistencies; do not fill them from the current runtime. For a comparable
out-of-tolerance observation, use preserved pre-change source and the current
device as a control before attributing the difference to a migration.

Show measured values, historical values, deltas, comparability, failures, and
raw evidence locations before updating result summaries. Keep the executed
source fingerprints even if documentation is edited afterward. A failed gate
blocks promotion. Continue only diagnostics or measurements already within the
documented scope. For a structure-only migration, preserve algorithms and
runtime policy; behavioral fixes require separate scope.

Methods belong in these references; public result summaries belong in
`docs/results/`. Raw artifacts remain external. Preserve historical result
bytes and field names. Do not publish private users, hosts/IPs, scheduler or
device-instance IDs, absolute deployment paths, credentials, or asset bytes.
Report an unavailable prerequisite precisely rather than claiming an unrun
branch passed.
