---
name: gpu-ros-experiments
description: Use this skill whenever a user asks to build, smoke-test, capture, audit, or explicitly benchmark the GPU ROS monorepo. It guides agents through the checked-out AMD Docker/Apptainer, NVIDIA Isaac ROS, and standalone transport runbooks without inventing commands or claiming results that were not run.
compatibility: Requires a shell, a checked-out GPU ROS repository, and the runtime, assets, and external dependencies required by the selected runbook.
---

# GPU ROS experiments


This project skill is distributed under the root Apache-2.0 license. It is
navigation guidance, not a replacement for the repository's runtime gates or
runbooks.
## Start from the checkout

The skill's installation directory is not the repository root. Locate the
checked-out GPU ROS repository first, then run all relative commands from that
root:

```bash
cd /path/to/gpu-ros
git rev-parse --show-toplevel
```

Read the checked-out revision's relevant file under `docs/experiments/` before
running anything. The public project reference is
<https://github.com/gpu-ros/gpu-ros>, but the checked-out runbook is the
authority for the revision being exercised. Keep models, datasets, bags, ORT,
external checkouts, caches, build/install/log state, and results outside Git.
The project works without this skill.

## Choose a workflow

### AMD Docker or Apptainer

Read `docs/experiments/amd-runtime-contract.md` and then the Phase 2A or
Phase 2B runbook. The one checkout is `/workspaces/gpu-ros`; the launcher uses
that path for `GPU_ROS_REPO_ROOT` and `OVG_WORKSPACE_ROOT`. Supply the required
external ORT install, device access, target, assets, and fresh state root. Use
the existing launcher; it retains the device, asset, provider, image, and
optional site-allocation checks:

```bash
./docker/phase2-amd.sh preflight
./docker/phase2-amd.sh up
./docker/phase2-amd.sh colcon
./docker/phase2-amd.sh verify
./docker/phase2-amd.sh shell
```

Inside the runtime, use `phase2 env --verify` and `phase2 assets status` before
a capture. Call the existing capture command from the runbook, for example:

```bash
ros2 run gpu_ros_detection_validation \
  run_amd_phase2a_fixed_input_capture.sh \
  "<new-run-id>"
```

Do not recreate runtime gates in the skill. If the required allocation, device,
ORT, model, or dataset is unavailable, report the external prerequisite as
blocked instead of substituting another input or provider.

### NVIDIA Isaac ROS

Read `docs/experiments/phase1-nvidia.md` or
`docs/experiments/phase2b-nvidia.md`. NVIDIA keeps
`/workspaces/isaac_ros-dev` as the outer workspace and this repository at
`/workspaces/isaac_ros-dev/src/gpu-ros`; pinned external checkouts remain under
`src/nvidia_external`.

From the repository checkout, resolve the existing Compose file and launcher:

```bash
docker compose -f docker-compose.yaml config
./docker/phase1-nvidia.sh up
./docker/phase1-nvidia.sh colcon
./docker/phase1-nvidia.sh verify
./docker/phase1-nvidia.sh shell
```

Inside the runtime, run the capture script at its checked-out path under
`src/gpu-ros/evaluation/gpu_ros_detection_validation/scripts/`. Keep the
NVIDIA image, external-source pins, runtime hooks, assets, and outer-workspace
boundary unchanged.

### Standalone transport

For the backend-neutral transport core, read `transport/README.md` and run the
existing CMake contract either from the monorepo root:

```bash
cmake -S transport -B /tmp/gpu-ros-transport-core \
  -DGPU_ROS_MANAGED_BUILD_TESTING=ON \
  -DGPU_ROS_MANAGED_BUILD_CUDA=OFF \
  -DGPU_ROS_MANAGED_BUILD_HIP=OFF
cmake --build /tmp/gpu-ros-transport-core
ctest --test-dir /tmp/gpu-ros-transport-core --output-on-failure
```

or from `transport/` with `cmake -S .`. ROS adapter builds use the same
monorepo's `transport` and `interfaces` roots. Do not add a sibling checkout.

## Captures and explicit benchmarks

Use a fresh run identifier and preserve the actual command. Fixed-input capture
is a smoke/correctness workflow, not a throughput claim. A benchmark matrix
runs only when the user explicitly requests it; invoke the existing matrix
script named by the selected runbook. There is no `phase2 test`, `phase2
smoke`, or `phase2 benchmark` command.

Report the actual checkout `monorepo_revision`, runtime/provider and model/data
identity, command, exit/status, and external result location. Keep the
unversioned capture record's `monorepo_revision` and
`monorepo_worktree_diff_sha256`; current matrix manifests use schema 3. Never
add a schema marker to a capture or promoted summary, silently fall back to a
provider/model, or claim a pass when a required gate was skipped.

Do not put Slurm allocation/submission, private hostnames, accounts, driver
setup, private paths, or site-specific configuration in this skill. Allocation
is an external prerequisite; the runbooks document only the public runtime
contract and existing project commands.
