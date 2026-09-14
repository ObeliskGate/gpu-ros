# Contributing

GPU ROS is a ROS 2 monorepo. Keep changes scoped to the affected source,
interface, evaluation, documentation, or runtime configuration area. Preserve
package directory basenames and ROS package names (`gpu_ros_*`); top-level
`transport`, `interfaces`, `perception`, `compat`, and `evaluation` directories
are classification boundaries, not new package names.

## Before changing code

Read the closest `AGENTS.md` and the canonical runbook for the lane being
changed:

- `transport/AGENTS.md` for managed-buffer ownership, stream/device, and
  standalone CMake rules;
- `docs/experiments/amd-runtime-contract.md` for AMD runtime paths and gates;
- `docs/experiments/phase1-nvidia.md` or `phase2b-nvidia.md` for NVIDIA;
- `docs/results/README.md` for public provenance fields and historical-record
  handling.

Keep the one-repository contract intact. AMD uses `/workspaces/gpu-ros`; NVIDIA
keeps `/workspaces/isaac_ros-dev` as its outer workspace and mounts this
repository at `src/gpu-ros`. Do not restore managed sibling variables,
second source binds, old path aliases, or unrelated environment renames.

## Local development

From the repository root, a ROS build uses the package roots selected by the
relevant tracked colcon profile. The standalone transport core can be built
without ROS or a GPU SDK:

```bash
cmake -S transport -B /tmp/gpu-ros-transport-core \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_DISABLE_FIND_PACKAGE_ament_cmake=TRUE \
  -DGPU_ROS_MANAGED_BUILD_TESTING=ON \
  -DGPU_ROS_MANAGED_BUILD_CUDA=OFF \
  -DGPU_ROS_MANAGED_BUILD_HIP=OFF
cmake --build /tmp/gpu-ros-transport-core
ctest --test-dir /tmp/gpu-ros-transport-core --output-on-failure
```

The same commands may be run from `transport/` by changing the source option
to `cmake -S .`. Do not put build, install, log, cache, model, bag, trace, or
result output in a commit. Use the AMD or NVIDIA launcher for a real runtime;
do not run GPU or large compilation workloads on a shared login node.

## Tests and evidence

Run focused checks for the layer you changed. For ROS packages, source the
matching ROS setup and workspace overlay, then select affected packages with
colcon. For runtime/provider or capture changes, follow the full gate sequence
in the relevant runbook and retain raw output outside Git.

A reported result must identify the actual monorepo revision, worktree diff and
untracked-content fingerprints, runtime/image or SIF identity, ORT/provider
identity, model/data hashes, command, and external result location. A smoke run
is not a performance reproduction. Do not run a complete benchmark matrix or
full workspace campaign merely to validate a documentation/path change.

Matrix manifests written by the current tree use schema 3 and one
`monorepo_revision`. Fixed-input captures and promoted summaries remain
unversioned and use the documented `monorepo_*` fields. Do not add schema
markers to captures or summaries, emit old/new aliases together, or rewrite
historical archives.

## Assets and licenses

The project does not redistribute credentials, model weights, checkpoints, ONNX
files, engines, datasets, bags, traces, profiles, SIF images, caches, or
private deployment metadata. Use the offline `tools/phase2-assets` helper for
user-held inputs. A recorded digest establishes compatibility only; it does not
grant a license or permission to redistribute an asset.

Preserve upstream copyright and file-level modification notices when editing
derived code. Keep the root and transport notice boundaries intact and consult
`THIRD_PARTY_NOTICES.md` before changing ported files. Do not reopen historical
provenance or license audits as part of a normal source change.

## Pull requests

Describe the affected top-level area, package names, runtime lane, and focused
checks run. Include links to changed runbooks when command contracts change.
Mark unavailable hardware, assets, or external services as blocked rather than
claiming a pass. Keep private hostnames, scheduler/account identifiers,
absolute deployment paths, and credentials out of commits and public reports.

The optional `skills/gpu-ros-experiments/SKILL.md` may help an agent navigate the
runbooks, but installing a skill is never required to build or use this project.
