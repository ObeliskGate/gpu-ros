# GPU ROS documentation

This directory contains the public experiment runbooks and result-record
semantics for the GPU ROS monorepo. Runbooks describe commands and software
contracts; they do not embed hostnames, users, scheduler/account identifiers,
device-instance identifiers, private deployment paths, model bytes, or SIF
filenames.

- [`experiments/`](experiments/) contains build, capture, audit, and benchmark
  runbooks.
- [`experiments/amd-runtime-contract.md`](experiments/amd-runtime-contract.md)
  is the canonical AMD container contract for Docker and Apptainer.
- [`experiments/phase1-nvidia.md`](experiments/phase1-nvidia.md) and
  [`experiments/phase2b-nvidia.md`](experiments/phase2b-nvidia.md) describe the
  NVIDIA outer-workspace and pinned external-source boundary.
- [`results/`](results/) contains cleaned historical records and the rules for
  interpreting new provenance.

The repository works without the optional project skill. Agents that install
[`skills/gpu-ros-experiments/SKILL.md`](../skills/gpu-ros-experiments/SKILL.md)
should still read the checked-out revision of these runbooks before running a
command.

## Source and dependency identity

Record the following in each external run archive. The non-sensitive
scientific/software fields also belong in a public result summary:

| Item | Policy |
| --- | --- |
| Monorepo | Record the exact `monorepo_revision` used by the runner, plus tracked-diff and untracked-content fingerprints. |
| Transport | Transport packages are part of the same monorepo identity; do not record a separate current checkout or transport revision. |
| Hardware | Record the public GPU model and architecture target. |
| Runtime image | Record the public recipe/base identity and, when available, the immutable image or SIF digest. |
| ROS/ROCm/provider | Record ROS 2, ROCm/HIP, MIGraphX or CUDA package/source, and provider versions. |
| ONNX Runtime | Record version, source revision, build fingerprint, and applied patches. |
| Model and data | Record the selected profile, source/checkpoint provenance, model digest, and dataset digest. |

Host, user, scheduler, device-instance, and deployment-path values may remain
in an access-controlled archive. They are not substitutes for the public
software and hardware provenance above. Historical records may retain their
original fields; do not rewrite their contents to match the current layout.

The NVIDIA `auto` profile selects the historical Synthetica asset. CPU, ROCm,
and MIGraphX `auto` select the formal RT-DETRv2 R50 profile. A selected profile
must exist and pass its digest check; there is no silent fallback.

## Asset and source boundary

Model files, checkpoints, bags, COCO data, external checkouts, ORT
source/build/install trees, caches, traces, SIF images, and benchmark output
stay outside the repository. The tracked manifests and lock files are
reproducibility inputs, not submodules or license grants. External checkout
licenses and model/data terms remain authoritative.

## Build policy

- The standalone transport core builds from `cmake -S transport` at the
  monorepo root or `cmake -S .` in `transport/`; it does not require ROS or a
  GPU SDK. Optional CUDA/HIP and ROS adapters are selected explicitly.
- NVIDIA builds enable only the CUDA/NITROS components required by their
  pinned reference lane and keep external sources under the outer workspace's
  `src/nvidia_external`.
- AMD builds disable application CUDA/NITROS transport and enable the requested
  MIGraphX or CPU provider explicitly.
- AMD validation uses the external ORT install selected for the run; image
  `/opt/onnxruntime` content is never a formal runtime fallback.
- Runtime and scheduler checks are performed by the launchers. Site allocation
  adapters are optional; their local identifiers remain outside public records.

## Evidence required for promotion

Each new result archive should include the monorepo revision, worktree diff and
untracked-content hashes, hardware/software/runtime identity, model and dataset
hashes, command lines, raw benchmark/audit output, and an explicit `PASS`,
`FAIL`, `INCONCLUSIVE`, or `REPORT_ONLY` status. Keep sensitive deployment
metadata in an access-controlled archive, not in these documents.

Matrix manifests and fixed-input captures use separate compatibility rules;
see [`results/README.md`](results/README.md). A matrix v3 record has one
monorepo identity, while matrix v2 retains its two historical repository
identities. A capture log has no schema marker: `monorepo_revision` identifies
new one-checkout output, while the legacy application/managed fields identify
old two-checkout output. Promoted summaries follow the same field-presence rule
and never gain a matrix schema marker.

## Current promotion status

The current AMD campaign has completed model export validation, fixed-input
numeric comparison, the HIP copy probe, and execution of the three-lane
performance matrices. The RT-DETR matrix's per-round numerical baseline is
recorded in the external archive. The RT-DETR transport audit remains
`INCONCLUSIVE` because trace evidence retained unresolved ambiguity. COCO
val2017 and Python CPU-to-MIGraphX-provider parity are deferred; the removed
high-load experiment is not a gate. These historical statuses do not establish
that a new checkout or runtime has passed its gates.
