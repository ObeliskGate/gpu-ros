# Experiment documentation

This directory contains acceptance-stage runbooks and result summaries. The
runbooks use logical external paths and contain no host, user, scheduler,
device-instance, or deployment identifiers. Runtime and result documents do
publish scientific/software provenance needed to reproduce an equivalent run:
hardware model, GPU target, driver/ROCm stack when captured, ROS and provider
versions, and the container recipe or immutable image identity.

- [`experiments/`](experiments/) contains the executable runbooks.
- [`experiments/amd-runtime-contract.md`](experiments/amd-runtime-contract.md)
  is the canonical AMD runtime definition. Site-specific scheduler and
  deployment adapters are optional and are not hidden prerequisites.
- [`results/`](results/) contains cleaned result summaries and the promotion
  status.

## Source and dependency identity

Record the following in each external run archive. The non-sensitive
scientific/software fields also belong in the public result summary:

| Item | Policy |
| --- | --- |
| Application | Record the exact repository revision used by the runner. |
| Managed sibling | Record the exact selected sibling revision. |
| Hardware | Record the public GPU model and architecture target. |
| Runtime image | Record the public recipe/base identity and, when available, the immutable image digest. |
| ROS/ROCm/provider | Record ROS 2, ROCm/HIP, MIGraphX package/source, and provider versions. |
| ONNX Runtime | Record version, source revision, build fingerprint, and applied patches. |
| Model | Record profile, source/checkpoint provenance, and digest. |

Host, user, scheduler, device-instance, and deployment-path values may remain
in an access-controlled archive. They are not substitutes for the public
software and hardware provenance above.

The NVIDIA `auto` profile selects the historical Synthetica asset. CPU, ROCm,
and MIGraphX `auto` select the formal RT-DETRv2 R50 profile. A selected profile
must exist and pass its digest check; there is no silent fallback.

## Asset and source boundary

Model files, checkpoints, bags, COCO data, external checkouts, ORT
source/build/install trees, caches, traces, and benchmark output stay outside
the repository. The tracked source manifests are reproducibility inputs, not
submodules or license grants. External checkout licenses remain authoritative.

## Build policy

- NVIDIA builds enable only the CUDA/NITROS components required by their
  reference lane.
- AMD and CPU builds disable application CUDA/NITROS transports and enable the
  requested MIGraphX or CPU provider explicitly.
- AMD validation uses the project-built external ORT install selected for the
  run; build-only fallback libraries are never formal runtime selections.
- Scheduler and runtime checks are performed by the launcher. Their local
  identifiers are retained only in the external archive.

## Evidence required for promotion

Each result archive must include application and sibling revisions, diff and
untracked-content hashes, hardware/software/runtime identity, model and dataset
hashes, command lines, raw benchmark/audit output, and an explicit `PASS`,
`FAIL`, `INCONCLUSIVE`, or `REPORT_ONLY` status. Keep sensitive deployment
metadata in an access-controlled archive, not in these documents.

## Current promotion status

The current AMD campaign has completed model export validation, fixed-input
numeric comparison, the HIP copy probe, and execution of the three-lane
performance matrices. The RT-DETR matrix's per-round numerical baseline still
needs to be promoted from the external JSON archive. The RT-DETR transport
audit remains `INCONCLUSIVE` because the trace parser retained unresolved evidence. COCO val2017 and Python
CPU↔MIGraphX-provider parity are deferred; the removed high-load experiment is
not a gate.
