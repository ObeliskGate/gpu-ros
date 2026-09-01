# Experiment documentation

This directory contains acceptance-stage runbooks and result summaries. The
runbooks use logical external paths and contain no host, user, scheduler,
device-instance, driver, kernel, container, or deployment identifiers. Result
records include the hardware model while omitting those local identifiers.

- [`experiments/`](experiments/) contains the executable runbooks.
- [`results/`](results/) contains cleaned result summaries and the promotion
  status.

## Source and dependency identity

Record the following in each external run archive rather than in a public
document:

| Item | Policy |
| --- | --- |
| Application | Record the exact repository revision used by the runner. |
| Managed sibling | Record the exact selected sibling revision. |
| Runtime image | Record the immutable image identity in the external archive. |
| ONNX Runtime | Record version, source revision, and applied patches. |
| Model | Record profile, source/checkpoint provenance, and digest. |

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
untracked-content hashes, runtime/ORT identity, model and dataset hashes,
command lines, raw benchmark/audit output, and an explicit
`PASS`, `FAIL`, `INCONCLUSIVE`, or `REPORT_ONLY` status. Keep any sensitive
deployment metadata in an access-controlled archive, not in these documents.

## Current promotion status

The current AMD campaign has completed model export validation, fixed-input
numeric comparison, the HIP copy probe, and the three-lane performance
matrices. The RT-DETR transport audit remains `INCONCLUSIVE` because the trace
parser retained unresolved evidence. COCO val2017 and Python
CPU↔MIGraphX-provider parity are deferred; the removed high-load experiment is
not a gate.
