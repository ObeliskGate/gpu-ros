# Result records

These records summarize experiment outcomes with the hardware model needed to
interpret performance, without embedding host, user, scheduler, driver,
kernel, container, or deployment-path identifiers.
Raw bags, benchmark JSON, profiles, traces, ORT installs, checkpoints, and
datasets remain in an external archive below `${OVG_RESULTS_ROOT}`.

## Current status

| Area | Status |
| --- | --- |
| NVIDIA reference work | Historical observations; keep separate from current AMD results. |
| AMD Phase 2A standard paths | Completed for the current model/assets used by the campaign. |
| AMD Phase 2B three-lane matrices | Completed for YOLOv8 and RT-DETRv2 R50. |
| RT-DETRv2 export and CPU validation | Passed; exact model digest is recorded in the source profile. |
| CPU ↔ Managed HIP report-only comparison | Completed with exact stamp pairing; see the AMD campaign record. |
| HIP copy probe | Completed with JSON, CSV, and manifest output. |
| AMD transport audit | RT-DETR remains `INCONCLUSIVE` because trace evidence contains unresolved ambiguity. |
| COCO val2017 | Deferred by decision. |
| Managed high-load gate | Removed from the formal experiment. |
| CPU ↔ MIGraphX Python provider parity | Deferred until a correctly built Python binding is available. |

`REPORT_ONLY` is intentionally a numeric archival status, not a correctness
gate. Keep `PASS`, `FAIL`, and `INCONCLUSIVE` labels exactly as emitted by the
relevant tool; do not infer a stronger status from throughput alone.

## Required summary fields

Promoted result summaries should include only reproducibility-safe fields:

```text
run_id=
runtime=
application_head=
managed_head=
ort_version=
ort_commit=
model_profile=
model_sha256=
dataset_sha256=
hardware_model=
graph_or_lane=
command=
status=PASS|FAIL|INCONCLUSIVE|REPORT_ONLY
```

Hostnames, usernames, IP addresses, scheduler IDs, partition names, GPU
targets, driver versions, kernel versions, image filenames, and absolute
deployment paths are excluded. Hardware model is the deliberate exception and
must be recorded in each result. If an access-controlled archive needs the
other fields, keep them outside the repository and refer to it by an opaque run
identifier.

## Promotion rule

Numbers become current only when the corresponding runbook gates pass and the
summary points to immutable raw evidence. A failure or `INCONCLUSIVE` result
remains visible; it is not replaced by a silent rerun or profile fallback.
