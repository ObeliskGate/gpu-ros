# Result records

These records summarize experiment outcomes with the hardware and software
identity needed to interpret performance, without embedding host, user,
scheduler, device-instance, or deployment-path identifiers. Driver and
container identity are scientific provenance when captured, not sensitive
deployment metadata.
Raw bags, benchmark JSON, profiles, traces, ORT installs, checkpoints, and
datasets remain in an external archive below `${OVG_RESULTS_ROOT}`.

## Current status

| Area | Status |
| --- | --- |
| NVIDIA reference work | Historical observations; keep separate from current AMD results. |
| AMD Phase 2A standard paths | Completed for the current model/assets used by the campaign. |
| AMD Phase 2B three-lane matrix execution | Completed for YOLOv8 and RT-DETRv2 R50; RT-DETR numerical baseline is pending per-round archive promotion. |
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
hardware_model=
gpu_arch=
driver_version=
rocm_version=
ros_distro=
migraphx_version=
migraphx_source=
container_recipe=
container_recipe_revision=
container_base_image=
image_digest=
application_head=
managed_head=
ort_version=
ort_commit=
ort_patchset_sha256=
ort_fingerprint=
model_profile=
model_sha256=
dataset_sha256=
graph_or_lane=
command=
status=PASS|FAIL|INCONCLUSIVE|REPORT_ONLY
```

Hostnames, usernames, IP addresses, scheduler IDs, partition names, device
instance IDs, and absolute deployment paths stay outside the public record.
Hardware model, GPU target, driver version, ROS/ROCm/MIGraphX/ORT identity, and
the image recipe or digest must be recorded in every promoted result. If a
field was not captured, write `not-captured` rather than silently omitting it.

## Matrix result requirements

An execution `PASS` from the matrix shell script means that all nine lane
processes completed and emitted JSON. It is not by itself a numerical
reproduction claim. A public matrix summary must include, for every formal
lane (standard, staged-control, and direct Managed):

- the three per-round peak predictions and mean output rates;
- the per-round peak missed/frames-sent counts;
- the aggregate peak values and the declared aggregation rule;
- per-round and aggregate fixed 10, 30, and 60 Hz output rates and
  missed/frames-sent counts;
- model/data/software/hardware identity and the exact matrix command; and
- links or opaque IDs for the raw JSON and logs in the external archive.

The canonical matrix aggregation is the arithmetic mean of the three round
reports for scalar rate/count fields; per-round values remain authoritative and
must not be discarded. The lane order is rotated by the runner as recorded in
`matrix_manifest.txt`.

## Reproduction acceptance

The following semantics apply to a new run against a published baseline:

1. The model profile and SHA, dataset tree hash, graph/lane, ROS/ROCm/
   MIGraphX/ORT identities, container recipe or digest, application and sibling
   revisions, and GPU model must match. A different GPU model is a new
   performance observation, not a failed reproduction.
2. All three rounds must complete. For each lane, the aggregate peak
   prediction and aggregate mean output must be within **±10%** of the baseline;
   each individual round may deviate by at most **±15%**. Fixed 10/30/60 Hz
   output must be within **±2%** of the requested rate and have zero missed
   frames. These are internal same-hardware-model acceptance rules, not a claim
   that different GPUs have equal throughput.
3. Correctness is evaluated separately from throughput. The formal same-bag
   gate requires at least 20 paired frames, mean and pairwise IoU at least
   0.99, mean and pair score delta at most 0.001, 100% paired and overall frame
   pass rates, class match 100%, and zero unmatched detections. A
   `REPORT_ONLY` comparison keeps its numbers but never becomes a correctness
   `PASS` merely because the aggregate numbers look close.
4. A copy audit is a separate evidence claim. `INCONCLUSIVE` means the trace
   did not resolve the requested direction or ownership; it must not be
   rewritten as zero-copy or as a detection failure.

Runs on the same GPU family but a different model, driver, or provider package
may be reported as `COMPARABLE` in an external analysis, but should not replace
the canonical baseline or receive a `reproduced` label.

## Promotion rule

Numbers become current only when the corresponding runbook gates pass and the
summary points to immutable raw evidence. A matrix shell `PASS` without the
per-lane/per-round values is an execution result, not a promoted performance
baseline. A failure or `INCONCLUSIVE` result remains visible; it is not
replaced by a silent rerun or profile fallback.
