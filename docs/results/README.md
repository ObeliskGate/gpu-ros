# Result records

These records summarize experiment outcomes with the hardware and software
identity needed to interpret performance, without embedding host, user,
scheduler, device-instance, or deployment-path identifiers. Driver and
container identity are scientific provenance when captured, not sensitive
deployment metadata.

Raw bags, benchmark JSON, profiles, traces, ORT installs, checkpoints, and
datasets remain in an external archive below `${OVG_RESULTS_ROOT}`. Historical
result archives in this directory retain their original bytes and field names;
this document defines how new records are written and how old records are
read.

## Current status

| Area | Status |
| --- | --- |
| NVIDIA reference work | Historical observations; keep separate from current AMD results. |
| AMD Phase 2A standard paths | Completed for the current model/assets used by the campaign. |
| AMD Phase 2B three-lane matrix execution | Completed for YOLOv8 and RT-DETRv2 R50; numerical baselines are recorded with per-round values and external raw archives. |
| RT-DETRv2 export and CPU validation | Passed; the exact model digest is recorded in the source profile. |
| CPU ↔ managed HIP report-only comparison | Completed with exact stamp pairing; see the AMD campaign record. |
| HIP copy probe | Completed with JSON, CSV, and manifest output. |
| AMD transport audit | RT-DETR remains `INCONCLUSIVE` because trace evidence contains unresolved ambiguity. |
| COCO val2017 | Deferred by decision. |
| Managed high-load gate | Removed from the formal experiment. |
| CPU ↔ MIGraphX Python provider parity | Deferred until a correctly built Python binding is available. |

`REPORT_ONLY` is intentionally a numeric archival status, not a correctness
gate. Keep `PASS`, `FAIL`, and `INCONCLUSIVE` labels exactly as emitted by the
relevant tool; do not infer a stronger status from throughput alone.

## One-checkout provenance

The current tree has one Git checkout. Transport and application components
share the same commit and worktree state. New records therefore use
`monorepo_revision` and repository-wide dirty-state fingerprints; they must not
carry a separate current transport revision or treat `transport/` as a nested
Git checkout.

Record these fields in each external archive and include the reproducibility-safe
ones in a public summary:

| Item | New-record policy |
| --- | --- |
| Monorepo | Exact `monorepo_revision`, `monorepo_diff_head_binary_sha256`, `monorepo_untracked_paths`, `monorepo_untracked_content_sha256`, and `monorepo_dirty`. |
| Runtime | Public recipe/base identity and, when available, immutable image or SIF digest. |
| Hardware | Public GPU model, architecture target, and driver version when captured. |
| ROS/provider | ROS 2, ROCm/HIP, MIGraphX or CUDA package/source, and provider versions. |
| ONNX Runtime | Version, source revision, build fingerprint, and applied patches. |
| Model/data | Selected profile, source/checkpoint provenance, model digest, and dataset digest. |
| Execution | Graph or lane, exact command, raw-output IDs, and explicit status. |

If a value was not captured, write `not-captured` rather than silently omitting
it. Hostnames, usernames, IP addresses, scheduler IDs, partition names, device
instance IDs, and absolute deployment paths stay outside the public record.

## Matrix manifests: schema 2 and schema 3

Matrix manifests are the only records in this transition with an explicit
schema marker.

### Legacy matrix v2

A manifest with `schema_version=2` retains two independent repository
identities: `application_revision` and the `gpu_ros_managed_*` state fields.
Those values describe the historical two-checkout layout and must be displayed
as such. Do not reinterpret the old managed revision as a current transport
version or replace its historical diff fingerprints with monorepo fingerprints.

### New matrix v3

A new manifest is written with `schema_version=3` and one monorepo identity:

```text
schema_version=3
monorepo_revision=<checkout commit>
monorepo_diff_head_binary_sha256=<repository diff fingerprint>
monorepo_untracked_paths=<newline-separated paths or empty>
monorepo_untracked_content_sha256=<untracked-content fingerprint>
monorepo_dirty=<true|false>
```

The writer preserves the existing asset, runtime, model, and lane metadata. It
does not emit separate `gpu_ros_managed_*` repository-state fields, carry the
pre-migration transport SHA forward, or emit old and new aliases together.
Unknown explicit matrix versions are unsupported. A missing required revision
or state field is incomplete evidence; never guess a layout or substitute the
current SHA.

When a consumer displays component revisions, a v3 record resolves both the
application and transport components to `monorepo_revision` for display only.
It does not rewrite the record or make two-layout and one-layout diff hashes
interchangeable.

## Fixed-input capture logs: unversioned format

Capture logs deliberately do **not** gain a `schema_version`. Their actual
command remains part of the record. A new one-checkout capture contains at
least:

```text
monorepo_revision=<checkout commit>
monorepo_worktree_diff_sha256=<repository worktree diff fingerprint>
<actual capture command and emitted metadata>
```

The field-presence discriminator is independent of matrix schema versions. A
legacy capture containing both `application_revision` and
`gpu_ros_managed_revision` retains its historical two-checkout interpretation.
A new capture containing `monorepo_revision` uses the one-checkout layout and
must not also emit the legacy revision fields. A capture with missing required
revision fields, or with conflicting old and new fields, is incomplete or
ambiguous and must not be guessed into a layout.

The repository has no generic manifest-ingestion parser. Existing readers, if
one is introduced at a genuine read boundary, must reject unknown matrix
versions and ambiguous capture fields while preserving historical `MISSING`
values. Do not add a conversion CLI, parser library, duplicate JSON format, or
new public compatibility API for this transition.

## Promoted summaries: unversioned format

Promoted summaries also remain unversioned. New summaries replace the old
`application_head` and `managed_head` pair with exactly one field:

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
monorepo_revision=
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

Historical summaries retain their original `application_head` and
`managed_head` fields. Do not add a matrix schema marker to a summary. When a
consumer displays component revisions from a new summary, it may resolve both
to `monorepo_revision` for display without rewriting the summary. Historical
and new repository-wide diff fingerprints are not interchangeable.

## Public migration ancestry

This one-time note identifies the public import ancestry using only the
rewritten commits present in this repository:

- rewritten application parent: `6263dadd285396da0ea586d7d4e2de6919c5a635`;
- rewritten transport parent: `60ede6806c4c86f1a3fad5bc0dca58d04a119ffa`;
- import merge: `2d4ba289bc8b03ae25dbca340680dad482aad2b8`.

All three commits are reachable from the public `main` history. This ancestry
note is not a numerical reproduction claim, does not identify private source
repositories, and does not change the identity rules above.

## Matrix result requirements

An execution `PASS` from the matrix shell script means that all nine lane
processes completed and emitted JSON. It is not by itself a numerical
reproduction claim. A public matrix summary must include, for every formal lane
(standard, staged-control, and direct managed):

- the three per-round peak predictions and mean output rates;
- the per-round peak missed/frames-sent counts;
- aggregate peak values and the declared aggregation rule;
- per-round and aggregate fixed 10, 30, and 60 Hz output rates and
  missed/frames-sent counts;
- model/data/software/hardware identity and the exact matrix command; and
- links or opaque IDs for raw JSON and logs in the external archive.

The canonical matrix aggregation is the arithmetic mean of the three round
reports for scalar rate/count fields; per-round values remain authoritative and
must not be discarded. The lane order is rotated by the runner as recorded in
`matrix_manifest.txt`.

## Reproduction acceptance

The following semantics apply to a new run against a published baseline:

1. The model profile and SHA, dataset tree hash, graph/lane, ROS/ROCm/
   MIGraphX/ORT identities, container recipe or digest, monorepo revision, and
   GPU model must match. A different GPU model is a new performance observation,
   not a failed reproduction.
2. All three rounds must complete. For each lane, the aggregate peak prediction
   and aggregate mean output must be within **±10%** of the baseline; each
   individual round may deviate by at most **±15%**. Fixed 10/30/60 Hz output
   must be within **±2%** of the requested rate and have zero missed frames.
   These are internal same-hardware-model rules, not a claim that different GPUs
   have equal throughput.
3. Correctness is evaluated separately from throughput. The formal same-bag
   gate requires at least 20 paired frames, mean and pairwise IoU at least 0.99,
   mean and pair score delta at most 0.001, 100% paired and overall frame pass
   rates, class match 100%, and zero unmatched detections. A `REPORT_ONLY`
   comparison keeps its numbers but never becomes a correctness `PASS` merely
   because aggregate numbers look close.
4. A copy audit is a separate evidence claim. `INCONCLUSIVE` means the trace
   did not resolve the requested direction or ownership; it must not be rewritten
   as zero-copy or as a detection failure.

Runs on the same GPU family but a different model, driver, or provider package
may be reported as `COMPARABLE` in an external analysis, but should not replace
the canonical baseline or receive a `reproduced` label.

## Promotion rule

Numbers become current only when the corresponding runbook gates pass and the
summary points to immutable raw evidence. A matrix shell `PASS` without the
per-lane/per-round values is an execution result, not a promoted performance
baseline. A failure or `INCONCLUSIVE` result remains visible; it is not replaced
by a silent rerun or profile fallback.
