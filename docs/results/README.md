# Result records

These records summarize experiment outcomes with the hardware and software
identity needed to interpret performance, without embedding host, user,
scheduler, device-instance, or deployment-path identifiers. Driver and
container identity are scientific provenance when captured, not sensitive
deployment metadata.

Raw bags, benchmark JSON, profiles, traces, ORT installs, checkpoints, and
datasets remain in an external archive below `${OVG_RESULTS_ROOT}`. Dated
historical records removed from this source release remain byte-for-byte in the
publication archive; this document defines how current records are written and
how archived records are read.

## Current status

The [library-layout verification](library-layout-20260918.md) records one
post-layout campaign: 18 native AMD reports across the three-round matrix and
four native NVIDIA graph reports. Package, build, and static migration checks
passed. AMD Slurm performance is highly variable, so these observations are
not a stable or reproduced baseline; the cause is not established.
These statuses describe the recorded evidence. They do not turn this source
release into a supported runtime or determine whether the experimental source
may be published.

The original strict same-lane verdicts remain in raw evidence and are not
promoted as a baseline. The four NVIDIA commands emitted JSON while their
component containers exited with SIGSEGV. One AMD RT-DETR standard round has a
missing 60-Hz field, which remains missing in the record. No thresholds or
scripts changed.

Methods and acceptance rules live in
[`skills/gpu-ros-experiments/references/`](../../skills/gpu-ros-experiments/references/).
Historical campaign files are retained outside this source release; see
[Historical records](#historical-records).

## Historical records

Dated AMD and NVIDIA result records remain byte-for-byte in the external
publication archive. The inventory ID is
`layout-20260918T061911Z/publication-history-archive/INVENTORY.txt`. These
records preserve their original commands, paths, model identities, and native
statuses. They are evidence, not current build instructions or support claims.

## Current limitations

- AMD measurements used Apptainer. A fresh AMD Docker image build and Docker GPU
  run were not tested.
- NVIDIA measurements reused an existing Docker image. All four formal
  component containers exited with SIGSEGV during teardown after writing JSON.
- AMD Slurm rates vary substantially and are not promoted as a performance
  baseline. The current record retains the missing fixed-rate field.
- AMD RT-DETR copy evidence remains `INCONCLUSIVE`; the HIP preprocessing
  mismatch and lifecycle issues remain deferred.

`REPORT_ONLY` is intentionally a numeric archival status, not a correctness
gate. Keep `PASS`, `FAIL`, and `INCONCLUSIVE` labels exactly as emitted by the
relevant tool; do not infer a stronger status from throughput alone.

## One-checkout provenance

The current tree has one Git checkout. Transport and application components
share the same commit and worktree state. New records therefore use
`monorepo_revision` and repository-wide dirty-state fingerprints; they must not
carry a separate current transport revision or treat `gpu_ros_managed/` as a
nested Git checkout.

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

See [matrix result requirements](../../skills/gpu-ros-experiments/references/result-acceptance.md#matrix-result-requirements)
for per-round fields, aggregation, and raw-evidence requirements.

## Reproduction acceptance

See [reproduction acceptance](../../skills/gpu-ros-experiments/references/result-acceptance.md#reproduction-acceptance)
for identity, tolerance, correctness, and copy-evidence rules.

## Promotion rule

See [the promotion rule](../../skills/gpu-ros-experiments/references/result-acceptance.md#promotion-rule).
Native failures and unresolved evidence remain visible in the result record.
