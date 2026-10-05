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

### Isaac ROS 5.0 migration campaign (2026-10-05)

This campaign uses the official Isaac ROS 5.0 / ROS 2 Lyrical package pairing,
external ONNX Runtime 1.30.0, the original SyntheticaDETR TopK-300 and YOLOv8s
ONNX bytes, and the original R2B robot-arm input. No model, graph, QoS, executor,
warmup, benchmark duration, or report schema was substituted.

The fresh build and tests executed monorepo revision
`5079939063bff473b5780a2cf4ea9446797cabf4`. Fixed-input captures and transport
audits executed `05805e6a38c709fdb39590a65d94915b613ec6b1`, which adds the
Lyrical recorder's required `--topics` syntax. Four performance commands and the
isolated eight-package CPU build also executed `05805e6`. The captured tracked
diffs are empty; the runtime and CPU records additionally capture no untracked
source paths. Later documentation revisions do not replace these
executed-source identities.

| Check | Observed result |
| --- | --- |
| Fresh explicit NVIDIA build | All 12 selected packages built. |
| Selected package test aggregation | 192 reported tests, zero errors/failures, 25 skips; separate focused CTests for detection common, both decoders, and validation also passed. The aggregate includes CTest and underlying test-result rows, not 192 distinct behavioral cases. |
| Original-model POL | Native and managed RT-DETR tests passed against the original TopK-300 model. |
| Installed native consumer | Exported CMake target linked; dtype, shape, strides, header, and payload round-trip checks passed on the GPU. |
| Isolated CPU profile | Eight packages built with CUDA/ROCm/MIGraphX/native transport disabled, in separate empty state; 255 aggregated test-result rows, zero errors/failures, 48 skips. Installed inference-node dependencies contain no CUDA/HIP/project-native-compatibility libraries. This is not an AMD GPU run. |
| Transport probe | Both native boundary components loaded, ROS endpoints became ready, one launch test passed, and the component exited cleanly. Real payload/pointer checks are supplied by the separate adapter tests and installed consumer, not the endpoint probe. |
| Six full-input capture lanes | RT-DETR C/D/managed: 396/396/396 output frames; YOLOv8 C/D/managed: 396/397/396. RT-DETR D exited `-11`; the other five unprofiled captures exited cleanly. |
| Detection observations | Four current cross-lane and six historical-before/current-after reports generated with default exact-stamp pairing and full coverage; numerical differences and unmatched coverage remain observations, not pass/fail thresholds. |
| Two native/managed copy audits | Provider, pointer/boundary, and copy analyses passed. Each profiled managed component exited `-11`; original audit exit status remains 1 and `OVERALL_AUDIT_STATUS=FAIL`. |
| Four approved performance graphs | All four native `launch_test` commands returned 0, emitted byte-exact archived SDK JSON, logged clean component exits, and left no component container or bag player. No nine-lane matrix expansion was run. |
| Experiment acceptance | Functional and copy results accepted with the user's explicit non-blocking SDK shutdown exception; lifecycle itself remains failed. |
| AMD Lyrical runtime | Not exercised: no authorized GPU compute allocation. |

#### Full-coverage detection observations

All rows use exact timestamps, complete input bags, and no score/detection
filters. Class agreement among matched detections is 1.0 in every row; that
does not describe unmatched detections or unpaired frames.

| Current comparison | Paired / reference / candidate frames | Candidate unpaired | Matched detections | Unmatched detections / affected frames | IoU mean / min | Max absolute score delta |
| --- | --- | --- | --- | --- | --- | --- |
| RT-DETR C → D | 396 / 396 / 396 | 0 | 1535 | 0 / 0 | 0.9999 / 0.9961 | 0.0029 |
| RT-DETR C → managed | 396 / 396 / 396 | 0 | 1535 | 0 / 0 | 0.9999 / 0.9926 | 0.0179 |
| YOLOv8 C → D | 396 / 396 / 397 | 1 | 1031 | 5 / 4 | 0.9192 / 0.8106 | 0.0203 |
| YOLOv8 C → managed | 396 / 396 / 396 | 0 | 1032 | 0 / 0 | 1.0000 / 1.0000 | 0.0000 |

The six historical same-lane comparisons use archived
`df69877403107127dec6b89662efacd694bc9dc5` bags, not newly fabricated baseline
captures. They cross A100-SXM4-40GB/Isaac 4.5/Jazzy/ORT 1.23.1 to
L40/Isaac 5.0/Lyrical/ORT 1.30.0 and are migration observations, not a
controlled same-runtime reproduction. Historical recorder TERM fallbacks and
masked wrapper exits do not establish clean baseline lifecycle. Historical
untracked-source fingerprints and refreshed post-reboot identity remain missing.

| Historical → current lane | Paired / reference / candidate frames | Candidate unpaired | Matched detections | Unmatched detections / affected frames | IoU mean / min | Max absolute score delta |
| --- | --- | --- | --- | --- | --- | --- |
| RT-DETR C | 392 / 392 / 396 | 4 | 1516 | 22 / 9 | 0.9998 / 0.9934 | 0.0211 |
| RT-DETR D | 393 / 393 / 396 | 3 | 1522 | 16 / 6 | 0.9998 / 0.9934 | 0.0224 |
| RT-DETR managed | 393 / 393 / 396 | 3 | 1521 | 17 / 7 | 0.9998 / 0.9960 | 0.0190 |
| YOLOv8 C | 392 / 392 / 396 | 4 | 1019 | 13 / 4 | 0.9999 / 0.9667 | 0.0017 |
| YOLOv8 D | 395 / 395 / 397 | 2 | 1028 | 7 / 3 | 0.9999 / 0.9997 | 0.0013 |
| YOLOv8 managed | 384 / 384 / 396 | 12 | 993 | 39 / 12 | 0.9999 / 0.9667 | 0.0013 |

#### Native/managed transport observations

Both audits retained 396 output frames in each lane. Offline continuation
analyzed the completed profiles/traces while preserving the original failed
managed teardown. It did not rerun the graph or rewrite the audit verdict.

| Model | CUDA memcpy events C → managed | D2D event / byte delta | D2H event / byte delta | H2D event / byte delta | Unknown-direction event / byte delta |
| --- | --- | --- | --- | --- | --- |
| RT-DETR | 37694 → 14545 | -12675 / -35845153712 | -1873 / -1281744560 | -2641 / -1907428360 | -5960 / -310621160 |
| YOLOv8 | 20387 → 11755 | -1398 / -5535667200 | -515 / -1796755200 | -5222 / -1906415524 | -1497 / -206242164 |

Each parser reports `Managed boundary status: PASS`, no additional managed
boundary memcpy signature or payload-copy rate, and a valid full-coverage
detection comparison. These are local pointer/boundary and observed copy
claims, not a whole-pipeline or inter-process zero-copy promise.
RT-DETR has 493 unique CPU and 1055 unique CUDA ORT nodes in each lane
(24650/52750 node events across 50 profiles); CPU placement is an observation,
not an invented zero-CPU gate. YOLOv8 has 175 unique CUDA nodes / 8750 events
per lane and no CPU nodes observed.

#### Four native performance commands

Run from the outer NVIDIA workspace, using the same selected image, checkout,
asset/ORT binds, and Compose override for every command:

```bash
launch_test src/gpu-ros/gpu_ros_object_detection/benchmarks/gpu_ros_rtdetr_config_c_graph.py
launch_test src/gpu-ros/gpu_ros_object_detection/benchmarks/gpu_ros_rtdetr_managed_graph.py
launch_test src/gpu-ros/gpu_ros_object_detection/benchmarks/gpu_ros_yolov8_config_c_graph.py
launch_test src/gpu-ros/gpu_ros_object_detection/benchmarks/gpu_ros_yolov8_managed_graph.py
```

All four commands, log tees, and SDK-report archives returned 0. Each actual
component exit was clean and before/after container/player snapshots were
empty. No Nsight or ORT profiling override was added to performance commands.
These clean unprofiled performance exits do not repair or erase the profiled
capture failures above.

SDK config 0.30.0 retains 5-second duration, five peak trials, publisher bounds
10–1000 Hz, fixed 10/30/60 Hz tests, buffer size 1, 5-second pre-trial wait, and
resource profiling. The image's `BasicPerformanceCalculator.conclude_performance`
discards the maximum and minimum independently for each metric, then averages
the remaining three values for mean metrics. Fractional missed-frame fields are
native SDK aggregates. This is not a three-round campaign or custom pooling;
fixed-rate fields and raw SDK JSON remain unchanged.

| Lane | SDK peak prediction (Hz) | Mean output (fps) | Peak missed / sent (SDK aggregate) | 10-Hz output (fps) | 30-Hz output (fps) | 60-Hz output (fps) |
| --- | --- | --- | --- | --- | --- | --- |
| RT-DETR C | 92.343750 | 86.388145 | 23 / 461 | 10.216906 | 30.265069 | 60.375157 |
| RT-DETR managed | 100.078125 | 94.761099 | 20.333333 / 500 | 10.213623 | 30.231849 | 60.332480 |
| YOLOv8 C | 136.015625 | 135.438360 | 0 / 680 | 10.204057 | 30.232196 | 60.277012 |
| YOLOv8 managed | 149.218750 | 140.810875 | 34.666667 / 746 | 10.204474 | 30.204991 | 60.286061 |

Every fixed-rate row has missed/sent counts of 0/50, 0/150, and 0/300 at
10, 30, and 60 Hz respectively. The 10-Hz output fields are +2.04% to +2.17%,
outside a literal ±2% reproduction tolerance; no tolerance or estimator was
changed to relabel them. These are new-device observations, not a promoted
same-hardware reproduction baseline. In this campaign only, managed mean output
is +9.69% for RT-DETR and +3.97% for YOLOv8 relative to C; this does not establish
a stable migration speedup.

The earlier four A100/4.5 reports have mean outputs 102.983766, 98.870657,
153.101195, and 157.868137 fps in the same lane order, but their component
containers exited `-11` and their pooling formula is missing. Do not apply the
new SDK's aggregation retrospectively or attribute cross-device/stack rate
differences to migration.

| Lane | Immutable SDK report ID | SHA-256 |
| --- | --- | --- |
| RT-DETR C | `performance/rtdetr-c/reports/r2b-log-20261005-103857.json` | `70e91f0b414fe5dc567676c7cdf8c556e6917e0efb9b6dd17def38d437a16237` |
| RT-DETR managed | `performance/rtdetr-managed/reports/r2b-log-20261005-104236.json` | `10b0020843a73fe99cdc46a6ae8af89e503eceeecfc284081323769e00dc50ea` |
| YOLOv8 C | `performance/yolov8-c/reports/r2b-log-20261005-104614.json` | `1dd974b73cbce78b1719e55ad50d28f329ee4645c5e30ff270ad9511ffcac796` |
| YOLOv8 managed | `performance/yolov8-managed/reports/r2b-log-20261005-105002.json` | `cb724c0160803f124b30244dbfcfd3ec50a643562e52b86035b9af184866b72e` |

The four corresponding raw command/log IDs are
`full-nvidia-05805e6-performance-{rtdetr-c,rtdetr-managed,yolov8-c,yolov8-managed}`.
The aggregation source observation is
`paired-sdk-native-five-trial-aggregation-method-read-only`.

#### Independent SDK lifecycle repair task

This work item is separated from migration experiment acceptance at the user's
request; no speculative library pinning or SDK replacement was applied.
An untraced native core and the same process's pre-shutdown library mappings
locate the crashing instruction at `libpad_node.so+0xc1039`,
`cuda_buffer_backend::CudaVmmIPCManager::FDDispatcher::run()`. The component
library is absent from the crash mappings while its worker thread remains
alive. ONNX Runtime, its CUDA provider, and project inference/compatibility
libraries remain mapped. This identifies the SDK code-unload/worker-lifetime
failure; it does not claim a completed repair or clean component teardown.
The profiled YOLOv8 exit shares the observed teardown stage but has not been
independently attributed by a YOLOv8 core; do not substitute the RT-DETR stack
as a captured YOLOv8 stack.

Reproduce with the unchanged full-input managed RT-DETR capture under Nsight,
retaining component exit status and collecting a native core plus library
mappings before SIGINT. The repair must establish worker/code-owner lifetime
without changing model bytes, component graphs, executor/QoS, provider policy,
or measurement parameters. Acceptance requires clean affected native and
managed exits for both models, including the profiled audit path; an
intermittently clean rerun alone is not a fix.

Correlated core SHA-256:
`23c4cf1660e963775a72c08341e30c27bccfc53eaa0f128a7b30e2cdbeab572a`.
Raw diagnostic IDs:
`rtdetr-05805e6-authorized-native-core-live-mapping`,
`paired-sdk-pad-crash-symbol-and-module-marker`; snapshots:
`native-core-mapping/maps-2/before-drain.maps`.
The temporary host core configuration was restored to its captured original
value after each collection window.

The correlated 3,510,194,176-byte native core, matching eight SDK worker
libraries, and pre-shutdown maps were copied off the temporary host into a
private diagnostic archive. No model or project source bundle is included;
these binaries/core are not part of the public source release.

| Private diagnostic artifact ID | SHA-256 |
| --- | --- |
| `sdk-lifecycle-offhost-evidence/correlated-component.core.gz` | `a2dbbcbcae7611b7658c14c0cd6b29cc38697d743b76bee1bcf0a263d5d8a7cb` |
| `sdk-lifecycle-offhost-evidence/paired-sdk-worker-binaries.tar.gz` | `eb0381e8216b264d0899eec716557497a7c97d45c3d8c3566ac748e6c9049e61` |
| `run-05805e6-experiment-raw-artifacts.tar.gz` (six capture bags/logs, comparison reports, two copy-audit inputs/traces, four performance reports/logs) | `f6e6fdbefe2cc152f7d9cd25bc076094b5d44cb9958e28de18ff955530387715` |

The last archive preserves raw experiment evidence separately from source,
models, external installations, and build state.

#### Runtime and asset identity

GPU: NVIDIA L40; driver `595.91.07`. Official base image:
`sha256:8b08a75a95692d64ba68cd244b04519cd837789c33781340b166427043dcc42d`.
Completed dependency image:
`sha256:7c0f74ca2ac4752134cf0f106547feb59146bc7cfdab3a0ea796028c848b7807`.
The paired SDK baseline was preserved: 403 additional paired packages, no
baseline upgrades/removals. The image export's initial disk-full failure is
retained; successful resumed unpack is not recorded as a fabricated build exit.

| Artifact | SHA-256 |
| --- | --- |
| ORT 1.30.0 library | `292591ed61befc515112570ae8eb9bb0d47716cd0ed60c38865800de4e829544` |
| ORT CUDA provider | `32fb1e28e5eafe8a39d52ca2e1e9c7333f3285968d288b43485cbaa32af6686e` |
| SyntheticaDETR | `8b372a7dcc9b6730633307f7dd844df39f51e6816fc3b138f45c509bd6c71653` |
| YOLOv8s | `d6e22418dd1acc69a232a1b297c01dfc785842fd11a4a84546c84e14cdeb235c` |
| R2B metadata | `16728fa72538ea25e5f2b7b6ec3a35578a96d206844ffcafa826175a6f547b08` |
| R2B MCAP | `56e878f8eb2f14a1514c323da99e27ecb4e71de2df9fc930682e776d121693f5` |

ORT source: `f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7`.
CUDA transport plugin 0.1.2 source:
`7e723061d03b347bedea69009b003f33e9b53314`.
Runtime pairing: CUDA 13.2, TensorRT 10.16, cuDNN 9.22.

Primary raw artifact IDs (commands, stdout/stderr, and actual exits retained):
`full-nvidia-5079939-clean-build-and-rtdetr-pol-resume`,
`full-nvidia-5079939-all-tests-consumer-and-benchmark-contract`,
`full-nvidia-5079939-installed-consumer-and-sdk-contract`,
`full-nvidia-05805e6-isolated-eight-cpu-build-and-tests`,
`full-nvidia-05805e6-actual-native-transport-probe`,
`full-nvidia-05805e6-six-lane-fixed-input-capture`,
`full-nvidia-05805e6-four-cross-lane-observations`,
`full-nvidia-05805e6-six-historical-before-after-observations`,
`full-nvidia-05805e6-rtdetr-native-managed-transport-audit`,
`full-nvidia-05805e6-yolov8-native-managed-transport-audit-dispatched`,
`rtdetr-05805e6-offline-copy-lifecycle-fail-resume-corrected`,
`yolov8-05805e6-offline-copy-observation-lifecycle-remains-fail`.
The undispatched initial YOLO audit log contains no executed payload and is not
audit evidence.

### Archived library-layout campaign

The [library-layout verification](library-layout-20260918.md) records one
post-layout campaign: 18 native AMD reports across the three-round matrix and
four native NVIDIA graph reports. Package, build, and static migration checks
passed. AMD Slurm performance is highly variable, so these observations are
not a stable or reproduced baseline; the cause is not established.
These statuses describe the recorded evidence. They do not turn this source
release into a supported runtime or determine whether the experimental source
may be published.

The archived detection-comparison output was generated by earlier
threshold-based tooling. Its numeric verdict labels remain historical tool
output, not current project failures. The four NVIDIA commands emitted JSON while
their component containers exited with SIGSEGV. One AMD RT-DETR standard round
has a missing 60-Hz field, which remains missing in the record. Current
comparison policy and report semantics are described below; archived results
are not rewritten.

Methods and acceptance rules live in
[`skills/gpu-ros-experiments/references/`](../../skills/gpu-ros-experiments/references/).
Historical campaign files are retained outside this source release; see
[Historical records](#historical-records).

## Current detection comparison reports

The offline detection comparator writes schema
`phase2b_detection_report_only_v2`. A successful report has status
`REPORT_ONLY`, meaning the report was generated; it does not mean numeric
equality. Score, IoU, class, matched/unmatched detection, and paired/unpaired
frame observations are recorded without numeric acceptance thresholds.
Numeric differences and unmatched coverage do not make a valid comparison fail
or return a nonzero status. Invalid or unreadable inputs, invalid data, and
report-writing errors remain nonzero.

The default method uses exact timestamp pairing, FIFO pairing for repeated
timestamps, then deterministic class-unconstrained IoU-greedy matching.
Reports retain full coverage. Explicit index matching or score/detection filters
remain diagnostic options; their selected method and filters are recorded and
must not be presented as a full-coverage default comparison. When no samples
can support a metric, the metric is `null` with its sample count rather than a
fabricated equality value.

## Historical records

Dated AMD and NVIDIA result records remain byte-for-byte in the external
publication archive. The inventory ID is
`layout-20260918T061911Z/publication-history-archive/INVENTORY.txt`. These
records preserve their original commands, paths, model identities, and native
statuses. They are evidence, not current build instructions or support claims.

## Archived library-layout campaign limitations

- AMD measurements used Apptainer. A fresh AMD Docker image build and Docker GPU
  run were not tested.
- NVIDIA measurements reused an existing Docker image.
- AMD Slurm rates vary substantially and are not promoted as a performance
  baseline. The current record retains the missing fixed-rate field.
- The archive records `INCONCLUSIVE` AMD RT-DETR copy evidence. Its HIP
  preprocessing and lifecycle observations belong to that dated campaign record
  and must not be mistaken for a current-run result.

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
monorepo_untracked_paths=<one-line semicolon-separated, shell-escaped paths or empty>
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

New matrix manifests include
`dataset_tree_hash_algorithm=sha256sum-sorted-relative-path-v1`. The runner
sorts dataset files by their relative `./...` paths in the C locale, emits
each file's `sha256sum` line from within the dataset directory, and hashes
that stream. The absolute mount root does not enter this digest. Earlier
matrix dataset hashes included absolute file paths and did not carry this
algorithm marker; neither those hashes nor the different
`phase2-assets.tree_digest` algorithm can be compared directly with the new
`dataset_tree_sha256`. Historical manifests retain their original values.

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
status=PASS|FAIL|INCONCLUSIVE|REPORT_ONLY
```

Summary status describes the recorded execution/evidence; it is not a numeric
detection-equivalence verdict. Detection-comparison reports use `REPORT_ONLY`.

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
for identity, performance tolerances, comparison observations, and copy-evidence rules.

## Promotion rule

See [the promotion rule](../../skills/gpu-ros-experiments/references/result-acceptance.md#promotion-rule).
Native failures and unresolved evidence remain visible in the result record.
