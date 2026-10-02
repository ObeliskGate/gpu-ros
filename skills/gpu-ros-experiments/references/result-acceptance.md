# Result acceptance

These rules distinguish execution, correctness, copy evidence, and performance
reproduction. They define when measurements may be described as reproduced or
promoted evidence; they do not decide whether this experimental source release
may be published. Result formats and historical identities are described in
[Result records](../../../docs/results/README.md).

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
3. Fixed-input detection comparisons are observations, not numeric acceptance
   gates. Offline bag-comparison reports use schema
   `phase2b_detection_report_only_v2` and report score/IoU/class metrics,
   matched and unmatched detections, and paired/unpaired frame coverage.
   `REPORT_ONLY` means a report was generated, not that outputs are equal.
   Numeric differences or unmatched coverage do not fail a valid comparison;
   invalid inputs/data or report-writing errors remain errors.
   Metrics without comparable samples are `null` with sample counts.
4. A copy audit is a separate evidence claim. `INCONCLUSIVE` means the trace
   did not resolve the requested direction or ownership; it must not be rewritten
   as zero-copy or as a detection failure.

Runs on the same GPU family but a different model, driver, or provider package
may be reported as `COMPARABLE` in an external analysis, but should not replace
the canonical baseline or receive a `reproduced` label.

## Promotion rule

Promote performance baselines only after their runbook gates pass and the
summary points to immutable raw evidence. A matrix shell `PASS` without
per-lane/per-round values is an execution result, not a promoted performance
baseline. A failure or `INCONCLUSIVE` result remains visible; it is not replaced
by a silent rerun or profile fallback. Detection reports record observations,
not numerical acceptance verdicts.

## Migration comparisons and independent failures

Compare each lane with the same lane from the actual pre-change worktree,
using identical model and input bytes, provider, runtime, parameters, and test
code. Preserve dirty and untracked source in an independent checkout with real
Git metadata. Use separate empty build, install, log, and result directories;
an old overlay must not satisfy a new package or launch path.

Use the default exact-timestamp pairing with complete, unfiltered inputs;
repeated timestamps pair FIFO, followed by deterministic class-unconstrained
IoU-greedy detection matching. Optional index pairing or score/detection
filtering is allowed for diagnostics, but record the method/options and make
its coverage clear. A filtered or index-paired report is not the full-coverage
default observation.

Record command exit codes and component startup, runtime, and teardown logs
separately. A capture or launch-test wrapper can return zero after its component
exits with SIGSEGV. A copy report PASS does not establish clean teardown, and a
detection report does not override lifecycle or copy evidence. Missing
fixed-rate fields remain MISSING, not zero.

Preserve raw evidence for any baseline error using the same lane, parameters,
and failure stage. New execution, input-validity, model-contract, lifecycle, or
copy errors block acceptance. Numeric detection differences and coverage
changes are observations, not comparator failures. An unchanged-source control
can investigate repeatability; it does not dismiss an independent error. Stop
promotion when a required independent gate fails, preserving the failure and
every subsequent in-scope diagnostic or measurement.

Compare historical performance numbers only after checking identity and
aggregation. Keep missing driver, image, revision, and model identity fields
explicit. If old performance numbers violate an existing tolerance, report the
inconsistency rather than changing the rule. For an out-of-tolerance comparable
throughput observation, run the same native command on the preserved source and
current device before attributing the difference to a migration. Do not change
algorithms, provider settings, warmup, or benchmark configuration to make a
structural change pass.
