# GPU ROS documentation

This directory contains result records and the provenance rules used to read
them. Experiment methods live under
[`skills/gpu-ros-experiments/`](../skills/gpu-ros-experiments/); there is no second
runbook copy in `docs/`.

- [Result records](results/README.md) explain source identities and archival
  formats, and link the current measurement summary.
- [Library-layout verification](results/library-layout-20260918.md) records
  the current campaign, known limitations, and raw evidence identifiers.
- [Historical records](results/README.md#historical-records) describe archived
  dated measurements. Their commands and paths are not current instructions.
- [Method index](../skills/gpu-ros-experiments/references/README.md) covers
  preparation, runtime execution, capture, audit, and requested benchmarks.
- [Acceptance rules](../skills/gpu-ros-experiments/references/result-acceptance.md)
  define numeric and reproduction gates for evidence; they do not determine
  whether this experimental source release may be published.

Users can read the method files directly; no installation is required. The
method index is the public entry point for runtime procedures.

## Source and dependency identity

Record the executed monorepo revision and dirty/untracked fingerprints,
public GPU model and architecture, driver, runtime image or SIF digest,
ROS/provider versions, actual loaded ORT libraries, model provenance and
hashes, input-data hashes, commands, exit codes, and raw artifact IDs. Preserve
missing fields as `not-captured`. A framework's historical 32-character data
hash is not a file SHA-256.

Transport and detection share one checkout identity. Current matrix manifests
use schema 3; captures and promoted summaries remain unversioned. Legacy
records retain their original fields and interpretation. Do not rewrite a
historical record or substitute the final documentation worktree's fingerprints
for the source that was measured.

## Assets and privacy

Keep external checkouts, ORT installs, model/data bytes, bags, traces, profiles,
caches, runtime images, and generated output outside Git. Manifests and locks
record compatibility inputs; they do not grant redistribution rights.

Public summaries include scientific software and hardware identity. Private
users, hosts, IPs, scheduler IDs, device-instance identifiers, and absolute
deployment paths remain in the access-controlled archive. Those private values
do not replace the public identity fields.

## Reading a result

A build, fixed-input comparison, transport audit, component teardown, and
throughput measurement establish different facts. Preserve each native status.
A wrapper may return zero after a child crashes; inspect the component log.
`REPORT_ONLY` is not a correctness PASS, and `INCONCLUSIVE` copy evidence is
not proof of zero-copy.

Historical campaigns do not certify the current source or runtime. The current
layout record keeps strict comparisons and lifecycle/copy evidence as
observations, not as a promoted baseline. AMD Slurm measurements are unstable,
and the documented AMD Docker path was not tested in that campaign. See the
current record and acceptance rules rather than inferring support from JSON.
