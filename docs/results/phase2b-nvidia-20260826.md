# NVIDIA Phase 2B results — 2026-08-26 (historical record)

This record covers the first NVIDIA Phase 2B campaign after the application
checkout was updated in an existing Isaac ROS 4.5-compatible runtime. It
compares Config C with the separate Managed transport candidate; it does not
change the Phase 1 A/B/C/D definitions.

The throughput and audit observations below were captured before the
component-runtime teardown fix. They are retained as a historical pre-fix
record; the clean rerun is recorded separately in
`phase2b-nvidia-20260827.md`.

## Reproduction scope

| Field | Value |
| --- | --- |
| Runtime | Isaac ROS 4.5 / ROS 2 Jazzy |
| Hardware model | NVIDIA A100-SXM4-40GB |
| GPU architecture | `not-captured` |
| Driver version | `not-captured` |
| Container base/image identity | Isaac ROS 4.5-compatible; immutable digest `not-captured` |
| ORT | Triton `libonnxruntime.so`, version 1.23.1 |
| ORT source/build revision | `not-captured` in this checked-in historical record |
| Application/Managed revisions | `not-captured` in this checked-in historical record |
| Dataset | `r2bdataset2024_v1/r2b_robotarm`; hash `8eee68848ee1a95e21b1cd44d5d6ba71` |
| Capture policy | Same input, `CAPTURE_PLAYBACK_RATE=0.25`, stamp pairing |
| Release status | `INCONCLUSIVE` pending teardown repair |

Exact revisions, dirty-state hashes, and archive paths remain in the
access-controlled external result archive. The software fields marked
`not-captured` were not promoted from that historical archive; they must be
captured in a new public baseline rather than inferred.

## Throughput observations

The benchmark JSONs include peak-search metrics, fixed 10/30/60 Hz trials,
missed frames, and latency endpoints. “Peak” is the predicted publisher rate.

| Model/lane | Peak prediction (Hz) | Mean output at peak (fps) | Peak misses / sent | Fixed 10 / 30 / 60 Hz misses |
| --- | ---: | ---: | ---: | --- |
| RT-DETR Config C | 102.813 | 100.544 | 4 / 514 | 0 / 0 / 0 |
| RT-DETR Managed | 102.813 | 97.403 | 19.333 / 514 | 0 / 0 / 0 |
| YOLOv8 Config C | 133.750 | 130.588 | 8.667 / 668 | 0 / 0 / 0 |
| YOLOv8 Managed (no timing) | 141.484 | 126.665 | 67 / 707 | 0 / 0 / 0 |

The 30 Hz first/last sent-to-received latency endpoints (milliseconds) were
RT-DETR C `14.540 / 13.805`, RT-DETR Managed `19.181 / 14.463`, YOLOv8 C
`17.181 / 14.511`, and YOLOv8 Managed `16.121 / 13.799`.

All four graph processes wrote valid `Ran 1 test ... OK` reports, but the
component runtime subsequently exited with `SIGSEGV (-11)` during teardown.
The rows are therefore observations, not clean-release `PASS` results. No
bridge timing, ORT profile, or Nsight instrumentation was enabled for these
formal rows.

## Fixed-input numerical comparison

Captures were paired by exact source header stamp with FIFO handling and run
in report-only mode. The RT-DETR audit capture had 388 frames per lane; the
YOLOv8 audit capture had 390. Standalone captures contained 391 and 392 frames.

| Comparison | Paired / reference / candidate | Mean IoU | Minimum IoU | Mean / max score delta | Class match | Unmatched detections | Frame pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RT-DETR C → Managed | 390 / 391 / 391 | 0.999947 | 0.997523 | 0.000060 / 0.002654 | 1.0 | 5 | 0.994898 |
| YOLOv8 C → Managed | 392 / 392 / 392 | 1.000000 | 1.000000 | 0 / 0 | 1.0 | 0 | 1.0 |

The RT-DETR result is numerically close and labels agree, but its maximum
score delta exceeds a strict `1e-3` gate. Exact-stamp pairing leaves one
startup frame unpaired on each side; every paired frame has zero unmatched
detections. Keep this as an observation rather than a strict score-gate pass.

## Provider, pointer, and copy audit

The unified audit returned `PASS` for both models. It found CUDA provider
activity, complete pointer/lifetime evidence, no Managed-only tensor-payload
copy signature, and explainable frame-normalized copy records. RT-DETR CPU
shape/bookkeeping nodes were diagnostic fallback and did not invalidate CUDA
placement.

| Audit | Captured C / Managed frames | Audit status | Managed-only payload copy | Detection report |
| --- | ---: | --- | --- | --- |
| RT-DETR C → Managed | 388 / 388 | PASS | none | report PASS; min IoU 0.998747 |
| YOLOv8 C → Managed | 390 / 390 | PASS | none | PASS; IoU 1.0, score delta 0 |

## Tests and archive

The incremental Release build and targeted runtime suites passed. An earlier
full workspace run had Python docstring-lint failures in the RT-DETR standard
launch module; after correcting that module, the package suite passed 9/9.
The complete workspace suite still requires a fresh promotion run.

Raw bags, profiles, traces, and logs remain outside Git under the configured
external results root. Keep this record synchronized with that immutable
archive and do not rewrite its pre-fix `INCONCLUSIVE` status.
