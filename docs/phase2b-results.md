# Phase 2B Validation Results

This report keeps the historical archive and the current NVIDIA validation
observation together. The revised AMD direct Managed HIP implementation is
present, but no allocation-backed AMD real-model result is claimed until its
required capture, contract, topology, and high-load gates complete. Procedures
and interpretation rules are in phase2b-managed-transport.md.

The current NVIDIA campaign used the existing Isaac ROS 4.5 development
container after a fast-forward `git pull`; it did not rebuild or change the
Phase 1 A/B/C/D definitions. Its raw archive remains outside Git on the NV
host. The benchmark reports below were produced before the teardown fix, so
their numerical data is retained but their process-exit status was initially
`INCONCLUSIVE` for clean-release promotion.

## Current NVIDIA Phase 2B campaign — 2026-08-26

The campaign compares Config C with the separate Managed transport candidate
for RT-DETR and YOLOv8. Both lanes use the same NVIDIA/NITROS preprocessing and
decoder; Managed only wraps the ORT CUDA boundary with the two TensorBundle
bridges. The formal YOLOv8 Managed row is the rerun with bridge timing disabled;
the timing-enabled run is diagnostic and is not used as the result.

| Field | Value |
| --- | --- |
| GPU / driver | NVIDIA A100-SXM4-40GB / 595.84 |
| Runtime | Isaac ROS 4.5, ROS 2 Jazzy |
| Image | `sha256:52ba16a38a6c03eeb4533c6006d44459140e0517b59483b186df798b79ec1453` |
| Application HEAD | `f427e1ab5dbbba6ccb22d2026edcb0b119a3f484` |
| Managed HEAD | `caaf6cbb599bfb194671bc957f096443b94bef01` |
| ORT | Triton `libonnxruntime.so`, version 1.23.1 |
| Dataset | `r2bdataset2024_v1/r2b_robotarm`, hash `8eee68848ee1a95e21b1cd44d5d6ba71` |

### Throughput observation

| Model/lane | Peak prediction (Hz) | Mean output at peak (fps) | Peak misses / sent | Fixed 10 / 30 / 60 Hz misses |
| --- | ---: | ---: | ---: | --- |
| RT-DETR Config C | 102.813 | 100.544 | 4 / 514 | 0 / 0 / 0 |
| RT-DETR Managed | 102.813 | 97.403 | 19.333 / 514 | 0 / 0 / 0 |
| YOLOv8 Config C | 133.750 | 130.588 | 8.667 / 668 | 0 / 0 / 0 |
| YOLOv8 Managed (no timing) | 141.484 | 126.665 | 67 / 707 | 0 / 0 / 0 |

The 30 Hz first/last sent-to-received latency endpoints (milliseconds) were
RT-DETR C `14.540 / 13.805`, RT-DETR Managed `19.181 / 14.463`, YOLOv8 C
`17.181 / 14.511`, and YOLOv8 Managed `16.121 / 13.799`; the raw JSONs retain
the peak-search and other fixed-rate endpoint values.

Each graph wrote a valid benchmark JSON and reported `Ran 1 test ... OK`.
In that campaign the component container then exited with `SIGSEGV (-11)` during
teardown; this did not invalidate the already-written throughput measurements,
but it made clean-release promotion `INCONCLUSIVE`.

The teardown race was subsequently fixed in the ONNX/NITROS lifecycle. The
node now drains active inference callbacks before destroying its IO object, and
the NITROS IO object counts callbacks through input readiness conversion before
destroying its CUDA stream. A short real-input regression (15 seconds of the
same R2B bag at 0.25x) then exited cleanly for all four ORT lanes: RT-DETR C,
RT-DETR D, YOLOv8 C, and YOLOv8 Managed. The ONNX package suite passed 17/17.
The formal throughput rows above still need one clean rerun after this fix
before they are promoted as final release results.

### Fixed-input numerical comparison

The formal comparator used exact source-header stamp matching in report-only
mode. Standalone captures contained 391 RT-DETR and 392 YOLOv8 messages; the
audit captures contained 388 and 390 messages.

| Comparison | Paired / reference / candidate | Mean IoU | Minimum IoU | Mean / max score delta | Class match | Unmatched detections |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RT-DETR C → Managed | 390 / 391 / 391 | 0.999947 | 0.997523 | 0.000060 / 0.002654 | 1.0 | 5 |
| YOLOv8 C → Managed | 392 / 392 / 392 | 1.000000 | 1.000000 | 0 / 0 | 1.0 | 0 |

RT-DETR is numerically close and labels agree, but the maximum score delta is
above a strict `1e-3` gate, so it is retained as an observation rather than a
strict score-gate pass. The aggregate `5` unmatched detections are not five
paired-frame model mismatches: exact-stamp pairing leaves one first frame
unpaired on each side, containing 2 reference detections and 3 candidate
detections. All 390 paired frames have `unmatched_count=0`. The first source
header stamps differ by 33.36 ms, while the next source stamp is identical;
this is a capture/playback startup boundary artifact.

The independent audit capture had no unpaired frames and one paired-frame
unmatched candidate detection (frame index 178). It is a class-22 box with
score `0.600004554` and is almost fully contained in the other candidate
class-22 box (box IoU about `0.544`). The NVIDIA RT-DETR decoder threshold is
`0.6`, so this is consistent with a tiny C-versus-Managed score perturbation
at the emission boundary (the reference does not emit that query). As a
diagnostic only, filtering detections below `0.6001` removes this one extra
box and gives zero unmatched detections; the formal `min_score=0` comparison
is retained unchanged. YOLOv8 passes the reported numerical checks.

### Provider, pointer, and copy audit

The unified audit returned `PASS` for both models. It found CUDA provider
activity, complete pointer/lifetime evidence, no Managed-only tensor-payload
copy signature, and explainable frame-normalized copy records. RT-DETR CPU
shape/bookkeeping nodes were reported as diagnostic fallback and did not
invalidate the CUDA placement result.

| Audit | C / Managed frames | Status | Detection comparison |
| --- | ---: | --- | --- |
| RT-DETR C → Managed | 388 / 388 | PASS | paired; min IoU 0.998747 |
| YOLOv8 C → Managed | 390 / 390 | PASS | paired; IoU 1.0, score delta 0 |

The raw audit roots are approximately 851 MB and 176 MB. They remain on the
NV host; no single audit command exceeded 1 GB, although the two raw audit
directories together are about 1.1 GB. Concise reports and the full raw paths
are recorded in
`inner_docs/docs_drafts/results/phase2b-nvidia-20260826.md`.

The incremental Release build succeeded. Targeted Managed, adapter, RT-DETR,
and YOLOv8 runtime suites passed. The RT-DETR package's complete post-fix CTest
suite passed 9/9, including its Python lint checks.

## Provenance

| Field | Value |
| --- | --- |
| Date | 2026-08-01 |
| GPU | NVIDIA A100-SXM4-40GB |
| ROS 2 / Isaac ROS | Jazzy / 4.5 |
| ONNX Runtime | 1.23.1 |
| Build | Release |
| Input | r2bdataset2024_v1/r2b_robotarm |
| Input hash | 8eee68848ee1a95e21b1cd44d5d6ba71 |
| gpu_ros_managed | d33381358f2259b9ad16e8847b5a6dccd0357ebc |

RT-DETR was benchmarked at application commit
6dd1154d6873e4c0f0ed0fdeaab5806425d5c40d. YOLOv8 was benchmarked at
6f1b4ea51853831fc15d9ee3fa8b224805d165b3; its transport audit used
917979dc14fd03667ea019449d512c0990c69b9f. The legacy archive did not retain
an image digest or external ORT fingerprint. Future formal runs must record
the image digest and exact ONNX Runtime library identity; record the external
ORT fingerprint when external ORT mode is used.

## Throughput

| Model/config | Peak prediction | Mean output at peak |
| --- | ---: | ---: |
| RT-DETR A_fp32 | 242.031 fps | 235.309 fps |
| RT-DETR C | 102.813 fps | 99.504 fps |
| RT-DETR Managed | 102.813 fps | 99.109 fps |
| YOLOv8 C | 156.953 fps | 150.683 fps |
| YOLOv8 Managed | 149.219 fps | 147.752 fps |

The current Phase 2B archive contains no formal YOLOv8 A result, so it is not
listed as a Phase 2B measurement.

At fixed 30 Hz and 60 Hz, the RT-DETR configurations missed no frames. The
YOLOv8 C and Managed configurations also missed no frames at those rates.

## Fixed-input detection comparison

RT-DETR C versus Managed used stamp matching: 390 messages per bag, 389 paired
frames, one unpaired frame on each side, mean IoU 0.999933, mean score delta
0.0000745, and class match rate 1.0. Numeric thresholds passed; strict
no-unpaired validation remains pending.

YOLOv8 C versus Managed used 393 paired frames with one additional Managed
frame. Paired-frame IoU and class match rate were 1.0, with zero score delta
and no unmatched detections.

## Managed bridge audit

Pointer-identity and lifecycle tests passed. The byte-precision Nsight audit
reported identical CUDA kernel-name sets, zero Managed-only memcpy signatures,
and zero D2D count/byte delta.

The 2,822,400-byte decoder D2H operation occurred once per recorded YOLOv8
output frame in both lanes. The 4,915,200-byte D2D input operation occurred
396 times in both lanes. The official decoder D2H operation is not a new
Managed bridge copy.

Mean bridge callback/readiness costs were:

| Model | NITROS to Managed | Managed to NITROS |
| --- | ---: | ---: |
| RT-DETR | 0.159640 ms | 0.058857 ms |
| YOLOv8 | 0.064333 ms | 0.050348 ms |

These values include callback, readiness, and publish work; they are not copy
latencies.

## Limitations

The C/M component container sometimes exited -11 after benchmark JSON and
comparison output had already been written. Fixed-input captures also differed
by one boundary frame in some runs. These issues do not change the completed
pointer, copy-count, or paired-frame conclusions.

The archived NVIDIA results above predate the AMD Managed HIP production path
and are not an AMD Phase 2B result. AMD validation must first prove the direct
topology and strict contract, then compare direct Managed with the standard and
staged-control lanes. The production direct lane must not contain application
TensorBundle staging; staged-control adapter copies are measured separately.
MI350X model, memory-contract, high-load, and report-only comparison results
remain to be archived after the allocation-backed validation run.
