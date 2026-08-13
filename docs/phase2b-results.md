# Phase 2B Validation Results

This report records the historical NVIDIA validation archive. The revised AMD
direct Managed HIP implementation is present, but no allocation-backed AMD
real-model result is claimed until its required capture, contract, topology,
and high-load gates complete. Procedures and interpretation rules are in
phase2b-managed-transport.md.

The unified NVIDIA/AMD audit tooling added afterward has not been run against
new hardware traces in this change. Its implementation status is therefore
`tooling implemented`; no new `PASS` is claimed here. The NVIDIA values below
remain historical, and the allocation-backed AMD copy audit remains pending.

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
TensorList staging; staged-control adapter copies are measured separately.
MI350X model, memory-contract, high-load, and report-only comparison results
remain to be archived after the allocation-backed validation run.
