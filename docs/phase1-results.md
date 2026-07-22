# Phase 1 Completion Report

Phase 1 is complete at the implementation and detection-output validation
levels. RT-DETR and YOLOv8 run end to end on the NVIDIA A100 with TensorRT or
ONNX Runtime and with the NITROS or standard ROS 2 pipeline variants.

The final performance runs used an A100-SXM4-40GB, a Release build, the same
1280x720 input dataset, a 640x640 network resolution, and dataset hash
`8eee68848ee1a95e21b1cd44d5d6ba71`. The machine-readable values are in
[`phase1_final_summary_20260722.json`](../migrated_packages/benchmark_results/phase1_final_summary_20260722.json).

## RT-DETR Results

RT-DETR is precision-aligned: A/B use a TensorRT FP32 engine generated from the
same FP32 ONNX model used by C/D.

| Config | Backend and transport | Peak prediction | Mean output at peak | 30 Hz first / last latency |
|---|---|---:|---:|---:|
| A_fp32 | TensorRT FP32 + NITROS | 195.63 fps | 188.07 fps | 15.38 / 11.54 ms |
| B_fp32 | TensorRT FP32 + std ROS 2 bridge | 187.89 fps | 180.05 fps | 21.94 / 18.47 ms |
| C | ORT FP32 CUDA EP + NITROS | 100.08 fps | 95.61 fps | 19.03 / 16.60 ms |
| D | ORT FP32 CUDA EP + std ROS 2 | 87.34 fps | 83.86 fps | 29.64 / 27.73 ms |

At peak, changing A to B reduced predicted throughput by 3.95%; changing C to
D reduced it by 12.72%. Reciprocal throughput is more useful than subtracting
FPS: the implied period increased by about 0.21 ms for A to B and 1.46 ms for C
to D. At fixed 60 Hz, the last-frame transport penalties were much closer:
4.29 ms for A to B and 4.77 ms for C to D.

This is not contradictory. B performs its host-to-device copy in a separate
bridge component, so the multithreaded pipeline can overlap that stage with
work on other frames. D asks ORT to consume host input and produce host output;
those transfers and synchronization are part of the inference callback's
critical path. Pipeline throughput is controlled by the slowest stage rather
than the sum of every component's wall time.

## YOLOv8 Results

YOLOv8 is not precision-aligned across backends. A/B use TensorRT FP16 while
C/D use ORT FP32. A/C also use the upstream NITROS decoder, while B/D use the
migrated standard decoder. Backend or transport causality must not be inferred
from the four peak-FPS values alone.

| Config | Backend and transport | Peak prediction | Mean output at peak | 30 Hz first / last latency |
|---|---|---:|---:|---:|
| A | TensorRT FP16 + NITROS | 141.48 fps | 133.60 fps | 21.31 / 15.83 ms |
| B | TensorRT FP16 + std ROS 2 bridge | 249.77 fps | 242.68 fps | 20.44 / 12.04 ms |
| C | ORT FP32 CUDA EP + NITROS | 126.02 fps | 122.89 fps | 19.64 / 19.48 ms |
| D | ORT FP32 CUDA EP + std ROS 2 | 141.48 fps | 137.88 fps | 19.59 / 13.29 ms |

B and D can outperform their NITROS counterparts because the migrated decoder
avoids the upstream decoder's per-box temporary allocations and repeated class
scans. TensorRT inference is fast enough for decoder work to become the
bottleneck, which makes the B-over-A difference much larger than D-over-C.
These results compare complete deployable configurations, not a decoder-neutral
transport microbenchmark.

## Detection-Output Validation

Offline `Detection2DArray` comparison passed for both pipelines using
index-paired messages and ignoring unpaired tail frames produced by different
benchmark rates.

For RT-DETR, A versus B matched exactly on paired frames. A versus C and A
versus D both produced mean IoU 0.9946 with a mean score delta of approximately
0.0004.

For YOLOv8, A versus B matched exactly. A versus C and A versus D both produced
mean IoU 1.0, class match rate 1.0, and mean score delta 0.00195. The comparison
used a 0.3 score filter. This validates the paired outputs; it does not imply
that FP16 and FP32 are bit-identical.

## Provider Placement Audit

The verbose CUDA EP audit assigned 1,055 RT-DETR nodes to CUDA and 493 nodes to
CPU. The CPU nodes were ORT CPU-preferred optimization, shape, and bookkeeping
placements rather than evidence of whole-model fallback. Node count is not a
measure of compute share.

On AMD, MIGraphX claimed one fused RT-DETR subgraph and no CPU fallback was
observed. This difference reflects provider partitioning policy. It is not, by
itself, an explanation for the CUDA-versus-MIGraphX performance difference.

Provider auditing remains separate from formal benchmarking. ORT profiling and
bridge timing are opt-in and disabled in the final benchmark graphs.

## Implementation Completed

- Added a transport-neutral ONNX Runtime inference core with CUDA and MIGraphX
  provider selection and clear failure for unavailable requested providers.
- Added standard ROS 2 and NITROS TensorList adapters around the shared core.
- Kept configuration C inputs and outputs device-resident through CUDA I/O
  Binding, borrowed NITROS device pointers, and ORT output lifetime callbacks.
- Removed avoidable standard-path copies by borrowing input message storage,
  moving output buffers into ROS messages, and moving the RT-DETR preprocessor
  input tensor.
- Added the standard RT-DETR image encoder, preprocessor, decoder, YOLOv8
  decoder, proof-of-life tests, CUDA ownership tests, benchmark graphs, and
  offline detection-bag comparison tooling.
- Enforced Release builds and pinned the NVIDIA benchmark asset root.
- Kept provider profiling and bridge timing out of formal benchmark graphs.

## Interpretation Limits And Final Artifact Note

The reported fixed-rate latency fields are the benchmark framework's first and
last endpoint latencies, not a per-frame latency mean or percentile. First-frame
values can include startup variation, so both endpoints and missed-frame counts
should be retained in reports.

The B results above were collected before bridge timing was disabled in the
formal graphs. The instrumentation only records a timestamp and periodic
summary, but B and B_fp32 should be rerun once after rebuilding to freeze the
clean raw artifacts. This does not require rerunning A, C, or D.

Older July 2 JSON files remain in `migrated_packages/benchmark_results` for
history. They predate device-resident I/O Binding and Release enforcement and
must not be mixed into the final matrix.
