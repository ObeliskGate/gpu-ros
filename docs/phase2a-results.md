# Phase 2A Completion Report

## Status

AMD Phase 2A is complete for both RT-DETR and YOLOv8. The accepted paths use
standard ROS 2 TensorBundle messages, ONNX Runtime, and the MIGraphX execution
provider:

```text
RT-DETR: Image -> RT-DETR image encoder/preprocessor -> ORT MIGraphX
          -> RT-DETR decoder -> Detection2DArray

YOLOv8:  Image -> YOLOv8 image encoder -> ORT MIGraphX
          -> YOLOv8 decoder -> Detection2DArray
```

Both paths passed the AMD build, package and proof-of-life tests, fixed-input
capture, Detection2DArray comparison, provider audit, external ORT library
audit, and benchmark runs. These standard ROS 2 paths remain independently
available and are the reference for Phase 2B managed-transport work.

The recorded RT-DETR numbers below belong to the pre-migration closure asset.
The AMD default is now the pinned Apache-2.0 RT-DETRv2 R50 export described in
[`phase2a-experiment.md`](phase2a-experiment.md). Its correctness, provider
placement, and benchmark closure must be rerun before those numbers are used
as RT-DETRv2 results; the historical values and their evidence are retained
unchanged. The YOLOv8 closure record remains the same model and contract.

## Provenance

The MI350X results are the primary AMD Phase 2A record. The MI300X RT-DETR
benchmark is included as supplementary data and is not used as the primary
performance result.

| Field | Phase 2A closure value |
| --- | --- |
| Primary GPU | AMD Instinct MI350X, `gfx950` |
| ROS 2 | Jazzy |
| Isaac ROS | 4.5 |
| ONNX Runtime | 1.23.1, external MIGraphX build |
| Build | Release |
| RT-DETR MI350X application revision | `9fd3f79cfec8e2e530009d4a0a25d5742e877b2f` |
| YOLOv8 MI350X application revision | `635e285f2546535bd02cd3eda0ab47cc1b3aaa89` |
| `gpu_ros_managed` revision | `d33381358f2259b9ad16e8847b5a6dccd0357ebc` |
| External ORT | 1.23.1; source and patch revisions are recorded in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) |
| Input dataset | `r2bdataset2024_v1/r2b_robotarm` |
| Input dataset hash | `8eee68848ee1a95e21b1cd44d5d6ba71` |
| Image input | 1280x720 |
| Network input | 640x640 |

The RT-DETR MI350X benchmark was recorded on 2026-08-04. The YOLOv8 MI350X
benchmark was recorded on 2026-08-06. Raw benchmark records, comparisons,
profiles, bags, logs, and container metadata are external run artifacts; this
report keeps the summarized evidence and does not encode their site-specific
paths or filenames.

The canonical YOLOv8 asset is a user-provided YOLOv8s ONNX model with the
following recorded identity: Ultralytics 8.4.67, COCO 80 classes, opset 17,
static input `[1,3,640,640]` named `images`, and output `output0` without
built-in NMS. Its compatibility SHA-256 is:

```text
d6e22418dd1acc69a232a1b297c01dfc785842fd11a4a84546c84e14cdeb235c
```

The repository does not provide, download, or export this model. Users are
responsible for obtaining it and complying with its applicable license.

## RT-DETR results

### Throughput

The MI350X run is the formal result:

| Platform | Peak prediction | Mean output at peak | 30 Hz missed | 60 Hz missed |
| --- | ---: | ---: | ---: | ---: |
| MI350X, `gfx950` | 79.047 FPS | 75.312 FPS | 0 | 0 |

For context, the supplementary MI300X run measured 86.852 FPS peak and
80.653 FPS mean output. It also missed no frames at 30 Hz or 60 Hz. The two
measurements are from different AMD platforms and should not be treated as a
single hardware trend.

### Numeric validation

The final comparison used the NVIDIA Config C reference bag and the MI350X
fixed-input candidate bag. The recordings were made at different wall-clock
times but preserved matching source header stamps, so stamp matching was used.
The seven unpaired candidate frames are a capture-boundary discrepancy; they
remain in the overall frame denominator but do not affect the 390 paired
frames. The comparator passes when there are at least 20 paired frames, mean
IoU is at least 0.80, mean score delta is at most 0.20, overall frame pass rate
is at least 0.70, and mean class match rate is 1.0. This comparison met those
criteria:

| Metric | Result |
| --- | ---: |
| Reference frames | 390 |
| Candidate frames | 397 |
| Paired frames | 390 |
| Unpaired candidate frames | 7 |
| Mean IoU | 0.9983 |
| Minimum IoU | 0.9896 |
| Mean score delta | 0.0025 |
| Class match rate | 1.0000 |
| Unmatched detections | 20 |
| Paired-frame pass rate | 94.9% |
| Overall frame pass rate | 93.2% |
| Comparison status | PASS |

An earlier index-matching attempt was a diagnostic result, not the closure
result. The seven extra candidate frames shifted index pairing and caused an
artificial failure. The stamp-matched result is the authoritative comparison.

### Provider placement

The final MI350X RT-DETR profile recorded MIGraphX kernel events for four
unique nodes and CPU events for five postprocessor nodes:

```text
/postprocessor/Div
/postprocessor/GatherElements
/postprocessor/Mod
/postprocessor/Tile_1
/postprocessor/Unsqueeze_5
```

The saved profile records a known and controlled performance limitation. The
main inference path ran on MIGraphX, and the CPU placement did not cause a
correctness failure.
Unexpected additional CPU placement remains an audit failure. Any future
fallback reduction must preserve the model's integer-division semantics.

## YOLOv8 results

### Throughput

The MI350X YOLOv8 run measured 242.945 FPS peak and 232.349 FPS mean output at
peak. The NVIDIA A100 Config D reference measured 141.48 FPS peak. The AMD
peak is 71.7% higher than that reference:

| Platform and configuration | Peak prediction |
| --- | ---: |
| NVIDIA A100 Config D, ORT CUDA EP plus standard ROS 2 | 141.48 FPS |
| AMD MI350X, ORT MIGraphX plus standard ROS 2 | 242.945 FPS |

Both the 30 Hz and 60 Hz fixed-rate trials missed no frames. The recorded
first and last endpoint latencies were approximately 6.6 to 7.8 ms. These are
complete end-to-end configurations on different GPUs and with different ORT
providers. The percentage is a system result, not a single-variable CUDA to
MIGraphX comparison.

### Numeric validation and provider placement

The final YOLOv8 comparison paired all 397 reference and candidate frames:

| Metric | Result |
| --- | ---: |
| Paired frames | 397 / 397 |
| Mean IoU | 1.0000 |
| Minimum IoU | 1.0000 |
| Mean score delta | 0.0000 |
| Class match rate | 1.0000 |
| Unmatched detections | 0 |
| Frame pass rate | 100% |
| Comparison status | PASS |

The MI350X profile showed MIGraphX execution with no CPU fallback. The standard
YOLOv8 image encoder preserved the input header and produced the canonical
`images` tensor contract used by the ONNX model.

## Validation coverage

The closure covered the following independent checks for the standard ROS 2
AMD paths:

- Release selective build with NITROS, CUDA, and ROCm application transport
  options disabled, and MIGraphX enabled.
- Package unit tests and graph proof-of-life tests.
- MIGraphX warm-up followed by fixed-input capture at the required image and
  network resolutions.
- Detection2DArray comparison using the existing comparator and stamp matching
  where timestamps were preserved.
- ORT provider profiling, including MIGraphX kernel events and the documented
  RT-DETR CPU placement.
- Runtime library inspection confirming the selected external ORT install.
- RT-DETR and YOLOv8 benchmark sweeps with 30 Hz and 60 Hz fixed-rate trials.

## Remaining optimization

The only remaining Phase 2A follow-on is performance work to reduce the known
RT-DETR postprocessor CPU fallback. It is not an acceptance task. Changes must
retain the current numeric results and preserve integer-division semantics.
