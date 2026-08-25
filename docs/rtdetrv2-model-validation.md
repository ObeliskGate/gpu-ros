# RT-DETRv2 R50 asset and validation contract

`rtdetrv2_r50` is the formal default for MIGraphX, ROCm, and CPU. It is not an
experimental fallback. CUDA `auto` continues to select `nvidia_synthetica` so
historical NVIDIA A/B/C/D graphs retain their original model identity.

All checkouts, checkpoints, ONNX files, datasets, and validation output stay
outside the source repository. `config/model-profiles.json` records the
distributor, architecture upstream, weight source, export recipe, and separate
source/weight/ONNX license evidence. `NOASSERTION` assets must not be included
in a source release.

## Reproducible local export

Acquire the official checkpoint outside the repository and verify the SHA-256
recorded in `config/model-profiles.json`. Check out the exact architecture
commit, then run:

```bash
export OVG_ASSETS_ROOT=<external-assets-root>
export OVG_CACHE_ROOT=<external-cache-root>

tools/phase2-assets build-rtdetrv2-r50 \
  --upstream-dir <external-RT-DETR-checkout> \
  --checkpoint <external-rtdetrv2-r50-checkpoint>
```

The command exports twice in separate paths, validates the ONNX contract, and
fails unless both byte digests equal the formal digest in the source recipe.
The offline config override disables only the redundant pretrained-backbone
download; the complete, verified EMA checkpoint is loaded afterward.

## PyTorch and CPU ORT

```bash
UV_PROJECT_ENVIRONMENT=<external-export-venv> \
TORCH_HOME=<external-torch-cache> \
UV_OFFLINE=1 \
uv run --frozen \
  --project tools/rtdetrv2-export \
  tools/rtdetrv2-export/validate_export.py \
  --upstream-dir <external-RT-DETR-checkout> \
  --checkpoint <external-rtdetrv2-r50-checkpoint> \
  --onnx <external-assets-root>/models/rtdetrv2_r50/rtdetrv2_r50.onnx
```

All 300 labels must match. Scores use `rtol=1e-4, atol=1e-5`; boxes use
`rtol=1e-4, atol=1e-3`.

## CPU ORT and MIGraphX

Run this on the AMD benchmark node in a uv-managed environment containing the
Python binding built from the exact project ORT commit. The PyPI CPU wheel in
the export environment cannot expose MIGraphX and must not be used for this
step:

```bash
uv run --active --no-sync python \
  tools/rtdetrv2-export/validate_provider_parity.py \
  --onnx <external-assets-root>/models/rtdetrv2_r50/rtdetrv2_r50.onnx \
  --images-npy <fixed-images.npy> \
  --orig-target-sizes-npy <fixed-orig-target-sizes.npy>
```

Labels must match, score maximum absolute error must be at most `1e-3`, and
box maximum absolute error must be at most `0.1 px`.

## Same-bag end-to-end comparison

Capture CPU/reference and MIGraphX candidate output from the same input bag,
then use strict class-aware per-pair thresholds:

```bash
ros2 run gpu_ros_detection_validation compare_detection2d_bags.py \
  --reference-bag <reference-bag> \
  --candidate-bag <candidate-bag> \
  --match-policy stamp \
  --min-score <formal-detection-threshold> \
  --class-aware-matching \
  --min-mean-iou 0.99 \
  --min-pair-iou 0.99 \
  --max-mean-score-delta 0.001 \
  --max-pair-score-delta 0.001 \
  --min-frame-pass-rate 1.0 \
  --output-json <external-results>/rtdetrv2-cpu-vs-migraphx.json
```

No detection at or above the threshold may remain unmatched.

## COCO val2017

This command reads the full external val2017 dataset and retains all 300 model
candidates for the official COCO evaluator. Its dataset and working-set size
can exceed 1 GB.

```bash
uv run --frozen \
  --project tools/rtdetrv2-export \
  tools/rtdetrv2-export/evaluate_coco.py \
  --onnx <external-assets-root>/models/rtdetrv2_r50/rtdetrv2_r50.onnx \
  --val2017 <external-coco-root>/val2017 \
  --annotations <external-coco-root>/annotations/instances_val2017.json \
  --output-json <external-results>/rtdetrv2-r50-coco-val2017.json
```

AP and AP50 must each be within 0.5 percentage point of the upstream-reported
53.4 and 71.6. MIGraphX parity, same-bag closure, and full COCO evaluation are
required before publishing new accuracy or performance conclusions; failures
are bugs in the formal default and do not trigger profile fallback.
