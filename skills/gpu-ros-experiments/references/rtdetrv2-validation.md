# RT-DETRv2 R50 validation runbook

## Identity and source boundary

The formal `rtdetrv2_r50` asset is generated locally from the pinned RT-DETR
source and checkpoint. Keep the source checkout, checkpoint, generated ONNX,
COCO data, and validation output outside the repository. Do not infer a weight
or ONNX license from the source repository license; the model manifest records
`NOASSERTION` where evidence is missing.

The required identities are:

```text
RT-DETR source: b8957b30431abc938db16016f6b5e395b562c5dd
formal ONNX:    ba0c2c830edece85335aef60e30ad649e3e739c82ae90c3ec9074498a8092086
```

Verify the checkpoint SHA against `gpu_ros_object_detection/config/model-profiles.json` before export.

## Execution environments

The AMD Apptainer runtime intentionally contains the ROS/ONNX validation
packages but does not contain `uv` or the PyTorch RT-DETR export environment.
Do not try to build the formal model from that shell. Prepare a separate,
external Python environment from the locked `gpu_ros_object_detection/tools/rtdetrv2-export/uv.lock`,
then point the commands below at its interpreter. The AMD shell can import and
validate an already-generated ONNX with `python`; it must not silently use a
different model or a PyPI CPU ORT wheel for the MIGraphX gate.

## Reproducible export (external export environment)

Reuse a verified existing formal ONNX when available. Export only when
required and authorized, outside the AMD SIF, with an interpreter containing
the locked dependencies:

```bash
export OVG_ASSETS_ROOT=/absolute/path/to/external-assets
export OVG_CACHE_ROOT=/absolute/path/to/external-cache
export OVG_RTDETR_EXPORT_PYTHON=/absolute/path/to/export-venv/bin/python
test -x "${OVG_RTDETR_EXPORT_PYTHON}"
"${OVG_RTDETR_EXPORT_PYTHON}" \
  gpu_ros_object_detection/tools/phase2-assets build-rtdetrv2-r50 \
  --upstream-dir /absolute/path/to/RT-DETR \
  --checkpoint /absolute/path/to/rtdetrv2_r50vd_6x_coco_ema.pth
```

`OVG_RTDETR_EXPORT_PYTHON` may be `python3` only when that interpreter already
has every dependency in the lock. The helper uses this interpreter directly;
it falls back to `uv` only when `uv` is installed. No download is performed by
the helper.

The helper runs the official export twice in isolated temporary paths,
validates the ONNX contract, and refuses to install a result unless both
digests match the locked formal SHA. If reproducibility fails, stop and repair
the environment or export metadata; do not publish one successful digest.

## PyTorch versus CPU ORT

Use the same external export interpreter and an external cache/venv:

```bash
TORCH_HOME=/absolute/path/to/torch-cache \
"${OVG_RTDETR_EXPORT_PYTHON}" \
  gpu_ros_object_detection/tools/rtdetrv2-export/validate_export.py \
  --upstream-dir /absolute/path/to/RT-DETR \
  --checkpoint /absolute/path/to/rtdetrv2_r50vd_6x_coco_ema.pth \
  --onnx /absolute/path/to/external-assets/models/rtdetrv2_r50/rtdetrv2_r50.onnx
```

All 300 labels must match. Scores use `rtol=1e-4, atol=1e-5`; boxes use
`rtol=1e-4, atol=1e-3`.

## CPU ORT versus MIGraphX

Run in the provider runtime with a pre-mounted Python interpreter whose
`onnxruntime` binding was built from the exact project ORT/MIGraphX build, not
the PyPI CPU wheel. The SIF does not provide `uv`:

```bash
export OVG_RTDETR_AMD_PYTHON=/absolute/path/to/ort-migraphx-venv/bin/python
"${OVG_RTDETR_AMD_PYTHON}" -c \
  'import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())'
"${OVG_RTDETR_AMD_PYTHON}" \
  gpu_ros_object_detection/tools/rtdetrv2-export/validate_provider_parity.py \
  --onnx /absolute/path/to/external-assets/models/rtdetrv2_r50/rtdetrv2_r50.onnx \
  --images-npy /absolute/path/to/fixed-images.npy \
  --orig-target-sizes-npy /absolute/path/to/fixed-orig-target-sizes.npy
```

Labels must match; maximum score absolute error must be at most `1e-3`; maximum
box absolute error must be at most `0.1 px`.

If the provider check does not list `MIGraphXExecutionProvider`, stop and fix
the Python binding/environment. Do not replace it with a CPU-only wheel and
continue.

## Same-bag end-to-end comparison

Capture CPU/reference and MIGraphX candidate outputs from the same input bag,
then run the comparator with class-aware one-to-one matching:

```bash
ros2 run gpu_ros_detection_validation compare_detection2d_bags.py \
  --reference-bag /absolute/path/to/reference-bag \
  --candidate-bag /absolute/path/to/migraphx-bag \
  --match-policy stamp --min-score 0.0 --max-detections-per-frame 0 \
  --class-aware-matching --min-paired-frames 20 --min-class-match-rate 1.0 \
  --min-mean-iou 0.99 \
  --min-pair-iou 0.99 \
  --max-mean-score-delta 0.001 \
  --max-pair-score-delta 0.001 \
  --min-frame-pass-rate 1.0 \
  --output-json /absolute/path/to/results/rtdetrv2-cpu-vs-migraphx.json
```

No detection may remain unmatched. The full gate and identity requirements
are defined in [result acceptance](result-acceptance.md).

## COCO val2017

Run the official evaluator with all 300 candidates:

```bash
"${OVG_RTDETR_AMD_PYTHON}" \
  gpu_ros_object_detection/tools/rtdetrv2-export/evaluate_coco.py \
  --onnx /absolute/path/to/external-assets/models/rtdetrv2_r50/rtdetrv2_r50.onnx \
  --val2017 /absolute/path/to/coco/val2017 \
  --annotations /absolute/path/to/coco/annotations/instances_val2017.json \
  --output-json /absolute/path/to/results/rtdetrv2-r50-coco-val2017.json
```

This command can consume more than 1 GB of working space. Announce that
resource requirement before starting it. AP and AP50 must each be within 0.5
percentage point of the upstream-reported 53.4 and 71.6. A failed validation is
a bug in the formal default; do not silently fall back to another profile.
