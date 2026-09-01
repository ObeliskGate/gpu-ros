# AMD Phase 2 campaign — result record

This record describes the current campaign without host, user, scheduler,
device-instance, driver, kernel, container, or deployment-path identifiers.
Raw artifacts remain outside the repository under the configured results root.
Values below are observations from the run and are not portable performance
claims.

## Completed checks

| Result field | Value |
| --- | --- |
| Hardware model | AMD Instinct MI350X |


| Check | Result | Interpretation |
| --- | --- | --- |
| RT-DETRv2 R50 export reproducibility | PASS | The pinned PyTorch checkpoint and two independent ONNX exports matched the formal digest. |
| PyTorch ↔ CPU ONNX validation | PASS | 300 candidates; labels equal; boxes and scores passed the script tolerances. |
| CPU ↔ direct Managed HIP output comparison | REPORT_ONLY | 395 reference frames, 394 candidate frames, 394 paired; one unpaired reference frame; mean/min IoU 1.0000; maximum score delta 0.0001; class match 1.0000; zero unmatched detections; paired-frame rate 100.0%, overall rate 99.7%. |
| HIP copy probe | PASS | Non-empty ROCprofiler JSON, agent/API/kernel CSV files, and probe manifest were produced. |
| RT-DETR AMD three-lane matrix | PASS | All three rounds completed for standard, staged-control, and direct Managed lanes. Fixed-rate rows are included in the matrix reports. |
| RT-DETR transport audit | INCONCLUSIVE | The capture completed, but the copy parser retained unresolved/ambiguous trace evidence. This is not a detection-output failure. |

`REPORT_ONLY` is the comparator's fixed archival mode. It reports the numeric
relationship and deliberately does not manufacture an aggregate PASS/FAIL.

## YOLOv8 observations from the same campaign

The following values are retained as run observations. Peak-search misses are
expected saturation behavior; the fixed-rate rows are the reliability check.

| Lane | Predicted peak (Hz) | Mean output at peak (fps) | Peak misses / sent | Fixed 10 / 30 / 60 Hz misses |
| --- | ---: | ---: | ---: | --- |
| Standard ROS 2 | 320.992 | 311.971 | 37.667 / 1604 | 0 / 0 / 0 |
| Staged control | 500.500 | 478.428 | 103.333 / 2502 | 0 / 0 / 0 |
| Direct Managed HIP | 562.9375 | 549.841 | 58 / 2814 | 0 / 0 / 0 |

The YOLOv8 three-round matrix also completed successfully. The direct and
staged lanes are separate topology measurements; the table must not be read as
an isolated transport-cost measurement or a correctness result.

## Scope decisions

- The direct Managed high-load script was removed from the formal experiment
  set. A looping publisher can produce a different number of output callbacks
  than input messages, so that counter is not a meaningful transport or
  throughput gate. No high-load result is promoted here.
- COCO val2017 evaluation was intentionally deferred.
- CPU↔MIGraphX Python provider parity was not run because the available export
  environment does not contain a Python binding exposing
  `MIGraphXExecutionProvider`. A CPU-only wheel must not be substituted.
- No host identity is part of this record. Scheduler, driver, and deployment
  metadata, when needed for reproducibility, belongs in an access-controlled
  external archive.

## External artifact layout

Use logical paths only in documentation:

```text
${OVG_RESULTS_ROOT}/phase2b-benchmark-matrix/<model>_<run-id>/
${OVG_RESULTS_ROOT}/phase2b_amd_audits/<model>_<run-id>/
${OVG_RESULTS_ROOT}/phase2a-bags/<capture-name>/
${OVG_RESULTS_ROOT}/rocprof-hip-copy-probe-<run-id>/
```

Each archive should retain its manifests, command lines, model/dataset hashes,
repository revisions, raw reports, and logs. Do not copy those paths or any
site-specific identifiers into a public result document.

## Promotion status

The model export, numerical report, profiler probe, and performance matrices
are ready for review. The RT-DETR transport audit remains an explicitly
labelled `INCONCLUSIVE` observation, and the deferred COCO/provider-parity
checks must remain visible as deferred rather than silently treated as passed.
