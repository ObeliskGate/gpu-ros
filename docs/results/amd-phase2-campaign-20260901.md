# AMD Phase 2 campaign — result record

This record describes the current campaign without host, user, scheduler,
device-instance, or deployment-path identifiers. It intentionally retains the
hardware and software provenance needed to reproduce an equivalent run. Raw
artifacts remain outside the repository under the configured results root.
Values below are observations from the run and are not portable performance
claims.

## Runtime and software provenance

| Field | Recorded value |
| --- | --- |
| Runtime adapter | Apptainer (validated SIF; immutable digest `not-captured`) |
| Hardware model | AMD Instinct MI350X |
| GPU target | `gfx950` |
| ROS 2 | Jazzy |
| ROCm/HIP | 7.1.1 |
| MIGraphX | 2.14 behavior from the ROCm 7.1.1 package set; Debian package revision `not-captured` |
| MIGraphX source | ROCm package repository configured by the base image; exact package revision `not-captured` |
| ONNX Runtime | 1.23.1 |
| ORT source commit | `d9b2048791efb5804fe3d53a04b4971256addebf` |
| ORT external build fingerprint | `d9b2048791ef-66fb994b0374-gfx950` |
| ORT provider patches | GridSample capability enablement; int64-div CPU fallback (`v1`) |
| Benchmark framework | `ros2_benchmark` `v4.5-0` plus the tracked standalone patch |
| Container base image | `rocm/dev-ubuntu-24.04:7.1.1-complete` |
| Container recipe | `docker/phase2a-amd.Dockerfile`, `docker-compose.phase2a-amd.yaml`, `apptainer/phase2-amd.def` |
| Container recipe revision | `1a02236ce09b245ed6d6d4f26128f11f54470c95` (same launch tip; verify against the external matrix manifest) |
| Image digest | `not-captured`; use the recipe/base identity above until a SIF or OCI digest is archived |
| Application revision (RT-DETR matrix) | `1a02236ce09b245ed6d6d4f26128f11f54470c95` (verify against the external matrix manifest) |
| Application revision (YOLOv8 records) | `698cc86ab179a04d274b7aeb80c44580052c3122` (verify each raw report) |
| `gpu_ros_managed` revision | `caaf6cbb599bfb194671bc957f096443b94bef01` |
| Driver version | `not-captured` |
| RT-DETR model | `rtdetrv2_r50`; SHA-256 `ba0c2c830edece85335aef60e30ad649e3e739c82ae90c3ec9074498a8092086` |
| RT-DETR source | commit `b8957b30431abc938db16016f6b5e395b562c5dd` |
| YOLOv8 model | `yolov8/yolov8s.onnx`; SHA-256 `d6e22418dd1acc69a232a1b297c01dfc785842fd11a4a84546c84e14cdeb235c` |
| R2B dataset | `r2bdataset2024_v1/r2b_robotarm`; hash `8eee68848ee1a95e21b1cd44d5d6ba71` |
| Matrix archive IDs | YOLOv8 `yolov8_phase2b_matrix_amd-mi350x-20260901-072305-continue`; RT-DETR `rtdetr_phase2b_matrix_amd-mi350x-20260901-112808` |
| Audit archive IDs | RT-DETR transport `rtdetr_amd_transport_amd-mi350x-20260901-111526`; HIP copy probe `rocprof-hip-copy-probe-amd-mi350x-20260901-121109` |

## Completed checks

| Check | Result | Interpretation |
| --- | --- | --- |
| RT-DETRv2 R50 export reproducibility | PASS | The pinned PyTorch checkpoint and two independent ONNX exports matched the formal digest. |
| RT-DETRv2 R50 PyTorch ↔ CPU ONNX validation | PASS | 300 candidates; labels equal; boxes max absolute error 0.00213623046875; scores max absolute error 3.935769200325012e-06; both passed the script tolerances. |
| RT-DETRv2 R50 CPU ↔ direct Managed HIP output comparison | REPORT_ONLY | 395 reference frames, 394 candidate frames, 394 paired; one unpaired reference frame; mean/min IoU 1.0000; maximum score delta 0.0001; class match 1.0000; zero unmatched detections; paired-frame rate 100.0%, overall rate 99.7%. |
| HIP copy probe | PASS | Non-empty ROCprofiler JSON, agent/API/kernel CSV files, and probe manifest were produced. |
| RT-DETR AMD three-lane matrix execution | PASS | All three rounds completed for standard, staged-control, and direct Managed lanes. The shell `PASS` means all nine JSON reports were emitted; it is not a numeric reproduction claim. |
| RT-DETR transport audit | INCONCLUSIVE | The capture completed, but the copy parser retained unresolved/ambiguous trace evidence. This is not a detection-output failure. |

`REPORT_ONLY` is the comparator's fixed archival mode. It reports the numeric
relationship and deliberately does not manufacture an aggregate PASS/FAIL.

## RT-DETR matrix numerical record

The checked-in campaign record currently contains the matrix runner's execution
status but not the nine per-round RT-DETR JSON summaries. Until those JSONs are
copied to the public result archive, the matrix has the deliberately split
status **execution PASS / numerical baseline pending**:

| Formal lane | Peak prediction (Hz) | Mean output at peak (fps) | Peak misses / sent | Fixed 10 / 30 / 60 Hz output (fps) | Fixed 10 / 30 / 60 misses / sent |
| --- | ---: | ---: | ---: | --- | --- |
| Standard ROS 2 | pending archive | pending archive | pending archive | pending archive | pending archive |
| Staged control | pending archive | pending archive | pending archive | pending archive | pending archive |
| Direct Managed HIP | pending archive | pending archive | pending archive | pending archive | pending archive |

Do not fill this table from the older Synthetica/RT-DETR archive: its model and
software provenance do not match the formal `rtdetrv2_r50` campaign. The raw
matrix archive must retain each round's values, the arithmetic-mean aggregation,
the fixed-rate rows, and the manifest before this campaign can be used as a
performance reproduction baseline. The acceptance semantics are defined in
[`results/README.md`](README.md).

## YOLOv8 observations from the same campaign

The following values are retained as run observations. Peak-search misses are
expected saturation behavior; the fixed-rate rows are the reliability check.

| Lane | Predicted peak (Hz) | Mean output at peak (fps) | Peak misses / sent | Fixed 10 / 30 / 60 Hz output (fps) | Fixed 10 / 30 / 60 misses / sent |
| --- | ---: | ---: | ---: | --- | --- |
| Standard ROS 2 | 320.992 | 311.971 | 37.667 / 1604 | 10.204 / 30.202 / 60.209 | 0 / 50 · 0 / 150 · 0 / 300 |
| Staged control | 500.500 | 478.428 | 103.333 / 2502 | 10.204 / 30.201 / 60.200 | 0 / 50 · 0 / 150 · 0 / 300 |
| Direct Managed HIP | 562.9375 | 549.841 | 58 / 2814 | 10.205 / 30.203 / 60.207 | 0 / 50 · 0 / 150 · 0 / 300 |

The YOLOv8 three-round matrix also completed successfully. The values above are
the graph reports retained from this campaign; the per-round matrix JSONs remain
the authority for an aggregated baseline. The direct and staged lanes are
separate topology measurements; the table must not be read as an isolated
transport-cost measurement or a correctness result.

## Scope decisions

- The direct Managed high-load script was removed from the formal experiment
  set. A looping publisher can produce a different number of output callbacks
  than input messages, so that counter is not a meaningful transport or
  throughput gate. No high-load result is promoted here.
- COCO val2017 evaluation was intentionally deferred.
- CPU↔MIGraphX Python provider parity was not run because the available export
  environment does not contain a Python binding exposing
  `MIGraphXExecutionProvider`. A CPU-only wheel must not be substituted.
- Host and scheduler identity are not part of this record. The driver package
  revision was not captured in the current run and is marked explicitly above;
  future formal runs must record it alongside the public ROCm/MIGraphX identity.

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

The model export, numeric comparison, profiler probe, and matrix executions are
ready for review. The RT-DETR transport audit remains an explicitly labelled
`INCONCLUSIVE` observation. The RT-DETR performance baseline is not promoted
until its per-round JSON values are added to this record. The deferred
COCO/provider-parity checks must remain visible as deferred rather than silently
treated as passed.
