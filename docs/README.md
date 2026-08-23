# Project Documentation

This directory contains the public build, experiment, validation, and result
documentation for the AMD object-detection migration. Procedures use
parameterized paths and external artifact locations so that they can be used
on different hosts and runtime backends.

## Supported baseline

| Dependency | Baseline |
| --- | --- |
| ROS 2 | Jazzy |
| Isaac ROS compatibility | 4.5 |
| ONNX Runtime | 1.23.1 with the MIGraphX patch series described in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) |
| `gpu_ros_managed` | Required sibling checkout; select and record its revision for each run |

Set the locations used by a local runtime before following a procedure:

```bash
export GPU_ROS_MANAGED_DIR=<path-to-gpu_ros_managed>
export OVG_ASSETS_ROOT=<external-assets-root>
export OVG_RESULTS_ROOT=<external-results-root>
export OVG_STATE_ROOT=<persistent-runtime-state-root>
```

The launchers record the selected sibling revision and external ORT identity
in the run archive. Exact source revisions and license boundaries are
centralized in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md), rather
than repeated in source headers.

## Documents

Procedures:

- [`open-source-migration.md`](open-source-migration.md) — supported package
  names, namespace boundaries, and the NVIDIA compatibility edge.
- [`phase0-benchmark-reproduction.md`](phase0-benchmark-reproduction.md) — the
  NVIDIA RT-DETR and Grounding DINO reference benchmarks.
- [`phase1-experiment.md`](phase1-experiment.md) — the NVIDIA A/B/C/D comparison.
- [`phase2a-experiment.md`](phase2a-experiment.md) — AMD standard ROS 2 and
  MIGraphX build, validation, and benchmark procedure.
- [`phase2b-managed-transport.md`](phase2b-managed-transport.md) — managed
  transport contracts and comparison procedure.

Results and interpretation:

- [`phase1-results.md`](phase1-results.md) — historical NVIDIA results.
- [`phase2a-results.md`](phase2a-results.md) — AMD Phase 2A closure results.
- [`phase2b-results.md`](phase2b-results.md) — current Phase 2B implementation
  and validation status.
- [`nitros-4.5-port-map.md`](nitros-4.5-port-map.md) — the NITROS port and
  upstream attribution map.

Raw JSON, bags, traces, profiles, logs, container images, and model/data
assets are run artifacts. They are kept in an explicitly selected external
result or asset location and are not part of the source tree.

## Phase boundaries

- Phase 0 reproduces the unmodified NVIDIA RT-DETR and Grounding DINO
  benchmarks.
- Phase 1 compares TensorRT and ONNX Runtime, and NITROS and standard ROS 2,
  on NVIDIA.
- Phase 2A validates AMD RT-DETR and YOLOv8 standard ROS 2 TensorBundle paths
  with ONNX Runtime and MIGraphX. It does not reproduce the NVIDIA A/B/C/D
  matrix on AMD.
- Phase 2B develops reusable `gpu_ros_managed` device-buffer transport against
  the Phase 2A reference. The standard ROS 2 path remains independently
  available, and direct Managed HIP and staged-control graphs are documented
  as separate comparison lanes.
