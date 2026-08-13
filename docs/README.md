# Project Documentation

The docs directory contains version-controlled project documentation that can
be reused across developers and runtime environments.

It documents architecture, supported software baselines, build and run
contracts, experiment procedures, acceptance criteria, and reproducible
results. Usernames, hostnames, SSH settings, site-specific mount points,
Slurm allocation details, temporary image/archive names, one-off file transfers,
and unstructured work logs belong in inner_docs/ instead.

## Supported baseline

| Dependency | Baseline |
| --- | --- |
| ROS 2 | Jazzy |
| Isaac ROS | 4.5 |
| ONNX Runtime | 1.23.1 |
| gpu_ros_managed | Revision in dependencies/gpu_ros_managed.repos for clean imports |

gpu_ros_managed is a required sibling repository. Import it from the
workspace src directory:

~~~bash
vcs import < amd_ros_object_detection/dependencies/gpu_ros_managed.repos
~~~

An existing checkout may be selected with GPU_ROS_MANAGED_DIR. The NVIDIA
helper reports its commit without blocking standard Phase 1 Config D runs,
which do not use managed transport. For a reproducible managed-transport run,
set GPU_ROS_MANAGED_EXPECTED_COMMIT to the revision being audited.

## Document types

The experiment documents are general procedures and do not contain
machine-specific performance numbers:

- phase0-benchmark-reproduction.md
- phase1-experiment.md
- phase2a-experiment.md
- phase2b-managed-transport.md

Result reports contain hardware, software versions, dates, input identity,
measurements, and interpretation limits:

- phase1-results.md
- phase2b-results.md
- [`phase2a-results.md`](phase2a-results.md), the formal Phase 2A closure report

Container-internal paths such as /workspaces/ovg-assets are project runtime
contracts. They are different from private host mount paths and may appear in
the experiment documents.

## Phase boundaries

- Phase 0 reproduces the unmodified NVIDIA RT-DETR and Grounding DINO
  benchmarks.
- Phase 1 compares TensorRT and ONNX Runtime, and NITROS and standard ROS 2,
  on NVIDIA.
- Phase 2A is complete for the AMD RT-DETR and YOLOv8 standard ROS 2
  TensorList + ONNX Runtime MIGraphX paths. It does not reproduce the Phase 1
  A/B/C/D matrix on AMD.
- Phase 2A is the accepted standard ROS 2 reference. Phase 2B develops the
  reusable gpu_ros_managed device-buffer transport against that reference and
  must not remove its independent availability. The revised AMD Phase 2B
  production lane is direct Managed HIP with a strict ORT I/O contract; the
  adapter-based staged-control lane remains an explicit comparison graph.
