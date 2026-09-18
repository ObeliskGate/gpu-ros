# AGENTS.md

## Project overview

GPU ROS is one ROS 2 monorepo with two library collections:
`gpu_ros_managed/` and `gpu_ros_object_detection/`. The compatibility targets
are ROS 2 Jazzy, Ubuntu 24.04, ROCm 7.1.1 and the selected AMD device target,
with the external ORT/MIGraphX build locked in
`gpu_ros_object_detection/config/onnxruntime.lock`; NVIDIA reference work uses
the pinned Isaac ROS 4.5-compatible runtime. Other combinations are not implied
by these targets.

Read the checked-out [experiment skill](skills/gpu-ros-experiments/SKILL.md)
and its references before runtime work. These files are the method authority;
installing a skill is optional.

## Collection ownership

- `gpu_ros_managed/` contains seven packages: core, CUDA, HIP, ROS wrappers,
  managed TensorBundle, TensorBundle messages, and NVIDIA TensorList conversion.
  Its parent CMake project builds only core and optional CUDA/HIP. Do not add
  an aggregate package.xml or include ROS/message/compat packages in that build.
- `gpu_ros_object_detection/` contains six packages: detection common, ONNX
  inference, RT-DETR, YOLOv8, validation, and NVIDIA reference compositions.
  Its benchmarks directory is not a ROS package. Detection tools, config,
  external manifests, Docker, and Apptainer facilities belong to this collection.
- `gpu_ros_nvidia_tensor_bundle_compat` owns generic conversions and the
  boundary launch. It is the only project package with a direct NVIDIA
  TensorList interface dependency. Full NVIDIA detection launches belong to
  `gpu_ros_nvidia_reference`. AMD excludes both packages. Core model packages
  must not acquire Isaac runtime dependencies; existing test dependencies are
  a separate boundary.
- Root `tools/` owns repository release and namespace-boundary checks.
  Methods belong under `skills/gpu-ros-experiments/`; results belong in `docs/`.

Preserve package basenames and ROS names, C++ namespaces, public includes,
CMake targets/aliases, plugin identifiers, message schemas, and model contracts.
Do not recreate the old classification roots, duplicate launches, or path
aliases. Update every caller when an owning path changes. Do not change
algorithms, kernels, compiler flags, provider/model policy, graphs, QoS,
executors, warmup, benchmark parameters, or output schemas during a layout change.

Managed transport owns buffer lifetime and synchronization, not preprocessing,
inference, decoding, or an inter-process zero-copy promise. ROS serialization
of device storage performs a host copy. Keep project TensorBundle messages
separate from NVIDIA TensorList types. A public `ITensorBundleIO` or buffer
contract change must update all affected callers and packages together.

## Repository and runtime paths

The monorepo root identifies source and releases. The detection root owns
facilities. Neither replaces the runtime workspace:

- AMD: `/workspaces/gpu-ros`, also `GPU_ROS_REPO_ROOT` and `OVG_WORKSPACE_ROOT`.
- NVIDIA: `/workspaces/isaac_ros-dev`, with the repository at `src/gpu-ros` and
  pinned external sources at `src/nvidia_external`.

Use one checkout bind, with assets, cache, results, ORT, build/install/log, and
home state on separate mounts. Never restore sibling-checkout variables or
redefine unrelated `OVG_*` fields. AMD requires the complete external install
selected by `OVG_ORT_ROOT`; image `/opt/onnxruntime` content is not a formal
runtime fallback.

The standalone source is `cmake -S gpu_ros_managed` from the repository root,
or `cmake -S .` from the managed directory. Use fresh out-of-tree build state.
Colcon must receive explicit package directories, not the managed parent CMake
project. Runtime defaults are in `gpu_ros_object_detection/docker/`.

## Execution and evidence

Use existing project commands and the selected method reference. AMD defaults
to explicit `OVG_RUNTIME=docker`. A site without Docker may select Apptainer
inside an authorized compute allocation. Never run GPU workloads on a login
node. Reuse verified dependency images, external sources, models, and data;
rebuild or export only when required and authorized.

Run acceptance payloads non-interactively. Keep the same external Compose
override on every command; do not call a launcher branch that drops it and
starts another container. Apptainer payloads use the launcher's bind/environment
contract and the mounted checkout entrypoint. `shell` is for human debugging,
not a verification prerequisite. The helper at
`gpu_ros_object_detection/tools/phase2` has only `env`, `assets`, `cache`, and
`build` groups, not `test`, `smoke`, or `benchmark`.

Run focused checks and preserve their actual exit codes and raw logs.
Benchmark `launch_test` commands measure performance and require an explicit
request. A wrapper exit of zero or valid JSON does not prove clean component
teardown. Keep numeric correctness, copy evidence, lifecycle status, and
throughput separate. Do not weaken thresholds, drop unmatched frames, or use
cross-lane REPORT_ONLY evidence to pass a failed same-lane migration gate.
Use [result acceptance](skills/gpu-ros-experiments/references/result-acceptance.md).

Record the executed revision, dirty/untracked fingerprints, runtime identity,
loaded ORT/provider, model/data hashes, commands, and raw artifact IDs. Matrix
manifests use schema 3 and one monorepo identity. Capture logs and promoted
summaries remain unversioned. Legacy records keep their original identities;
historical result files remain byte-for-byte unchanged. Documentation written
after a run must not replace its executed-source fingerprints.

## Assets, licenses, and privacy

Do not commit models, checkpoints, ONNX files, engines, datasets, bags, traces,
profiles, logs, caches, runtime images, external source trees, or generated
results. The offline asset helper is
`gpu_ros_object_detection/tools/phase2-assets`. A digest establishes identity,
not a redistribution license. Never substitute an asset or provider silently.

Preserve file-level copyright and modification notices. Root/component notices
and external licenses remain authoritative. Public methods and results must
not include credentials, private users/hosts/IPs, scheduler or device-instance
identifiers, allocation commands, driver repair, or absolute deployment paths.
Public GPU models, driver versions, and runtime digests are scientific identity,
not deployment secrets. Follow CONTRIBUTING.md and SECURITY.md.
