# Phase 0: NVIDIA Benchmark Reproduction

## Scope

Phase 0 runs the unmodified NVIDIA Isaac ROS object-detection benchmark graphs
to establish a reference baseline for the migration:

- RT-DETR
- Grounding DINO

DetectNet is outside the current migration scope because its decoder is a GXF
component rather than the TensorBundle-based structure used by RT-DETR and
Grounding DINO.

The active environment is Isaac ROS 4.5. Existing 4.4 results remain
historical until explicitly rerun and labeled as 4.5.

## Requirements

- NVIDIA GPU and compatible driver
- Docker and NVIDIA Container Toolkit
- Git
- user-acquired copies of the required models and dataset outside this repository
- gpu_ros_managed checkout (the exact revision is enforced only for a managed
  transport reproduction)

NITROS requires the CUDA memory-pool API. If
cudaDeviceGetDefaultMemPool is unsupported, startup may fail as a cascade of
GXF component errors. Treat this as a runtime capability issue, not a graph
configuration issue.

The host driver and image CUDA version must be compatible. Do not force an old
compatibility library ahead of a newer host driver; use the compatibility
library only when the image and driver requirements call for it.

## Start the NVIDIA runtime

Configure `OVG_NVIDIA_EXTERNAL_SOURCE_ROOT` to the source root populated by
`tools/bootstrap-nvidia-external.sh`, then run:

~~~bash
./docker/phase1-nvidia.sh bootstrap
./docker/phase1-nvidia.sh verify
./docker/phase1-nvidia.sh shell
~~~

The helper checks the pinned image, ORT 1.23.1, the manually selected sibling
checkout, the benchmark packages, and the configured asset root. It reports
the sibling commit but does not enforce a commit pin.

It checks the asset root itself, not the contents of every model and dataset.

## Prepare assets

Acquire vendor-controlled assets outside this repository using the account and
license workflow appropriate to your organization. The repository contains no
NGC client or credential path. Import the resulting local files/directories:

~~~bash
export OVG_ASSETS_ROOT=<persistent-assets-root>
export ASSETS_ROOT="${OVG_ASSETS_ROOT}"

tools/phase2-assets import-model \
  --profile nvidia_synthetica \
  --source <external-sdetr_grasp.onnx>

tools/phase2-assets import-r2b \
  --source <external-r2b_robotarm-directory>
~~~

Create the paths expected by the official benchmark packages:

~~~bash
ln -sfn \
  ${OVG_ASSETS_ROOT}/models/synthetica_detr_v1.0.0_onnx \
  ${OVG_ASSETS_ROOT}/models/sdetr

mkdir -p ${OVG_ASSETS_ROOT}/datasets/r2b_dataset
ln -sfn \
  ${OVG_ASSETS_ROOT}/datasets/r2bdataset2024_v1/r2b_robotarm \
  ${OVG_ASSETS_ROOT}/datasets/r2b_dataset/r2b_robotarm

mkdir -p ${OVG_ASSETS_ROOT}/models/grounding_dino
cp <external-grounding-dino.onnx> \
  ${OVG_ASSETS_ROOT}/models/grounding_dino/grounding_dino_model.onnx
~~~

The required files, relative to `${ASSETS_ROOT}`, are:

~~~text
${ASSETS_ROOT}/models/sdetr/sdetr_grasp.onnx
${ASSETS_ROOT}/datasets/r2b_dataset/r2b_robotarm
${ASSETS_ROOT}/models/grounding_dino/grounding_dino_model.onnx
~~~

Prepare assets once and retain them in the persistent assets volume.

## Run the benchmarks

~~~bash
export ROS2_BENCHMARK_OVERRIDE_ASSETS_ROOT=${ASSETS_ROOT}

launch_test \
  "$(ros2 pkg prefix isaac_ros_rtdetr_benchmark)"/share/isaac_ros_rtdetr_benchmark/scripts/isaac_ros_rtdetr_graph.py

launch_test \
  "$(ros2 pkg prefix isaac_ros_grounding_dino_benchmark)"/share/isaac_ros_grounding_dino_benchmark/scripts/isaac_ros_grounding_dino_graph.py
~~~

The first run may build TensorRT engines and will take longer.

## Success and archival

The benchmark has passed when the real test reports:

~~~text
Ran 1 test ... OK
~~~

A later shutdown message such as Ran 0 tests / NO TESTS RAN does not invalidate
an earlier successful test.

Archive the benchmark JSON, complete launch log, GPU and driver information,
CUDA/ROS/Isaac ROS versions, model and dataset identities, command line, and
repository commit in a persistent result location. Files left only in
container /tmp are not archived results.
