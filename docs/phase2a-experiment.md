# Phase 2A: AMD Standard ROS 2 + MIGraphX

## Status

Phase 2A is complete. This document remains the procedure and regression
runbook for the accepted AMD standard ROS 2 paths. It covers both RT-DETR and
YOLOv8; the closure results are recorded separately in
[`phase2a-results.md`](phase2a-results.md).

## Target path

Phase 2A validates these AMD production paths:

~~~text
RT-DETR:
Image
  -> RtDetrImageEncoderNode
  -> RtDetrPreprocessorNode
  -> OnnxInferenceNode(transport=std, execution_provider=migraphx)
  -> RtDetrDecoderNode
  -> Detection2DArray

YOLOv8:
Image
  -> YoloV8ImageEncoderNode
  -> OnnxInferenceNode(transport=std, execution_provider=migraphx)
  -> YoloV8DecoderNode
  -> Detection2DArray
~~~

It does not reproduce the NVIDIA A/B/C/D matrix on AMD and does not replace
the standard path with Phase 2B managed transport.

The supported baseline is ROS 2 Jazzy, ONNX Runtime 1.23.1, MIGraphX, and a
Release colcon build. The gpu_ros_managed sibling is required by the shared
inference core even when transport=std.

## Runtime and persistent state

Docker and Apptainer use docker/phase2-amd.sh:

~~~bash
export OVG_RUNTIME=docker
export OVG_STATE_ROOT=<persistent-host-state-directory>
export GPU_ROS_MANAGED_DIR=<host-path-to-gpu_ros_managed>
export OVG_ASSETS_ROOT=/workspaces/ovg-assets
export OVG_RESULTS_ROOT=/workspaces/ovg-results

OVG_PREPARE_ASSETS=1 ./docker/phase2-amd.sh bootstrap
~~~

Commands invoking the launcher run on the host. `OVG_ORT_ROOT` is a
container-side path. It is required for formal `bootstrap`, `up`, `verify`,
and `colcon` runs; `shell` may omit it only when the shell is being used to
build the external ORT. The launcher passes it into the runtime and uses its
fingerprint for workspace isolation.

### Apptainer on a managed scheduler

When using a scheduler, request an interactive AMD GPU allocation according to
the local site policy and run the launcher from the allocated compute node.
The following is intentionally parameterized; partition names, host paths,
and image locations are deployment choices:

~~~bash
salloc -N 1 -n 1 -p <amd-gpu-partition> -t <wall-time>
srun --pty bash -l

export OVG_RUNTIME=apptainer
export OVG_REQUIRE_SLURM=1
export OVG_STATE_ROOT=<persistent-host-state-directory>
export OVG_ORT_STATE_HOST=<persistent-external-ort-state-directory>
export OVG_APPTAINER_SIF=<path-to-apptainer-image>
export GPU_ROS_MANAGED_DIR=<host-path-to-gpu_ros_managed>
export OVG_ORT_ROOT=<container-path-to-external-ort-install>

./docker/phase2-amd.sh preflight
~~~

`preflight` is mandatory before `colcon`, `verify`, capture, or benchmark
work. It checks the active allocation, AMD GPU partition, device access, and
required host paths. After an SSH disconnect or an expired allocation, request
a new allocation and rerun the environment block; do not reuse stale scheduler
state.

The launcher requires the key Apptainer host variables explicitly. The
supported local bypass is `OVG_REQUIRE_SLURM=0` for a deliberate
non-scheduler Apptainer setup. Docker workflows do not use this scheduler
guard.

Every AMD build, validation, capture, and benchmark run uses the
project-built external ONNX Runtime, regardless of GPU architecture. Any
image-bundled ORT copy is legacy build-only content and is
never a valid formal runtime selection. The launcher discovers a single
complete install under `${OVG_ORT_STATE_HOST:-<state-root>/ort}/install`;
when more than one exists, set `OVG_ORT_ROOT` explicitly. A shell without an
external install is permitted only to build that external install.

Inside the runtime, verify the device, selected ORT, and workspace:

~~~bash
phase2 env --verify
phase2 assets verify
~~~

For Apptainer, set `OVG_RUNTIME=apptainer` and the explicit host paths shown
above. The compute node does not need Docker, sudo, root, or fakeroot.
The launcher uses --rocm for device and driver-library passthrough; do not add
manual /dev/kfd or /dev/dri binds. The SIF build definition is
apptainer/phase2-amd.def.

The default runtime paths inside the container are:

~~~text
${OVG_WORKSPACE_ROOT:-/workspaces/amd_ros_object_detection}
${GPU_ROS_MANAGED_DIR:-/workspaces/amd_ros_object_detection/src/gpu_ros_managed}
${OVG_ORT_STATE_ROOT:-/workspaces/ovg-ort}
${OVG_ASSETS_ROOT:-/workspaces/ovg-assets}
${OVG_CACHE_ROOT:-/workspaces/ovg-cache}
${OVG_RESULTS_ROOT:-/workspaces/ovg-results}
~~~

The launcher isolates build, install, and log directories by the external ORT
fingerprint. The image/SIF fingerprint is used only by the build-only shell
before an external ORT install exists; it is never used for formal AMD work.

## GPU target and assets

Discover the actual runtime target:

~~~bash
rocminfo \
  | awk '$1 == "Name:" && $2 ~ /^gfx[0-9a-fA-F]+$/ {print $2}' \
  | sort -u
~~~

Set AMD_GPU_TARGETS to the sorted, deduplicated target set used for the image.
The image manifest is not a substitute for a real-device validation.

After entering the runtime, verify assets:

~~~bash
phase2 assets status
phase2 assets verify
~~~

The canonical AMD paths, relative to `${OVG_ASSETS_ROOT}`, are:

~~~text
${OVG_ASSETS_ROOT}/models/synthetica_detr_v1.0.0_onnx/sdetr_grasp.onnx
${OVG_ASSETS_ROOT}/datasets/r2bdataset2024_v1/r2b_robotarm
~~~

### Optional YOLOv8 asset

The repository neither contains nor downloads YOLOv8 weights and never runs
an Ultralytics export. A user must provide a local ONNX file and is
responsible for its source, applicable license, and lawful use:

~~~bash
export OVG_YOLOV8_ONNX_SOURCE=/path/to/user-provided/yolov8s.onnx
phase2 assets import-yolov8
phase2 assets verify-yolov8
~~~

`phase2 assets` and the default `phase2 assets verify` manage and
verify only the required RT-DETR/R2B assets; YOLOv8 is optional. Compatibility
is identified by the recorded SHA-256
`d6e22418dd1acc69a232a1b297c01dfc785842fd11a4a84546c84e14cdeb235c` and the
following contract: YOLOv8s, Ultralytics 8.4.67, COCO 80 classes, opset 17,
static input `[1,3,640,640]` named `images`, and output `[1,84,8400]` named
`output0` without built-in NMS. An export recipe cannot guarantee identical
model bytes, so the SHA-256 check is authoritative.

The standard YOLOv8 launch, AMD fixed-input capture, and AMD benchmark fail
before graph startup when this asset is missing. The package-level AMD
MIGraphX POL uses a generated tiny YOLOv8-shaped ONNX model, like the NVIDIA
POL, so build and graph-contract tests do not require the user asset. The
fixed-input runner retains its RT-DETR invocation and adds the explicit
YOLOv8 lane:

~~~bash
run_amd_phase2a_fixed_input_capture.sh yolov8 <output-name>
~~~

The NVIDIA and AMD asset layouts are intentionally different launcher
contracts; they identify the same model and dataset content.

## External ONNX Runtime

The image-bundled ORT is used only to support image construction and the
external-ORT build shell. Formal AMD work always uses a repository-owned
external ORT 1.23.1 build:

~~~text
${OVG_ORT_STATE_ROOT}/source/onnxruntime
${OVG_ORT_STATE_ROOT}/build/<ort-fingerprint>
${OVG_ORT_STATE_ROOT}/install/<ort-fingerprint>
~~~

Prepare a clean recursive v1.23.1 checkout. Build it inside the runtime, then
select the resulting install from the host:

~~~bash
# Host, inside the active compute allocation. This is the only shell that may
# run without an external ORT install.
unset OVG_ORT_ROOT
OVG_ALLOW_BUILD_ONLY_SHELL=1 ./docker/phase2-amd.sh shell

# Runtime
export AMD_GPU_TARGETS=<actual-gfx>
export OVG_ORT_STATE_ROOT=<container-external-ort-state-root>
export ORT_BUILD_JOBS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
./tools/build-phase2a-external-ort.sh

exit

# Host
export OVG_ORT_ROOT=${OVG_ORT_STATE_ROOT}/install/<ort-fingerprint>
./docker/phase2-amd.sh preflight
./docker/phase2-amd.sh verify
./docker/phase2-amd.sh colcon
./docker/phase2-amd.sh shell
phase2 env --verify
phase2 assets verify
~~~

The ordered patch series contains MIGraphX Linux build compatibility,
GridSample provider support, and the controlled RT-DETR int64 query-index
Div CPU fallback. The fingerprint includes the ORT source, patch series and
GPU target.

External mode fails if headers, libonnxruntime.so,
libonnxruntime_providers_shared.so, libonnxruntime_providers_migraphx.so, or
the fingerprint marker is missing. It never silently falls back to
/opt/onnxruntime.

## AMD build profile

The official AMD colcon profile is
docker/colcon-defaults-phase2a-amd.yaml. It must use:

~~~text
CMAKE_BUILD_TYPE=Release
BUILD_NITROS_TRANSPORT=OFF
ORT_ENABLE_CUDA=OFF
ORT_ENABLE_ROCM=OFF
ORT_ENABLE_MIGRAPHX=ON
BUILD_MIGRAPHX_POL_TEST=ON
BUILD_NVIDIA_YOLOV8_POL_TEST=OFF
BUILD_YOLOV8_MIGRAPHX_POL_TEST=OFF
BUILD_TESTING=ON
~~~

`./docker/phase2-amd.sh colcon` remains the canonical complete-workspace build
entrypoint. For interactive development, use the supported shell and native
colcon commands directly:

~~~bash
./docker/phase2-amd.sh shell
colcon build --packages-select gpu_ros_managed_hip
colcon build --packages-up-to isaac_ros_onnx_inference
colcon test --packages-select isaac_ros_yolov8_std
~~~

The shell exports `COLCON_DEFAULTS_FILE`, so these commands inherit the AMD
profile, Release build type, merged install layout, HIP backend, MIGraphX
provider, and NVIDIA YOLOv8 POL exclusion. The shell does not wrap or restrict
native `colcon`; normal package-selection arguments remain available.

Do not reconstruct the full AMD profile manually with `--cmake-args` for
ordinary package debugging. A partial command can replace the profile defaults
and accidentally re-enable unrelated NVIDIA-only tests or transports.

## MIGraphX cache

Without an explicit ORT_MIGRAPHX_MODEL_CACHE_PATH, the entrypoint uses:

~~~text
${OVG_CACHE_ROOT:-/workspaces/ovg-cache}/migraphx/<image-or-sif-fingerprint>
~~~

Inspect it with:

~~~bash
phase2 cache status
~~~

phase2 cache clean migraphx clears all entries below the default MIGraphX
cache root. Use it only when investigating stale or incompatible cache data.
It does not clean an explicitly selected custom cache path.

## Tests and graph execution

Run the AMD test set:

The AMD profile excludes the legacy NVIDIA YOLOv8 POL test. That test imports
`isaac_ros_test` and belongs only to an explicit NVIDIA test configuration;
its registration is not part of AMD closure.

~~~bash
colcon test \
  --merge-install \
  --packages-select \
    gpu_ros_managed_core \
    gpu_ros_managed_hip \
    gpu_ros_managed_ros \
    gpu_ros_managed_tensor_list \
    isaac_ros_onnx_inference \
    isaac_ros_rtdetr_std \
    isaac_ros_yolov8_std \
    isaac_ros_detection_validation \
  --event-handlers console_direct+

colcon test-result --all --verbose

launch_test \
  migrated_packages/isaac_ros_rtdetr_std/test/isaac_ros_std_rtdetr_pol_test.py

launch_test \
  migrated_packages/isaac_ros_yolov8_std/test/isaac_ros_yolov8_migraphx_pol_test.py
~~~

The standard YOLOv8 unit tests run with the package test set. The generated
tiny-model MIGraphX POL can be enabled selectively with
`BUILD_YOLOV8_MIGRAPHX_POL_TEST=ON` when the package is reconfigured. It does
not require the canonical user-provided model.

For interactive images, the launch defaults are 640x640 input and
use_max_dim_for_orig_size=false:

~~~bash
ros2 launch \
  isaac_ros_rtdetr_std \
  rtdetr_ort_std_image.launch.py \
  model_file_path:=${OVG_ASSETS_ROOT}/models/synthetica_detr_v1.0.0_onnx/sdetr_grasp.onnx \
  image_topic:=/camera_1/color/image_raw \
  execution_provider:=migraphx
~~~

For fixed-input validation, use only the single-terminal runner:

~~~bash
CAPTURE_EXECUTION_PROVIDER=migraphx \
CAPTURE_WARMUP_ONLY=1 \
CAPTURE_FIRST_OUTPUT_TIMEOUT_SECONDS=900 \
ros2 run isaac_ros_detection_validation \
  run_amd_phase2a_fixed_input_capture.sh \
  migraphx_probe

CAPTURE_EXECUTION_PROVIDER=migraphx \
CAPTURE_WARMUP_ONLY=1 \
CAPTURE_FIRST_OUTPUT_TIMEOUT_SECONDS=900 \
ros2 run isaac_ros_detection_validation \
  run_amd_phase2a_fixed_input_capture.sh \
  yolov8 yolov8_migraphx_probe

ros2 run isaac_ros_detection_validation \
  run_amd_phase2a_fixed_input_capture.sh \
  amd_phase2a_fixed

ros2 run isaac_ros_detection_validation \
  run_amd_phase2a_fixed_input_capture.sh \
  yolov8 amd_phase2a_yolov8_fixed
~~~

The runner explicitly uses 1280x720, use_max_dim_for_orig_size=true, and
confidence_threshold=0.6. It checks only /detections_output and
/rtdetr/detections_output for RT-DETR; the YOLOv8 lane checks
/detections_output and /yolov8/detections_output. Do not start a second graph,
player, or recorder.

## Validation

Compare the candidate bag with the NVIDIA reference using stamp matching when
source timestamps are preserved; otherwise use index matching only after input
order is confirmed:

~~~bash
ros2 run isaac_ros_detection_validation \
  compare_detection2d_bags.py \
  --reference-bag <reference-bag> \
  --candidate-bag <candidate-bag> \
  --match-policy <stamp-or-index> \
  --output-json ${OVG_RESULTS_ROOT}/amd-vs-nvidia-config-c.json
~~~

For both lanes, require MIGraphX kernel events and record the provider
assignment. The final MI350X RT-DETR profile recorded five CPU fallback nodes,
all in the postprocessor:

~~~text
/postprocessor/Div
/postprocessor/GatherElements
/postprocessor/Mod
/postprocessor/Tile_1
/postprocessor/Unsqueeze_5
~~~

The fallback is a known, controlled performance limitation and does not
invalidate Phase 2A correctness. An unexpected additional CPU node is an audit
failure.
Do not use `--require-no-cpu-nodes` for RT-DETR. The YOLOv8 closure profile
showed no CPU fallback, so a YOLOv8 profile may use that strict check when the
profile is isolated from the RT-DETR run.

While the graph is running, inspect the ORT session process:

~~~bash
PID="$(pgrep -n -f '[c]omponent_container_mt')"
grep -E 'libonnxruntime.*\.so|/opt/onnxruntime' "/proc/${PID}/maps"
~~~

External mode must load all three ORT libraries from the selected install and
must not load an ORT library from /opt/onnxruntime.

Run the throughput benchmark only after tests, warm-up, validation, and the
library audit pass:

~~~bash
export R2B_RESULT_FILE=phase2a_amd_<platform-tag>.json
export MIGRAPHX_WARMUP_TIMEOUT_SEC=900
launch_test \
  migrated_packages/benchmarks/isaac_ros_rtdetr_phase2a_amd_graph.py

launch_test \
  migrated_packages/benchmarks/isaac_ros_yolov8_phase2a_amd_graph.py
~~~

An AMD platform is supported for Phase 2A after both lanes have passed their
target and environment checks, assets, Release build, package and POL tests,
MIGraphX output, fixed-input capture and comparison, provider audit, external
library audit, benchmark, and persistent-state checks. The standard ROS 2
paths remain independently usable during Phase 2B work.
