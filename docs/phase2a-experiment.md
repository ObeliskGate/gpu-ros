# Phase 2A: AMD Standard ROS 2 + MIGraphX

## Target path

Phase 2A validates only the AMD production path:

~~~text
Image
  -> RtDetrImageEncoderNode
  -> RtDetrPreprocessorNode
  -> OnnxInferenceNode(transport=std, execution_provider=migraphx)
  -> RtDetrDecoderNode
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

OVG_PREPARE_ASSETS=1 ./docker/phase2-amd.sh bootstrap
~~~

Commands invoking the launcher run on the host. `OVG_ORT_ROOT` is a
container-side path, but it must be exported in the host shell before
`verify`, `colcon`, or `shell`; the launcher passes it into the runtime and
uses its fingerprint for workspace isolation.

Inside the runtime, verify the device, selected ORT, and workspace:

~~~bash
phase2 env --verify
phase2 assets verify
~~~

For Apptainer, set OVG_RUNTIME=apptainer and OVG_APPTAINER_SIF to a
prebuilt SIF. The compute node does not need Docker, sudo, root, or fakeroot.
The launcher uses --rocm for device and driver-library passthrough; do not add
manual /dev/kfd or /dev/dri binds. The SIF build definition is
apptainer/phase2-amd.def.

The runtime paths are:

~~~text
/workspaces/amd_ros_object_detection
/workspaces/amd_ros_object_detection/src/gpu_ros_managed
/workspaces/ovg-ort
/workspaces/ovg-assets
/workspaces/ovg-cache
/workspaces/ovg-results
~~~

The launcher isolates build, install, and log directories by the external ORT
fingerprint, or by the image/SIF fingerprint when builtin ORT is used.

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

The canonical AMD paths are:

~~~text
/workspaces/ovg-assets/models/synthetica_detr_v1.0.0_onnx/sdetr_grasp.onnx
/workspaces/ovg-assets/datasets/r2bdataset2024_v1/r2b_robotarm
~~~

The NVIDIA and AMD asset layouts are intentionally different launcher
contracts; they identify the same model and dataset content.

## External ONNX Runtime

The image provides /opt/onnxruntime as a builtin fallback. Formal AMD
validation should use a repository-owned external ORT 1.23.1 build:

~~~text
/workspaces/ovg-ort/source/onnxruntime
/workspaces/ovg-ort/build/<ort-fingerprint>
/workspaces/ovg-ort/install/<ort-fingerprint>
~~~

Prepare a clean recursive v1.23.1 checkout. Build it inside the runtime, then
select the resulting install from the host:

~~~bash
# Host
./docker/phase2-amd.sh shell

# Runtime
export AMD_GPU_TARGETS=<actual-gfx>
export OVG_ORT_STATE_ROOT=/workspaces/ovg-ort
export ORT_BUILD_JOBS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
./tools/build-phase2a-external-ort.sh

exit

# Host
export OVG_ORT_ROOT=/workspaces/ovg-ort/install/<ort-fingerprint>
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
BUILD_TESTING=ON
~~~

Manual colcon commands must run inside the launcher runtime where
COLCON_DEFAULTS_FILE is loaded. Otherwise colcon may discover unrelated
packages.

## MIGraphX cache

Without an explicit ORT_MIGRAPHX_MODEL_CACHE_PATH, the entrypoint uses:

~~~text
/workspaces/ovg-cache/migraphx/<image-or-sif-fingerprint>
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
    isaac_ros_detection_validation \
  --event-handlers console_direct+

colcon test-result --all --verbose

launch_test \
  migrated_packages/isaac_ros_rtdetr_std/test/isaac_ros_std_rtdetr_pol_test.py
~~~

For interactive images, the launch defaults are 640x640 input and
use_max_dim_for_orig_size=false:

~~~bash
ros2 launch \
  isaac_ros_rtdetr_std \
  rtdetr_ort_std_image.launch.py \
  model_file_path:=/workspaces/ovg-assets/models/synthetica_detr_v1.0.0_onnx/sdetr_grasp.onnx \
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

ros2 run isaac_ros_detection_validation \
  run_amd_phase2a_fixed_input_capture.sh \
  amd_phase2a_fixed
~~~

The runner explicitly uses 1280x720, use_max_dim_for_orig_size=true, and
confidence_threshold=0.6. It checks only /detections_output and
/rtdetr/detections_output. Do not start a second graph, player, or recorder.

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
  --output-json /workspaces/ovg-results/amd-vs-nvidia-config-c.json
~~~

For the provider audit, require MIGraphX kernel events and record the exact
known CPU fallback node allowlist. Any additional CPU node is a failure.
Do not use --require-no-cpu-nodes because the documented int64 exception is
expected.

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
~~~

An AMD platform is supported for Phase 2A only after target, environment,
assets, Release build, tests, MIGraphX output, fixed-input comparison,
provider audit, library audit, benchmark JSON, and persistent-state reuse
all pass.
