# AMD Phase 2 environment

This document covers environment preparation and the correct Phase 2A run
method. It does not change the Phase 2B managed transport implementation or
add managed validation graphs.

## Correct Phase 2A run method

Phase 2A is the standard-ROS2 RT-DETR path.  The image-input launch is the
production path:

```text
Image -> RtDetrImageEncoderNode -> RtDetrPreprocessorNode
      -> OnnxInferenceNode(transport=std, execution_provider=migraphx)
      -> RtDetrDecoderNode -> Detection2DArray
```

After selecting an exact external ORT install (or deliberately using the
image's `/opt/onnxruntime` fallback), run these commands from the host:

```bash
export OVG_RUNTIME=docker                 # use apptainer on a SIF instead
export OVG_ORT_ROOT=/workspaces/ovg-ort/install/<ort-fingerprint>
./docker/phase2-amd.sh verify
./docker/phase2-amd.sh colcon
./docker/phase2-amd.sh shell
```

Inside the shell, verify the runtime and assets before running the graph:

```bash
phase2 env --verify
phase2 assets verify
```

For the reproducible fixed-input validation, use the one-terminal runner.  It
starts the graph, waits for the first MIGraphX output, plays the input bag,
records the actual `Detection2DArray` topic, and cleans up all child processes:

```bash
ros2 run isaac_ros_detection_validation \
  run_amd_phase2a_fixed_input_capture.sh \
  amd_phase2a_fixed
```

Do not start a second `ros2 launch`, `ros2 bag play`, or recorder for this
capture.  For an interactive graph with a live image publisher, the equivalent
launch is:

```bash
ros2 launch isaac_ros_rtdetr_std rtdetr_ort_std_image.launch.py \
  model_file_path:=/workspaces/ovg-assets/models/synthetica_detr_v1.0.0_onnx/sdetr_grasp.onnx \
  image_topic:=/camera_1/color/image_raw \
  execution_provider:=migraphx
```

The managed RT-DETR launch is a separate CUDA/NITROS path; it is not the AMD
Phase 2A graph.  Keep `execution_provider:=migraphx` explicit for AMD and do
not rely on a CPU fallback.

## Shortest environment workflow

Docker:

```bash
OVG_PREPARE_ASSETS=1 ./docker/phase2-amd.sh bootstrap
./docker/phase2-amd.sh colcon
./docker/phase2-amd.sh shell
phase2 env --verify
```

Apptainer on a compute allocation:

```bash
OVG_RUNTIME=apptainer OVG_PREPARE_ASSETS=1 ./docker/phase2-amd.sh bootstrap
OVG_RUNTIME=apptainer ./docker/phase2-amd.sh colcon
OVG_RUNTIME=apptainer ./docker/phase2-amd.sh shell
phase2 env --verify
```

`shell`, `verify`, and `colcon` use one-shot `apptainer exec` by default.
Set `OVG_APPTAINER_INSTANCE=1` only when a user-level instance is useful for
interactive work. The instance is optional and is not the HPC default.

## Persistent state

The default host state root is `.ovg/`; set `OVG_STATE_ROOT` to put it outside
the repository. Assets, results, and cache are shared across image revisions.
With builtin ORT, build, install, and log directories are isolated by the
Docker image ID or SIF SHA256. With an external ORT install they are isolated by
the exact ORT fingerprint, so an updated ORT build cannot reuse stale CMake
cache or RPATHs.

Inside every runtime the paths are:

```text
/workspaces/amd_ros_object_detection
/workspaces/amd_ros_object_detection/src/gpu_ros_managed
/workspaces/ovg-ort
/workspaces/ovg-assets
/workspaces/ovg-cache
/workspaces/ovg-results
```

The host state layout for external ORT is:

```text
<OVG_STATE_ROOT>/ort/source/onnxruntime
<OVG_STATE_ROOT>/ort/build/<ort-fingerprint>/source
<OVG_STATE_ROOT>/ort/install/<ort-fingerprint>
```

`OVG_ORT_ROOT` is a container path, normally
`/workspaces/ovg-ort/install/<ort-fingerprint>`.  The launcher binds the host
directory `<OVG_STATE_ROOT>/ort` (or `OVG_ORT_STATE_HOST`) at
`/workspaces/ovg-ort`.  An external install is immutable once created: the
builder refuses to overwrite an existing fingerprint directory.

The MIGraphX cache is placed below a fingerprinted directory under
`/workspaces/ovg-cache/migraphx`. It is not stored with the original assets.

## Assets and offline use

`phase2 assets` creates the canonical model and dataset layout and
generates `/workspaces/ovg-assets/manifest.json`. Existing, verified assets are
not downloaded again. In an offline environment, copy the following paths into
the assets directory before running `prepare`:

```text
models/synthetica_detr_v1.0.0_onnx/sdetr_grasp.onnx
datasets/r2bdataset2024_v1/r2b_robotarm
```

For an online first preparation, export `NGC_CLI_API_KEY` before entering the
container. The NGC client is already part of the runtime image.

The helper creates compatibility symlinks used by older scripts. If an asset is
missing and cannot be downloaded, it prints the exact missing path and exits.

## External ORT development build

The runtime image contains a fixed `/opt/onnxruntime` fallback.  For
development and reproducible AMD validation, keep the ORT checkout, patches,
build tree, install and colcon workspace outside the image.  The runtime image
contains the compiler, CMake, Ninja and Python packaging dependencies needed by
the builder.

From a MI350X allocation, first run only this environment audit and save the
output; then run the same command on the AMD Docker GPU host:

```bash
rocminfo | awk '$1 == "Name:" && $2 ~ /^gfx[0-9]/ {print $2}' | sort -u
```

Then build one image with the de-duplicated union of the actual targets from
the Docker GPU host and the MI350X allocation. Replace the angle-bracket
placeholders with the values printed by `rocminfo`; they are deliberately not
hard-coded here:

```bash
docker_host_gfx="<docker-host-gfx>"
mi350x_gfx="<mi350x-gfx>"
export AMD_GPU_TARGETS="${docker_host_gfx},${mi350x_gfx}"
export OVG_RUNTIME=docker
./docker/phase2-amd.sh build
```

The Docker host only proves its own target.  The union in the image manifest
does not prove MI350X compatibility; that requires the later allocation test.

Start the validated image and clone a pristine recursive ORT 1.23.1 source
into the persistent external state:

```bash
./docker/phase2-amd.sh shell
git clone \
  --branch v1.23.1 \
  --depth 1 \
  --recursive \
  https://github.com/microsoft/onnxruntime.git \
  /workspaces/ovg-ort/source/onnxruntime
export AMD_GPU_TARGETS="<docker-host-gfx>"
export OVG_ORT_STATE_ROOT=/workspaces/ovg-ort
export ORT_BUILD_JOBS="${SLURM_CPUS_PER_TASK:-24}"
./tools/build-phase2a-external-ort.sh
```

The builder applies the ordered series in
`docker/patches/onnxruntime-1.23.1.series`, computes
`<ort-commit12>-<patchset-sha12>-<gfx>`, and writes only a new install directory
containing the headers, shared ORT library, shared provider library,
MIGraphX provider library, `.ovg-ort-fingerprint`, and `build-info.txt`.
`build-info.txt` records the source commit, patch hashes, gfx target and exact
build command.  Image IDs and SIF hashes belong in the surrounding run record,
not in the ORT fingerprint.

After the builder prints its install path, export that exact container path in
the host shell and isolate the colcon workspace by the same ORT fingerprint:

```bash
export OVG_ORT_ROOT=/workspaces/ovg-ort/install/<ort-fingerprint>
./docker/phase2-amd.sh verify
./docker/phase2-amd.sh colcon
```

In external mode the entrypoint sets `ONNXRUNTIME_ROOT`,
`ONNXRUNTIME_INCLUDE_DIR`, `ONNXRUNTIME_LIBRARY`, and prepends the external
`lib` directory to `LD_LIBRARY_PATH`.  A missing header, library, provider, or
fingerprint marker fails immediately; `/opt/onnxruntime` is never used as a
fallback in that mode.  The workspace directories are
`build/<ort-fingerprint>`, `install/<ort-fingerprint>`, and
`log/<ort-fingerprint>`.

For the external-library audit, while the capture process is alive, inspect
the component container (or the process that owns the ORT session):

```bash
pid="$(pgrep -n component_container_mt)"
grep -E 'libonnxruntime|/opt/onnxruntime' "/proc/${pid}/maps"
```

The three ORT libraries must resolve below the selected
`/workspaces/ovg-ort/install/<ort-fingerprint>/lib`; no `/opt/onnxruntime`
library may be mapped.

## GPU targets

Image build and runtime detection are separate. A build machine without an AMD
GPU can use an explicit target:

```bash
AMD_GPU_TARGETS=<actual-gfx> ./docker/phase2-amd.sh build
```

If `AMD_GPU_TARGETS` is not set, the host build path attempts `rocminfo` and
fails when no target can be detected. Runtime `phase2 env --verify` runs
`rocminfo` inside the container and checks the result against the image
manifest; it never silently selects a target.

## Apptainer image

The Docker GPU host must also have a working Apptainer before this gate starts:

```bash
apptainer version
```

Do not move the SIF build to the cluster or another temporary machine. After
Docker external-ORT verification passes, save the exact image and archive hash
on that same machine:

```bash
image_id="$(docker image inspect --format '{{.Id}}' ovg-phase2-amd:local)"
image_id12="${image_id#sha256:}"
image_id12="${image_id12:0:12}"
docker image save \
  --output "phase2-amd-dev-${image_id12}.docker.tar" \
  ovg-phase2-amd:local
sha256sum "phase2-amd-dev-${image_id12}.docker.tar" \
  | tee "phase2-amd-dev-${image_id12}.docker.tar.sha256"
```

Build a candidate SIF from that archive and the repository definition, then
record its hash and manifest. `OVG_AMD_IMAGE_URI` must be an absolute archive
path:

```bash
apptainer build \
  --build-arg \
  OVG_AMD_IMAGE_URI="docker-archive:///absolute/path/phase2-amd-dev-${image_id12}.docker.tar" \
  phase2-amd-dev.partial.sif \
  apptainer/phase2-amd.def
sha256sum phase2-amd-dev.partial.sif \
  | tee phase2-amd-dev.partial.sif.sha256
apptainer exec phase2-amd-dev.partial.sif \
  cat /opt/ovg/image-manifest.json
```

Confirm that the manifest's `gpu_targets` is the same de-duplicated union
passed to Docker. Name and archive the final SIF using the image ID, target set,
and SIF SHA; keep those facts separate from the ORT fingerprint.

Run the candidate SIF on the same Docker GPU host before copying anything to
the cluster. Reuse the Docker-built external ORT install:

```bash
export OVG_RUNTIME=apptainer
export OVG_APPTAINER_SIF=/absolute/path/<final-sif>
export OVG_ORT_ROOT=/workspaces/ovg-ort/install/<docker-host-ort-fingerprint>
./docker/phase2-amd.sh verify
./docker/phase2-amd.sh colcon
./docker/phase2-amd.sh shell
```

At minimum repeat `phase2 env --verify`, the MIGraphX POL, the fixed-input
capture, the provider profile, and the `/proc/<pid>/maps` check for the three
external ORT libraries. Generate the NVIDIA Config C comparison JSON from this
SIF candidate as well. Only when both Docker and same-host SIF runs pass may
the artifacts be copied to `/work1`.

HPC nodes consume a prebuilt SIF through `OVG_APPTAINER_SIF`, or pull one time
from `OVG_APPTAINER_IMAGE_URI` when networking and the registry permit it. The
host launcher binds the application checkout, sibling checkout, assets, cache,
results, external ORT state, and fingerprinted build directories. It does not
require Docker, sudo, root, or fakeroot on the compute node.
The default device bind exposes `/dev/kfd` and `/dev/dri`; keep this mode and do
not add Apptainer's `--rocm` flag.

`apptainer/phase2-amd.def` is a build-machine definition for converting a
validated OCI runtime image. Supply the image URI as its `OVG_AMD_IMAGE_URI`
build argument; it is not required on the HPC node.

## Phase 2A fixed-input capture

The Phase 2A capture runner owns the graph, first-run MIGraphX warm-up, bag
recorder, input playback, and process cleanup. Run it inside the container; do
not start any of those processes in separate terminals:

```bash
ros2 run isaac_ros_detection_validation \
  run_amd_phase2a_fixed_input_capture.sh \
  amd_phase2a_fixed
```

The runner uses the same RT-DETR size contract as the NVIDIA Config C fixed-input
capture: a 1280x720 source with `use_max_dim_for_orig_size=true`, which supplies
`orig_target_sizes=[1280,1280]`. This setting is intentional for numerical
comparison with the archived Config C bag.

The bag is written below
`/workspaces/ovg-results/phase2a-bags/amd_phase2a_fixed`. Logs, the exact
command, and the first warm-up detection are written below the adjacent
`logs/` directory. Output names are never overwritten.

Compare that bag with the archived NVIDIA Config C bag using the actual
`Detection2DArray` topic discovered in each bag. Use index matching when the
two machines produced different recording timestamps:

```bash
ros2 run isaac_ros_detection_validation compare_detection2d_bags.py \
  --reference-bag /workspaces/ovg-results/<nvidia-config-c-bag> \
  --candidate-bag /workspaces/ovg-results/phase2a-bags/amd_phase2a_fixed \
  --match-policy index \
  --output-json /workspaces/ovg-results/amd-vs-nvidia-config-c.json
```

Do not assume a `/rtdetr` topic prefix, fixed message count, fixed hash, or a
new numeric threshold; the comparison tool's bag discovery and existing
thresholds are the contract.

For a one-frame provider diagnostic without recording the complete dataset:

```bash
CAPTURE_EXECUTION_PROVIDER=cpu \
CAPTURE_WARMUP_ONLY=1 \
ros2 run isaac_ros_detection_validation \
  run_amd_phase2a_fixed_input_capture.sh \
  cpu_probe

CAPTURE_EXECUTION_PROVIDER=migraphx \
CAPTURE_WARMUP_ONLY=1 \
ros2 run isaac_ros_detection_validation \
  run_amd_phase2a_fixed_input_capture.sh \
  migraphx_probe
```

For a profile-only provider audit, keep the graph and playback orchestration
in the same runner and disable bag recording:

```bash
CAPTURE_RECORD=0 \
CAPTURE_ORT_PROFILE_PREFIX=/workspaces/ovg-results/profiles/amd-migraphx \
ros2 run isaac_ros_detection_validation \
  run_amd_phase2a_fixed_input_capture.sh \
  amd_migraphx_profile
```

The default first-output timeout is 900 seconds because the first MIGraphX
compile on MI300X can take several minutes. Override it explicitly when
diagnosing a different machine with `CAPTURE_FIRST_OUTPUT_TIMEOUT_SECONDS`.

The repository-owned ORT patches keep GridSample assigned to MIGraphX and
normalize packed, non-contiguous graph outputs before ORT consumes them. This
correctness workaround uses the installed MIGraphX public API and stages only
affected outputs through host memory; it is not a zero-copy path. Padded or
broadcast output layouts fail explicitly. The output-layout patch also uses a
versioned MXR filename prefix, so an existing pre-patch cache is ignored without
deleting other cache entries.

## MI350X allocation gate

Copy the validated SIF and SHA, Docker archive and SHA, image manifest, both
repository checkouts, pristine ORT source, assets, the NVIDIA reference bag,
and the Docker/SIF run records to persistent `/work1`. In the allocation:

```bash
sha256sum --check phase2-amd-dev-*.sha256
export OVG_RUNTIME=apptainer
export OVG_APPTAINER_SIF=/work1/<final-sif>
export OVG_STATE_ROOT=/work1/ovg-state
rocminfo | awk '$1 == "Name:" && $2 ~ /^gfx[0-9]/ {print $2}' | sort -u
./docker/phase2-amd.sh verify
```

`phase2 env --verify` must show that every runtime gfx is present in the image
manifest target set. Reuse the same SIF, but build a new external ORT install
with the actual MI350X gfx target and then point `OVG_ORT_ROOT` at that install:

```bash
export AMD_GPU_TARGETS="<mi350x-gfx>"
export OVG_ORT_STATE_ROOT=/workspaces/ovg-ort
export ORT_BUILD_JOBS="${SLURM_CPUS_PER_TASK:-24}"
./tools/build-phase2a-external-ort.sh
export OVG_ORT_ROOT=/workspaces/ovg-ort/install/<mi350x-ort-fingerprint>
./docker/phase2-amd.sh colcon
```

Inspect the CMake cache and link information, then repeat managed/HIP package
smoke tests, the MIGraphX POL, the Phase 2A fixed-input capture, NVIDIA Config
C comparison, profile-only provider audit, and `/proc/<pid>/maps`. The maps
check must find only the selected external install's
`libonnxruntime.so`, `libonnxruntime_providers_shared.so`, and
`libonnxruntime_providers_migraphx.so`, with no `/opt/onnxruntime` copy loaded.
Use the bag's actual `Detection2DArray` topic and message count for comparison;
do not assume a `/rtdetr` prefix or introduce a new threshold.

Repeat the run from a fresh allocation to confirm that the SIF, source,
external install, fingerprinted workspace, cache, and result directories are
reusable. MI350X compatibility can only be claimed after this real-device
sequence passes; Docker host validation is not a substitute.

## Scope boundary

This work does not add AMD managed RT-DETR launch files, bridges, benchmarks,
or zero-copy behavior. The existing managed/HIP smoke tests remain the only
managed AMD check; the Phase 2A capture runner above is the standard ROS2 +
MIGraphX target path.
