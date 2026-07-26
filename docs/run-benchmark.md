# Benchmark Reproduction Guide

How to run the NVIDIA Isaac ROS 4.5 object-detection benchmarks (RT-DETR and
Grounding DINO) on a remote server using this repository's Docker dev
environment. The Phase 0/1 result artifacts in this repository were collected
on 4.4 and remain historical until explicitly rerun on 4.5.

## Prerequisites

### GPU

- ≥ 8 GB of VRAM
- **Must support the CUDA Memory Pool API** (`cudaDeviceGetDefaultMemPool` does not return `cudaErrorNotSupported`)
  - NITROS hard-depends on this API at startup. If unsupported, you'll see a cascade of `GXF_FACTORY_UNKNOWN_CLASS_NAME` errors that look like a GXF misconfiguration but are actually a GPU limitation.
  - Whole-GPU passthrough and MIG instances usually support it.
  - Some vGPU profiles (e.g. NVIDIA GRID `-XC` series under certain vGPU manager versions) disable memory pools at the hypervisor level. There is no software workaround on those machines — you have to use a different host.

### Driver / Container

- NVIDIA Driver ≥ 580 works out of the box. If your host driver is older (e.g. 535) and the base image ships CUDA 13, install `cuda-compat-13-0` inside the container for forward compatibility:
  ```bash
  apt-get install -y cuda-compat-13-0
  echo /usr/local/cuda-13.0/compat > /etc/ld.so.conf.d/cuda-compat.conf
  ldconfig
  ```
  Don't install `cuda-compat-13-0` if your host driver is already newer than 580 — forcing the compat lib first in `ld.so.conf.d` would cause the runtime to use older driver libraries instead of the host's.
- Docker + nvidia-container-toolkit
- Git

## Steps

### 1. Clone the repository

```bash
git clone --recurse-submodules <repo-url>
cd amd_ros_object_detection
```

### 2. Configure the NGC API key

Create a `.env` file in the project root with your NGC API key (get one at https://org.ngc.nvidia.com/setup/api-key):

```
NGC_CLI_API_KEY=<your-key>
```

### 3. Build the Docker image

```bash
docker compose build
```

First build takes ~10–15 minutes (base image pull + apt install).

### 4. Start the container

```bash
docker compose up -d
docker compose exec dev bash
```

### 5. Download models and datasets

Run inside the container:

```bash
mkdir -p /workspaces/isaac_ros-dev/assets/models /workspaces/isaac_ros-dev/assets/datasets

# RT-DETR model (~320 MB)
ngc registry model download-version nvidia/isaac/synthetica_detr:1.0.0_onnx \
  --dest /workspaces/isaac_ros-dev/assets/models/ --org nvidia

# r2b_robotarm dataset (in r2bdataset2024; shared between RT-DETR and Grounding DINO, ~1.5 GB)
ngc registry resource download-version nvidia/isaac/r2bdataset2024:1 \
  --dest /workspaces/isaac_ros-dev/assets/datasets/ --org nvidia

# Grounding DINO model (~689 MB)
ngc registry model download-version \
  nvidia/tao/grounding_dino:grounding_dino_swin_tiny_commercial_deployable_v1.0 \
  --dest /tmp/ --org nvidia --team tao
```

The downloads land in a Docker named volume (`assets`), so they survive image rebuilds.

### 6. Set up paths

Benchmark scripts expect specific paths. NGC's directory naming includes a version suffix that needs to be aliased to what the scripts look for:

```bash
cd /workspaces/isaac_ros-dev/assets

# RT-DETR model symlink (script looks for models/sdetr/sdetr_grasp.onnx)
ln -sf synthetica_detr_v1.0.0_onnx models/sdetr

# Dataset symlink (script looks for datasets/r2b_dataset/r2b_robotarm/)
mkdir -p datasets/r2b_dataset
ln -sf ../r2bdataset2024_v1/r2b_robotarm datasets/r2b_dataset/r2b_robotarm

# Grounding DINO model (NGC filename doesn't match what the script wants — rename it)
mkdir -p models/grounding_dino
find /tmp/grounding_dino_v* -name '*.onnx' \
  -exec mv {} models/grounding_dino/grounding_dino_model.onnx \;
```

### 7. Run the RT-DETR benchmark

```bash
# Tells the benchmark scripts where to find models and datasets
export ROS2_BENCHMARK_OVERRIDE_ASSETS_ROOT=/workspaces/isaac_ros-dev/assets

launch_test $(ros2 pkg prefix isaac_ros_rtdetr_benchmark)/share/isaac_ros_rtdetr_benchmark/scripts/isaac_ros_rtdetr_graph.py
```

The first run uses `trtexec` to build a TensorRT engine (a few minutes). Subsequent runs are faster.

### 8. Run the Grounding DINO benchmark

```bash
launch_test $(ros2 pkg prefix isaac_ros_grounding_dino_benchmark)/share/isaac_ros_grounding_dino_benchmark/scripts/isaac_ros_grounding_dino_graph.py
```

The model is large (689 MB ONNX), so the first FP16 plan conversion takes 5–10 minutes.

### 9. Collect the results

JSON results are written to `/tmp/r2b-log-<datetime>.json` inside the container. They live in `/tmp`, so they're lost when the container restarts. Copy them out via the bind mount:

```bash
# Inside the container — the repo is bind-mounted, so this lands on the host's git working tree
cp /tmp/r2b-log-*.json /workspaces/isaac_ros-dev/src/amd_ros_object_detection/
```

Useful keys in the JSON:
- `BasicPerformanceMetrics.MEAN_FRAME_RATE` — peak FPS
- `30.0fps.BasicPerformanceMetrics.FIRST_SENT_RECEIVED_LATENCY` — 30 Hz latency
- `ResourceMetrics.MAX_DEVICE_UTILIZATION` — peak GPU utilization

## Common Commands

```bash
# Start the container
docker compose up -d

# Open a shell in it
docker compose exec dev bash

# Stop it
docker compose down

# Rebuild after editing the Dockerfile
docker compose build

# GPU status (inside the container)
nvidia-smi
```

## Notes

- Total dataset download is ~3.5 GB; time depends on network speed.
- NGC CLI does not support resumable downloads, but rerunning the same `download-version` command skips files that already completed.
- First benchmark run builds a TRT engine: a few minutes for RT-DETR, 5–10 minutes for Grounding DINO.
- Absolute FPS varies with GPU model. Compare against NVIDIA's published numbers on equivalent hardware where possible, and always record the GPU model alongside results.

## Pitfalls Encountered

Recorded in the order they were hit, for future reference.

### 1. NITROS doesn't run on vGPU profiles

vGPU profiles like `GRID A100X-10C` have the CUDA Memory Pool API disabled at the hypervisor level. NITROS calls `cudaDeviceGetDefaultMemPool` at startup, the call fails, the GXF context goes into the INVALID state, and downstream you see a cascade of `GXF_FACTORY_UNKNOWN_CLASS_NAME` / `Allocator type not found` errors that look like configuration problems but are really a GPU limitation. **There is no software workaround — switch to a whole GPU, MIG instance, or passthrough host.**

How to tell from `nvidia-smi`: a name with the `GRID` prefix or the `-XC` suffix (e.g. `GRID A100X-10C`) is a vGPU slice; a plain name like `NVIDIA A100-SXM4-40GB` is a whole GPU.

### 2. Driver < 580 paired with a CUDA 13 base image

The Isaac ROS 4.5 base image is CUDA 13.0. Host drivers in the 535–545 range will report a driver/CUDA version mismatch on container startup. Install `cuda-compat-13-0` inside the container (commands are in the Prerequisites section). Don't install it if your host driver is already ≥ 580: putting the compat library first in `ld.so.conf.d` would cause the runtime to use the older libraries that ship with the compat package instead of the newer host driver.

### 3. Disk space

- Isaac ROS image (extracted): ~45 GB
- r2b dataset + models: ~5 GB
- TRT engines: ~2 GB
- Reserve ~10 GB+ for colcon build artifacts later

A 60 GB root disk isn't enough; aim for at least 100 GB. Jetstream2 instances default to 60 GB — pick a larger flavor or attach a volume.

### 4. NGC dataset versioning

A common mistake: `r2b_robotarm` (used by RT-DETR and Grounding DINO) is in `r2bdataset2024:1`, not `r2bdataset2023`. Only `r2b_hallway` (used by DetectNet) is in `r2bdataset2023:2`.

### 5. NGC CLI has no resumable downloads

But rerunning the same `download-version` command does skip already-completed files, so an interrupted download only loses the file currently in flight, not everything from the start.

### 6. Grounding DINO model needs renaming

NGC delivers the model as `grounding_dino_swin_tiny_commercial_deployable.onnx`, but the benchmark script expects `grounding_dino_model.onnx`. The `find -exec` in step 6 handles this.

### 7. `NO TESTS RAN` at the end of `launch_test` is not an error

The output ends with something like:
```
Ran 1 test in 189.622s
OK
... (a wall of GXF Job Statistics tables) ...
Ran 0 tests in 0.000s
NO TESTS RAN
```
The `NO TESTS RAN` line is from `launch_test`'s post-shutdown phase, which doesn't have any registered tests. The actual benchmark result is the earlier `Ran 1 test ... OK`.

### 8. Result JSON files are lost on container restart

`/tmp/r2b-log-*.json` is not on a mounted volume. To keep results, either copy them to `/workspaces/isaac_ros-dev/src/amd_ros_object_detection/` from inside the container (bind-mounted to the host repo), or use `docker compose cp dev:/tmp/<file>.json ./` from the host.

### 9. DetectNet was deliberately skipped

Not because it doesn't run, but because its decoder is implemented as a GXF component (`gxf_isaac_detectnet.so`) rather than a NITROS subscriber wrapper. The Phase 1 ONNX Runtime migration approach used for RT-DETR/Grounding DINO doesn't transfer cleanly, so DetectNet is out of scope for the first migration pass and reproducing its baseline isn't required yet.
