# AMD Phase 2 environment

This document covers environment preparation only. It does not change the
Phase 2B managed transport implementation or add managed validation graphs.

## Shortest workflow

Docker:

```bash
./docker/phase2-amd.sh bootstrap
./docker/phase2-amd.sh shell
phase2 assets
phase2 build
```

Apptainer on a compute allocation:

```bash
OVG_RUNTIME=apptainer ./docker/phase2-amd.sh bootstrap
OVG_RUNTIME=apptainer ./docker/phase2-amd.sh shell
phase2 env
phase2 build
```

`shell`, `verify`, and `colcon` use one-shot `apptainer exec` by default.
Set `OVG_APPTAINER_INSTANCE=1` only when a user-level instance is useful for
interactive work. The instance is optional and is not the HPC default.

## Persistent state

The default host state root is `.ovg/`; set `OVG_STATE_ROOT` to put it outside
the repository. Assets, results, and cache are shared across image revisions.
Build, install, and log directories are isolated by the Docker image ID or SIF
SHA256 so an updated image cannot reuse stale CMake cache or RPATHs.

Inside every runtime the paths are:

```text
/workspaces/amd_ros_object_detection
/workspaces/amd_ros_object_detection/src/gpu_ros_managed
/workspaces/ovg-assets
/workspaces/ovg-cache
/workspaces/ovg-results
```

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

## GPU targets

Image build and runtime detection are separate. A build machine without an AMD
GPU can use an explicit target:

```bash
AMD_GPU_TARGETS=gfx950 ./docker/phase2-amd.sh build
```

If `AMD_GPU_TARGETS` is not set, the host build path attempts `rocminfo` and
fails when no target can be detected. Runtime `phase2 env --verify` runs
`rocminfo` inside the container and checks the result against the image
manifest; it never silently selects `gfx942`.

## Apptainer image

HPC nodes consume a prebuilt SIF through `OVG_APPTAINER_SIF`, or pull one time
from `OVG_APPTAINER_IMAGE_URI` when networking and the registry permit it. The
host launcher binds the application checkout, sibling checkout, assets, cache,
results, and fingerprinted build directories. It does not require Docker,
sudo, root, or fakeroot on the compute node.

`apptainer/phase2-amd.def` is a build-machine definition for converting a
validated OCI runtime image. Supply the image URI as its `OVG_AMD_IMAGE_URI`
build argument; it is not required on the HPC node.

## Scope boundary

This environment change does not add AMD managed launch files, bridges, POLs,
benchmarks, provider audits, capture, compare, or zero-copy behavior. Missing
existing validation entrypoints are reported separately and are not fixed as
part of environment setup.
