# Third-party notices

The root Apache License 2.0 applies to project-authored material in this
repository. The entries below identify file-level boundaries; they do not
relicense an upstream repository, external checkout, model, dataset, container, or
runtime.

## NVIDIA Isaac ROS NITROS

The following local files are derived from file-level Apache-2.0 source in
NVIDIA Isaac ROS NITROS `v4.5-0`, commit
`82310fce298d3d9db26945a3a988d5c471d14973`:

- `gpu_ros_managed/gpu_ros_managed_core/include/gpu_ros_managed_core/buffer.hpp`
- `gpu_ros_managed/gpu_ros_managed_core/src/buffer.cpp`
- `gpu_ros_managed/gpu_ros_managed_core/include/gpu_ros_managed_core/fixed_device_memory_pool.hpp`
- `gpu_ros_managed/gpu_ros_managed_core/src/fixed_device_memory_pool.cpp`
- `gpu_ros_managed/gpu_ros_managed_tensor_bundle/include/gpu_ros_managed_tensor_bundle/tensor_bundle.hpp`
- `gpu_ros_managed/gpu_ros_managed_tensor_bundle/src/tensor_bundle.cpp`
- `gpu_ros_managed/gpu_ros_managed_ros/include/gpu_ros_managed_ros/managed_pub_sub.hpp`

The exact upstream file attribution is retained in each derived file, and the
files contain a notice describing the local backend-neutral or GXF-free
modification. NITROS repository/package metadata is not treated as a blanket
Apache license for all of NITROS.

The managed package uses the independent `gpu_ros_tensor_bundle_msgs` message
package at `gpu_ros_managed/gpu_ros_tensor_bundle_msgs` in this monorepo. Its
TypeAdapter is project-authored and does not depend on NVIDIA's `TensorList`
message package; the NVIDIA conversion boundary lives in the monorepo's
`gpu_ros_managed/gpu_ros_nvidia_tensor_bundle_compat` package.

## NVIDIA Isaac ROS object-detection sources

The following local groups were derived from file-level Apache-2.0 sources in
`isaac_ros_object_detection` commit
`060ced887bd8a3a0be60b1fa454365942eefd128`. This records historical source
provenance, not an active Isaac ROS 5.0 dependency pin. The files retain the
NVIDIA attribution plus a notice describing the standard-ROS, HIP, or
file-splitting changes. The materially migrated decoder/preprocessor files also carry a
parallel Boshen copyright for local original additions; benchmark composition
files do not mechanically claim one:

- RT-DETR decoder, decoder node, and preprocessor sources and headers under
  `gpu_ros_object_detection/gpu_ros_rtdetr/`;
- YOLOv8 decoder and decoder-node sources and headers under
  `gpu_ros_object_detection/gpu_ros_yolov8/`.

The local image encoders, Managed HIP nodes, and ONNX Runtime integration were
reviewed as project-original application code where no exact one-to-one
upstream file was found. They use upstream APIs but are not represented as
copies of a proprietary implementation.

## TensorBundle and NVIDIA compatibility boundary

`gpu_ros_tensor_bundle_msgs` is an independent Apache-2.0 interface package.
Its dtype values, `int64[]` shape field, and contiguous payload schema are
project-owned; it does not reproduce NVIDIA's `TensorList`, `TensorShape`,
GXF enum, or generated package metadata.

`gpu_ros_nvidia_tensor_bundle_compat` is a project-authored Apache-2.0
reference adapter. It is the only local package that depends on the external
`isaac_ros_tensor_list_interfaces` message package. That dependency remains
inside an explicit NVIDIA-only boundary, where the adapter validates and
converts rank, shape, and byte strides. AMD profiles do not build, install, or
resolve this package. The external NVIDIA message package and its license
remain authoritative for that side of the boundary; this project does not
claim NVIDIA sponsorship or endorsement.

## Project-original code

The backend interface and implementations, host-buffer utilities, ROS
TypeAdapter, tests, and build files are maintained as project-original code
unless a source file says otherwise. They are distributed under the Apache
License 2.0 in `LICENSE`.

## NVIDIA Isaac ROS benchmark sources

`gpu_ros_object_detection/benchmarks/rtdetr_common.py` and the RT-DETR benchmark
compositions were adapted from the file-level Apache-2.0 RT-DETR benchmark
graph in `isaac_ros_benchmark` commit
`f46699e124262c5bfb6f00099061f6718f026b3f`. This records source provenance, not
the active Isaac ROS 5.0 runtime version. The local YOLOv8 benchmark
composition has no exact upstream YOLOv8 benchmark counterpart in that
historical checkout; it is an adaptation of the ROS 2 benchmark framework and
is documented as other-open-source-derived/project composition.

The external `isaac_ros_benchmark` checkout contains packages with different
license boundaries, including NVIDIA Isaac ROS Software License metadata. It
remains an external checkout and is not covered by the root Apache declaration.

## ONNX Runtime

The Docker/build workflow selects ONNX Runtime `v1.30.0` at commit
`f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7`. AMD builds apply the three
MIGraphX policy patches under `gpu_ros_object_detection/docker/patches/`:
GridSample capability, rank-at-least-two int64 Div CPU fallback, and Linux
provider residency. The unused-helper backport is no longer needed upstream.
NVIDIA installs the complete official CUDA 13 distribution at the archive
hash in `gpu_ros_object_detection/config/onnxruntime.lock`, including its license
files. ONNX Runtime is MIT-licensed; the source notice is also preserved in
`LICENSES/ONNXRUNTIME-MIT.txt`.

The NVIDIA image builds only `cuda_buffer_backend` from released
`ros2/rosidl_buffer_backends` 0.1.2 at the source revision in
`gpu_ros_object_detection/config/nvidia-runtime.lock`. Its upstream license is
installed with the plugin; SDK CUDA Buffer, message, and ROS core packages
remain external binary dependencies.

## Standalone ros2_benchmark preparation

The old Isaac ROS 4.5 TensorList patch, its package editor, and the 4.5
benchmark patch have been removed from the active build.
`gpu_ros_object_detection/docker/prepare_ros2_benchmark_standalone.py`
adapts the selected image-paired external benchmark checkout's
`ros2_benchmark/CMakeLists.txt`, `ros2_benchmark/package.xml`,
`ros2_benchmark_interfaces/CMakeLists.txt`, and
`ros2_benchmark_interfaces/package.xml`. It removes only the standalone
build's version-stamp and `isaac_ros_common` build dependency, generating the
patch from the verified external revision rather than embedding package XML.
Upstream file-level notices and the external packages' license metadata
remain authoritative and separate from this repository's root license.

## NVIDIA test image fixture

The following unchanged test image copies are from NVIDIA-ISAAC-ROS/isaac_ros_object_detection, commit `060ced887bd8a3a0be60b1fa454365942eefd128`, upstream path `isaac_ros_rtdetr/test/test_cases/single_detection/color_000000.jpg`:

- `gpu_ros_object_detection/gpu_ros_onnx_inference/test/test_cases/single_detection/color_000000.jpg`
- `gpu_ros_object_detection/gpu_ros_rtdetr/test/test_cases/single_detection/color_000000.jpg`

The fixture is distributed under Apache-2.0, with upstream attribution to NVIDIA CORPORATION & AFFILIATES. Image contents are unchanged. SHA-256: `84b21f989fca7b98ca4cfb902346401a0c2d917cff2fbf4f207bad98b30bbdd1`.

Source and license: https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_object_detection/blob/060ced887bd8a3a0be60b1fa454365942eefd128/isaac_ros_rtdetr/test/test_cases/single_detection/color_000000.jpg and https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_object_detection/blob/060ced887bd8a3a0be60b1fa454365942eefd128/LICENSE .

## SyntheticaDETR external model

SyntheticaDETR 1.0.0_onnx is an external asset whose published metadata points to the NVIDIA Deep Learning Models License (August 10, 2021); section 1.1 limits applications to systems with NVIDIA GPUs. This source release does not grant model rights or establish permission for historical ORT/AMD uses.

Model metadata: https://api.ngc.nvidia.com/v2/models/nvidia/isaac/synthetica_detr/versions/1.0.0_onnx . Applicable published license: https://developer.download.nvidia.com/licenses/tao_toolkit_21-08_models_eula.pdf .

## ROS, CUDA, HIP, and external assets

ROS 2 packages, OpenCV, CUDA, ROCm/HIP, MIGraphX, and their headers and
libraries are external build/runtime dependencies. This repository does not
vendor those SDKs or relicense them. Consumers must follow the license terms
supplied by their chosen ROS distribution, CUDA toolkit, and ROCm installation.

No model, dataset, bag, trace, profile, log, SIF image, or benchmark result is
required or distributed by this source tree. Except for the Apache-2.0 test
image fixture identified above, models, datasets, NGC downloads, user ONNX
files, TensorRT/MIGraphX engines, bags, traces, profiles, logs, and SIF images
are external runtime assets and are not distributed by this source tree. An
application may use external assets subject to the asset provider's terms;
source-code license and model or dataset license are separate questions.
Source compatibility hashes do not grant asset redistribution rights.
