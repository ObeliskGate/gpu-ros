# Third-party notices

The root Apache License 2.0 applies to project-authored material in this
repository. The entries below identify file-level boundaries; they do not
relicense an upstream repository, external checkout, model, dataset, container, or
runtime.

## NVIDIA Isaac ROS object-detection sources

The opt-in `external/nvidia-isaac-ros.repos` manifest records
`isaac_ros_object_detection` at commit
`060ced887bd8a3a0be60b1fa454365942eefd128`. The following local groups are
derived from its file-level Apache-2.0 sources and retain the NVIDIA
attribution plus a notice describing the standard-ROS, HIP, or file-splitting
changes. The materially migrated decoder/preprocessor files also carry a
parallel Boshen copyright for local original additions; benchmark composition
files do not mechanically claim one:

- RT-DETR decoder, decoder node, and preprocessor sources and headers under
  `migrated_packages/gpu_ros_rtdetr/`;
- YOLOv8 decoder and decoder-node sources and headers under
  `migrated_packages/gpu_ros_yolov8/`.

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

## NVIDIA Isaac ROS benchmark sources

The opt-in external manifest records `isaac_ros_benchmark` at commit
`f46699e124262c5bfb6f00099061f6718f026b3f`. `migrated_packages/benchmarks/rtdetr_common.py`
and the RT-DETR benchmark compositions are adapted from the file-level
Apache-2.0 RT-DETR benchmark graph at that commit. The local YOLOv8 benchmark
composition has no exact upstream YOLOv8 benchmark counterpart in the pinned
checkout; it is an adaptation of the ROS 2 benchmark framework and is
documented as other-open-source-derived/project composition rather than
unresolved.

The external `isaac_ros_benchmark` checkout contains packages with different
license boundaries, including NVIDIA Isaac ROS Software License metadata. It
remains an external checkout and is not covered by the root Apache declaration.

## ONNX Runtime

The Docker/build workflow fetches ONNX Runtime `v1.23.1` at commit
`d9b2048791efb5804fe3d53a04b4971256addebf` and applies the three MIGraphX
patches under `docker/patches/`. ONNX Runtime is MIT-licensed; the applicable
copy is preserved in `LICENSES/ONNXRUNTIME-MIT.txt`. The Linux unused-helper
patch also records Microsoft upstream commit
`935affb848b1635be7d48d9c21c625300e7a3571`.

## Isaac ROS common and ros2_benchmark patches

`docker/patches/isaac-ros-common-v4.5-tensor-list-standalone.patch` now
contains only the file-level Apache-licensed CMake change. Its package
dependency change is performed by the XML-aware semantic editor against the
user's verified exact checkout; no proprietary package XML or long context is
embedded in this repository.

`docker/patches/ros2-benchmark-v4.5-standalone.patch` is a source patch for the
external benchmark checkout. Its file-level Apache source boundary and the
external package's own license metadata remain separate from this root
license.

## NVIDIA test image fixture

The following unchanged test image copies are from NVIDIA-ISAAC-ROS/isaac_ros_object_detection, commit `060ced887bd8a3a0be60b1fa454365942eefd128`, upstream path `isaac_ros_rtdetr/test/test_cases/single_detection/color_000000.jpg`:

- `migrated_packages/gpu_ros_onnx_inference/test/test_cases/single_detection/color_000000.jpg`
- `migrated_packages/gpu_ros_rtdetr/test/test_cases/single_detection/color_000000.jpg`

The fixture is distributed under Apache-2.0, with upstream attribution to NVIDIA CORPORATION & AFFILIATES. Image contents are unchanged. SHA-256: `84b21f989fca7b98ca4cfb902346401a0c2d917cff2fbf4f207bad98b30bbdd1`.

Source and license: https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_object_detection/blob/060ced887bd8a3a0be60b1fa454365942eefd128/isaac_ros_rtdetr/test/test_cases/single_detection/color_000000.jpg and https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_object_detection/blob/060ced887bd8a3a0be60b1fa454365942eefd128/LICENSE .

## SyntheticaDETR external model

SyntheticaDETR 1.0.0_onnx is an external asset whose published metadata points to the NVIDIA Deep Learning Models License (August 10, 2021); section 1.1 limits applications to systems with NVIDIA GPUs. This source release does not grant model rights or establish permission for historical ORT/AMD uses.

Model metadata: https://api.ngc.nvidia.com/v2/models/nvidia/isaac/synthetica_detr/versions/1.0.0_onnx . Applicable published license: https://developer.download.nvidia.com/licenses/tao_toolkit_21-08_models_eula.pdf .

## ROS, CUDA, HIP, and external assets

ROS 2, OpenCV, CUDA, ROCm/HIP, MIGraphX, and their libraries are external
dependencies under their own terms. Except for the Apache-2.0 test image fixture identified above, models, datasets, NGC downloads, user ONNX files, TensorRT/MIGraphX engines, bags, traces, profiles, logs, and SIF images are external runtime assets and are not distributed by this source tree. Source compatibility hashes do not grant asset redistribution rights.
