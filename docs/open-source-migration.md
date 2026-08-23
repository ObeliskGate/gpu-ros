# Open-source package migration

The public application namespace is `gpu_ros`. The package names below are
the supported names after the open-source preparation migration:

| Previous project package | Supported package |
| --- | --- |
| `isaac_ros_detection_common` | `gpu_ros_detection_common` |
| `isaac_ros_detection_validation` | `gpu_ros_detection_validation` |
| `isaac_ros_onnx_inference` | `gpu_ros_onnx_inference` |
| `isaac_ros_rtdetr_std` | `gpu_ros_rtdetr` |
| `isaac_ros_yolov8_std` | `gpu_ros_yolov8` |
| `gpu_ros_managed_tensor_list` | `gpu_ros_managed_tensor_bundle` |

The new interface and boundary packages are:

- `gpu_ros_tensor_bundle_msgs` — the project-owned contiguous TensorBundle
  message package.
- `gpu_ros_nvidia_tensor_bundle_compat` — an optional NVIDIA/reference-only
  adapter between the project message and NVIDIA's `TensorList` message.

There are no compatibility package aliases under the old project names. Update
`package.xml`, launch files, component plugin names, and `ros2 run` or
`ros2 launch` commands to use the supported names. The old `isaac_ros_*`
packages that remain in the source tree are external NVIDIA packages or
upstream/reference checkouts, not project-owned aliases.

The canonical AMD interface is `gpu_ros_tensor_bundle_msgs/msg/TensorBundle`.
The NVIDIA boundary is deliberately explicit: it is excluded from AMD
colcon profiles and is the only project package that depends on
`isaac_ros_tensor_list_interfaces`.

## C++ and wire API

Project-owned C++ namespaces and managed transport names follow the same
boundary:

| Previous API | Supported API |
| --- | --- |
| `nvidia::isaac_ros::onnx_inference` | `gpu_ros::onnx_inference` |
| `nvidia::isaac_ros::rtdetr_std` | `gpu_ros::rtdetr` |
| `nvidia::isaac_ros::yolov8_std` | `gpu_ros::yolov8` |
| `nvidia::isaac_ros::detection_common` | `gpu_ros::detection_common` |
| `ManagedTensorList` / `ManagedTensorListView` / `ManagedTensorListBuilder` | `ManagedTensorBundle` / `ManagedTensorBundleView` / `ManagedTensorBundleBuilder` |
| `ITensorListIO` / `TensorListIO` | `ITensorBundleIO` / `TensorBundleIO` |

`Tensor.msg` contains a project-owned dtype value, an `int64[] shape`, and
contiguous C-order `uint8[] data`. It intentionally has no NVIDIA rank,
stride, or GXF enum fields. Rank and byte-stride conversion is performed only
by `gpu_ros_nvidia_tensor_bundle_compat` when an NVIDIA reference graph crosses
the old-message boundary.

Run the source-only boundary audit with:

```bash
uv run --isolated --no-project python -B \
  tools/test_open_source_namespace_boundaries.py
```
