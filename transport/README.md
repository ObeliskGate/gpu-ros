# gpu_ros_managed

`gpu_ros_managed` is a small ROS 2 managed device-buffer transport layer for
CUDA and HIP. It ports the buffer/handle, fixed-pool, TensorList, TypeAdapter,
and GXF-free publisher/subscriber behavior from NVIDIA Isaac ROS NITROS 4.5
while keeping ONNX Runtime and model logic in applications.

The public core API contains no CUDA, HIP, ONNX Runtime, GXF, or `rclcpp`
headers. CUDA and HIP packages provide ordinary factories; there is no runtime
plugin registry.

## Standalone core verification

```bash
cmake -S . -B build -DGPU_ROS_MANAGED_BUILD_CUDA=OFF
cmake --build build
ctest --test-dir build --output-on-failure
```

The ROS packages are intended to be discovered as sibling packages in a
colcon workspace.
