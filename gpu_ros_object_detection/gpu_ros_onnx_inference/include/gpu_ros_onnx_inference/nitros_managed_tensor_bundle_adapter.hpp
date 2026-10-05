// Copyright 2026 Boshen Chen
// Licensed under the Apache License, Version 2.0.
#ifndef GPU_ROS_ONNX_INFERENCE__NITROS_MANAGED_TENSOR_BUNDLE_ADAPTER_HPP_
#define GPU_ROS_ONNX_INFERENCE__NITROS_MANAGED_TENSOR_BUNDLE_ADAPTER_HPP_

#include "gpu_ros_nvidia_tensor_bundle_compat/tensor_list_buffer_adapter.hpp"
#include "gpu_ros_onnx_inference/tensor_types.hpp"

namespace gpu_ros::onnx_inference
{
// The installed include path is retained; this interface has no NITROS dependency.
class NativeDeviceOutputAllocator final : public DeviceOutputAllocator
{
public:
  explicit NativeDeviceOutputAllocator(
    gpu_ros::nvidia_tensor_bundle_compat::TensorListTransport & transport)
      : transport_(transport)
  {
  }
  std::unique_ptr<DeviceOutputBatch> Allocate(
    const std_msgs::msg::Header & header, const std::vector<DeviceOutputSpec> & specs) override;

private:
  gpu_ros::nvidia_tensor_bundle_compat::TensorListTransport & transport_;
};
gpu_ros_managed::ManagedTensorBundle ToManagedOutput(TensorBundleOutput && output);
} // namespace gpu_ros::onnx_inference
#endif
