// Copyright 2026 Boshen Chen
// Licensed under the Apache License, Version 2.0.
#include "gpu_ros_onnx_inference/nitros_managed_tensor_bundle_adapter.hpp"
#include "gpu_ros_onnx_inference/tensor_dtype.hpp"

namespace gpu_ros::onnx_inference
{
namespace
{
using namespace gpu_ros::nvidia_tensor_bundle_compat;
class NativeBatch final : public DeviceOutputBatch
{
public:
  NativeBatch(NativeOutputBatch && batch, const std_msgs::msg::Header & header)
      : header_(header), batch_(std::move(batch))
  {
  }
  const std::vector<std::shared_ptr<gpu_ros_managed::DeviceBuffer>> & buffers() const override
  {
    return batch_.buffers();
  }
  void * pointer(size_t index) const override { return batch_.pointer(index); }
  void RetainOwner(std::shared_ptr<const void> owner) override
  {
    batch_.RetainOwner(std::move(owner));
  }
  void CompleteAfterSync() override { batch_.CompleteAfterSync(); }
  void CancelBeforeSubmit() noexcept override { batch_.CancelBeforeSubmit(); }
  void FailAfterSubmit() noexcept override { batch_.FailAfterSubmit(); }
  void CopyFrom(const std::vector<OutputTensor> & outputs) override
  {
    try {
      std::vector<gpu_ros_managed::ManagedTensor> tensors;
      tensors.reserve(outputs.size());
      for (const auto & output : outputs) {
        const auto dtype =
          static_cast<gpu_ros_managed::TensorDataType>(OnnxToBundleDtype(output.dtype));
        if (const auto * device =
              std::get_if<std::shared_ptr<gpu_ros_managed::DeviceBuffer>>(&output.storage))
        {
          tensors.emplace_back(output.name, dtype, output.shape, *device);
        } else {
          throw std::invalid_argument("ORT native output fallback requires device storage");
        }
      }
      batch_.CopyFrom(gpu_ros_managed::ManagedTensorBundle(header_, std::move(tensors)));
    } catch (...) {
      // CopyFrom itself closes submitted failures with FailAfterSubmit; this
      // cancels only failures in the metadata preparation before submission.
      batch_.CancelBeforeSubmit();
      throw;
    }
  }

private:
  std_msgs::msg::Header header_;
  NativeOutputBatch batch_;
};
} // namespace
std::unique_ptr<DeviceOutputBatch> NativeDeviceOutputAllocator::Allocate(
  const std_msgs::msg::Header & header, const std::vector<DeviceOutputSpec> & specs)
{
  std::vector<gpu_ros::nvidia_tensor_bundle_compat::NativeTensorSpec> native_specs;
  native_specs.reserve(specs.size());
  for (const auto & spec : specs) {
    native_specs.push_back({spec.name,
      static_cast<gpu_ros_managed::TensorDataType>(OnnxToBundleDtype(spec.dtype)), spec.shape});
  }
  auto batch = transport_.Allocate(header, native_specs);
  try {
    return std::make_unique<NativeBatch>(std::move(batch), header);
  } catch (...) {
    batch.CancelBeforeSubmit();
    throw;
  }
}
gpu_ros_managed::ManagedTensorBundle ToManagedOutput(TensorBundleOutput && output)
{
  std::vector<gpu_ros_managed::ManagedTensor> tensors;
  tensors.reserve(output.tensors.size());
  for (auto & tensor : output.tensors) {
    const auto dtype =
      static_cast<gpu_ros_managed::TensorDataType>(OnnxToBundleDtype(tensor.dtype));
    if (auto * host = std::get_if<std::vector<uint8_t>>(&tensor.storage)) {
      auto owner = std::make_shared<std::vector<uint8_t>>(std::move(*host));
      tensors.push_back(gpu_ros_managed::ManagedTensor::from_host_external(std::move(tensor.name),
        dtype, std::move(tensor.shape), owner, owner->data(), owner->size()));
    } else {
      tensors.emplace_back(std::move(tensor.name), dtype, std::move(tensor.shape),
        std::get<std::shared_ptr<gpu_ros_managed::DeviceBuffer>>(std::move(tensor.storage)));
    }
  }
  return gpu_ros_managed::ManagedTensorBundle(std::move(output.header), std::move(tensors));
}
} // namespace gpu_ros::onnx_inference
