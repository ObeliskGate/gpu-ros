// Copyright 2026 Boshen Chen
// Licensed under the Apache License, Version 2.0.
#include "gpu_ros_onnx_inference/tensor_bundle_io.hpp"
#include "gpu_ros_onnx_inference/rosidl_buffer_adapter.hpp"
#include "gpu_ros_onnx_inference/tensor_contract.hpp"
#include "gpu_ros_nvidia_tensor_bundle_compat/native_tensor_list.hpp"
#include "gpu_ros_rosidl_buffer/cuda_buffer_access.hpp"
#include <cuda_runtime_api.h>
namespace gpu_ros::onnx_inference
{
namespace
{
namespace wire = gpu_ros::nvidia_tensor_bundle_compat::native;
ONNXTensorElementDataType Dtype(const wire::TensorSpec & spec)
{
  if (spec.dtype_lanes == 1 && spec.dtype_code == 2 && spec.dtype_bits == 32) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
  if (spec.dtype_lanes == 1 && spec.dtype_code == 0 && spec.dtype_bits == 64) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  }
  throw std::invalid_argument("TensorList inference supports float32 or int64");
}
class WireBatch final : public DeviceOutputBatch
{
public:
  explicit WireBatch(wire::OutputBatch && batch) : batch_(std::make_shared<wire::OutputBatch>(std::move(batch))) {}
  size_t size() const noexcept override { return batch_->tensors().size(); }
  void * pointer(size_t i) const override { return batch_->data(i); }
  TensorStorage storage(size_t i) const override
  {
    const auto & spec = batch_->tensors().at(i);
    return {batch_, batch_, pointer(i), TensorByteSize(Dtype(spec), spec.shape, spec.name), "cuda_device"};
  }
  void RetainOwner(std::shared_ptr<const void> owner) override { batch_->RetainOwner(std::move(owner)); }
  void CompleteAfterSync() override { batch_->CompleteAfterSync(); }
  void CancelBeforeSubmit() noexcept override { batch_->CancelBeforeSubmit(); }
  void FailAfterSubmit() noexcept override { batch_->FailAfterSubmit(); }
  void CopyFrom(const std::vector<OutputTensor> & output) override
  {
    auto owners = std::make_shared<std::vector<TensorStorage>>();
    std::vector<wire::CopySource> sources;
    for (const auto & tensor : output) {
      owners->push_back(std::get<TensorStorage>(tensor.storage));
      const auto & source = owners->back();
      sources.push_back({source.data, source.byte_count, true});
    }
    batch_->CopyFrom(sources, owners);
  }
private:
  std::shared_ptr<wire::OutputBatch> batch_;
};
class WireIO final : public ITensorBundleIO, public DeviceOutputAllocator
{
public:
  WireIO(rclcpp::Node * node, bool publish) :
    device_(static_cast<int>(node->get_parameter("gpu_device_id").as_int())),
    transport_(node, device_, "tensor_output", publish),
    access_(gpu_ros::rosidl_buffer::CreateCudaBufferAccess(device_))
  {
    if (node->get_parameter("execution_provider").as_string() != "cuda") {
      throw std::invalid_argument("TensorList wire support requires CUDA execution provider");
    }
  }
  ~WireIO() override { transport_.Unsubscribe(); }
  void Subscribe(Callback callback) override
  {
    transport_.SubscribeBuffers([this, callback = std::move(callback)](wire::BufferTensorList input) {
      TensorBindingBatch batch{std::move(input.header), input.owner, {}};
      batch.bindings.reserve(input.tensors.size());
      const auto memory = memory_info();
      for (const auto & tensor : input.tensors) {
        batch.bindings.push_back(BindBuffer(*tensor.buffer, tensor.spec.name, Dtype(tensor.spec),
          tensor.spec.shape, tensor.byte_offset, tensor.strides, input.owner, *access_, memory));
      }
      callback(std::move(batch));
    });
  }
  OutputPlacement output_placement() const noexcept override { return OutputPlacement::kDevice; }
  DeviceOutputAllocator * device_output_allocator() noexcept override { return this; }
  void Publish(TensorBundleOutput && output) override
  {
    if (output.tensors.empty()) { throw std::invalid_argument("empty TensorList output"); }
    auto batch = std::static_pointer_cast<wire::OutputBatch>(std::get<TensorStorage>(output.tensors[0].storage).envelope);
    if (!batch) { throw std::invalid_argument("missing TensorList output envelope"); }
    transport_.Publish(batch->message());
  }
  std::unique_ptr<DeviceOutputBatch> Allocate(const std_msgs::msg::Header & header,
    const std::vector<DeviceOutputSpec> & specs) override
  {
    std::vector<wire::TensorSpec> wire_specs;
    for (const auto & spec : specs) {
      if (spec.dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && spec.dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        throw std::invalid_argument("unsupported TensorList output dtype");
      }
      wire_specs.push_back({spec.name, static_cast<uint8_t>(spec.dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? 2 : 0),
        static_cast<uint8_t>(spec.dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? 32 : 64), 1, spec.shape});
    }
    auto batch = transport_.Allocate(header, wire_specs);
    try { return std::make_unique<WireBatch>(std::move(batch)); }
    catch (...) { batch.CancelBeforeSubmit(); throw; }
  }
  Ort::MemoryInfo memory_info() const override { return ProviderMemoryInfo(ExecutionProvider::kCuda, device_); }
  bool SynchronizeAfterFailure() noexcept override
  { return cudaSetDevice(device_) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess; }
  bool shutdown(std::chrono::milliseconds) noexcept override { return true; }
private:
  int device_;
  wire::TensorListTransport transport_;
  std::unique_ptr<gpu_ros::rosidl_buffer::IBufferAccess> access_;
};
} // namespace
std::unique_ptr<ITensorBundleIO> CreateTensorListBufferIO(rclcpp::Node * node, bool publish)
{ return std::make_unique<WireIO>(node, publish); }
} // namespace gpu_ros::onnx_inference
