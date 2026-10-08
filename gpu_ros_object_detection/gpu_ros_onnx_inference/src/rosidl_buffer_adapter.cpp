// Copyright 2026 Boshen Chen
// Licensed under the Apache License, Version 2.0.
#include "gpu_ros_onnx_inference/rosidl_buffer_adapter.hpp"
#include "gpu_ros_onnx_inference/tensor_contract.hpp"
#include "gpu_ros_onnx_inference/tensor_dtype.hpp"
#include <cstring>
#include <utility>
#include <stdexcept>
#ifdef GPU_ROS_ROSIDL_CUDA
#include "gpu_ros_rosidl_buffer/cuda_buffer_access.hpp"
#include "cuda_buffer/cuda_buffer_api.hpp"
#include "cuda_buffer/cuda_buffer_impl.hpp"
#include <cuda_runtime_api.h>
#endif
namespace gpu_ros::onnx_inference
{
Ort::MemoryInfo ProviderMemoryInfo(ExecutionProvider provider, int device)
{
  if (device < 0) { throw std::invalid_argument("negative device ordinal"); }
  if (provider == ExecutionProvider::kCpu) {
    return Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  }
  return Ort::MemoryInfo(provider == ExecutionProvider::kRocm ? "Rocm" : "Cuda",
    OrtMemoryInfoDeviceType_GPU, provider == ExecutionProvider::kCuda ? 0x10de : 0x1002,
    static_cast<uint32_t>(device), OrtDeviceMemoryType_DEFAULT, 0, OrtDeviceAllocator);
}
TensorBinding BindBuffer(const rosidl::Buffer<uint8_t> & buffer,
  const std::string & name, ONNXTensorElementDataType dtype, const std::vector<int64_t> & shape,
  size_t offset, const std::vector<int64_t> & strides, std::shared_ptr<const void> owner,
  gpu_ros::rosidl_buffer::IBufferAccess & access, const Ort::MemoryInfo & memory)
{
  const bool host = buffer.get_backend_type() == "cpu";
  if ((host && memory.GetDeviceType() != OrtMemoryInfoDeviceType_CPU) ||
    (!host && (buffer.get_backend_type() != "cuda" ||
      memory.GetDeviceType() != OrtMemoryInfoDeviceType_GPU || memory.GetVendorId() != 0x10deU)))
  {
    throw std::invalid_argument("Buffer backend and ORT memory identity differ");
  }
  const auto bytes = TensorByteSize(dtype, shape, name);
  const auto element = TensorElementSize(dtype);
  if (offset % element || offset > buffer.size() || bytes > buffer.size() - offset) {
    throw std::invalid_argument("Buffer offset or capacity does not cover tensor: " + name);
  }
  if (!strides.empty()) {
    if (strides.size() != shape.size()) { throw std::invalid_argument("stride rank mismatch"); }
    size_t expected = 1;
    for (size_t d = shape.size(); d-- > 0;) {
      if (strides[d] < 0 || static_cast<size_t>(strides[d]) != expected) {
        throw std::invalid_argument("noncontiguous Buffer tensor strides are unsupported");
      }
      expected *= static_cast<size_t>(shape[d]);
    }
  }
  auto lease = access.AcquireRead(buffer, std::move(owner));
  if (!lease.owner || !lease.data || offset > lease.byte_count || bytes > lease.byte_count - offset) {
    throw std::invalid_argument("Buffer read lease does not cover tensor");
  }
  const auto * pointer = static_cast<const uint8_t *>(lease.data) + offset;
  if (reinterpret_cast<uintptr_t>(pointer) % element) {
    throw std::invalid_argument("misaligned Buffer tensor pointer");
  }
  const bool device = memory.GetDeviceType() == OrtMemoryInfoDeviceType_GPU;
  return {name, std::move(lease.owner), pointer, bytes,
    device ? "cuda_device" : "host", "original Buffer message and backend read lease through synchronization",
    Ort::Value::CreateTensor(memory, const_cast<uint8_t *>(pointer), bytes,
      shape.data(), shape.size(), dtype)};
}
TensorBindingBatch BindTensorBundle(
  gpu_ros_tensor_bundle_msgs::msg::TensorBundle::ConstSharedPtr message,
  ExecutionProvider provider, int device, gpu_ros::rosidl_buffer::IBufferAccess & cpu,
  gpu_ros::rosidl_buffer::IBufferAccess * cuda)
{
  if (!message) { throw std::invalid_argument("null TensorBundle message"); }
  TensorBindingBatch batch{message->header, message, {}};
  batch.bindings.reserve(message->tensors.size());
  for (const auto & tensor : message->tensors) {
    const bool host = tensor.data.get_backend_type() == "cpu";
    auto * access = &cpu;
    if (!host) {
      if (provider != ExecutionProvider::kCuda || tensor.data.get_backend_type() != "cuda") {
        throw std::invalid_argument("unsupported Buffer backend for execution provider");
      }
      if (!cuda) { throw std::invalid_argument("CUDA Buffer access is unavailable"); }
      access = cuda;
    }
    auto memory = ProviderMemoryInfo(host ? ExecutionProvider::kCpu : provider, device);
    batch.bindings.push_back(BindBuffer(tensor.data, tensor.name, BundleToOnnxDtype(tensor.data_type),
      tensor.shape, 0, {}, message, *access, memory));
  }
  return batch;
}
TensorBindingBatch BindTensorBundle(
  gpu_ros_tensor_bundle_msgs::msg::TensorBundle::ConstSharedPtr message,
  ExecutionProvider provider, int device)
{
  auto cpu = gpu_ros::rosidl_buffer::CreateCpuBufferAccess();
  std::unique_ptr<gpu_ros::rosidl_buffer::IBufferAccess> cuda;
#ifdef GPU_ROS_ROSIDL_CUDA
  if (provider == ExecutionProvider::kCuda) {
    cuda = gpu_ros::rosidl_buffer::CreateCudaBufferAccess(device);
  }
#endif
  return BindTensorBundle(std::move(message), provider, device, *cpu, cuda.get());
}
namespace
{
using Message = gpu_ros_tensor_bundle_msgs::msg::TensorBundle;
#ifdef GPU_ROS_ROSIDL_CUDA
void Check(cudaError_t status)
{
  if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
}
struct Stream
{
  explicit Stream(int ordinal) : device(ordinal)
  {
    Check(cudaSetDevice(device));
    Check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking));
  }
  ~Stream() { cudaSetDevice(device); cudaStreamDestroy(value); }
  int device;
  cudaStream_t value{};
};
#endif
struct BufferState
{
#ifdef GPU_ROS_ROSIDL_CUDA
  std::shared_ptr<Stream> stream;
#endif
  std::shared_ptr<Message> message;
  std::shared_ptr<const void> retained;
  std::vector<void *> pointers;
#ifdef GPU_ROS_ROSIDL_CUDA
  std::vector<cuda_buffer_backend::WriteHandle> writers;
#endif
  std::shared_ptr<BufferState> quarantine;
  bool terminal{false};
};
class BufferBatch final : public DeviceOutputBatch
{
public:
  BufferBatch(const std_msgs::msg::Header & header, const std::vector<DeviceOutputSpec> & specs,
    ExecutionProvider provider, int device) : state_(std::make_shared<BufferState>()),
    device_(provider == ExecutionProvider::kCuda)
  {
#ifdef GPU_ROS_ROSIDL_CUDA
    if (device_) { state_->stream = std::make_shared<Stream>(device); }
#else
    (void)device;
#endif
    state_->message = std::make_shared<Message>();
    state_->message->header = header;
    state_->message->tensors.reserve(specs.size());
    state_->pointers.reserve(specs.size());
    try {
      for (const auto & spec : specs) {
        auto & tensor = state_->message->tensors.emplace_back();
        tensor.name = spec.name;
        tensor.data_type = OnnxToBundleDtype(spec.dtype);
        tensor.shape = spec.shape;
        const auto bytes = TensorByteSize(spec.dtype, spec.shape, spec.name);
#ifdef GPU_ROS_ROSIDL_CUDA
        if (device_) {
          tensor.data = cuda_buffer_backend::allocate_buffer(bytes);
          state_->writers.push_back(cuda_buffer_backend::from_output_buffer(tensor.data, state_->stream->value));
          state_->pointers.push_back(state_->writers.back().get_ptr());
        } else
#endif
        {
          tensor.data = std::vector<uint8_t>(bytes);
          state_->pointers.push_back(tensor.data.data());
        }
      }
    } catch (...) { CancelBeforeSubmit(); throw; }
  }
  ~BufferBatch() override { if (!state_->terminal) { FailAfterSubmit(); } }
  size_t size() const noexcept override { return state_->pointers.size(); }
  void * pointer(size_t i) const override { return state_->pointers.at(i); }
  TensorStorage storage(size_t i) const override
  {
    return {state_, state_->message, pointer(i), state_->message->tensors.at(i).data.size(),
      device_ ? "cuda_device" : "host"};
  }
  void RetainOwner(std::shared_ptr<const void> owner) override { state_->retained = std::move(owner); }
  void CompleteAfterSync() override
  {
    if (state_->terminal) { throw std::logic_error("Buffer transaction already terminal"); }
#ifdef GPU_ROS_ROSIDL_CUDA
    if (device_) {
      Check(cudaSetDevice(state_->stream->device));
      state_->writers.clear();
      Check(cudaStreamSynchronize(state_->stream->value));
      // The official Buffer's clone/to_cpu paths use its stored stream. Once
      // the writer event is complete, detach this borrowed producer stream so
      // a moved ROS message remains serializable after the allocator dies.
      for (auto & tensor : state_->message->tensors) {
        auto * impl = dynamic_cast<cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(tensor.data.get_impl());
        if (!impl) { throw std::logic_error("output lost official CUDA backend"); }
        impl->set_stream(nullptr);
      }
    }
#endif
    state_->terminal = true;
    state_->retained.reset();
  }
  void CancelBeforeSubmit() noexcept override
  {
    if (state_->terminal) { return; }
#ifdef GPU_ROS_ROSIDL_CUDA
    if (device_) {
      if (cudaSetDevice(state_->stream->device) != cudaSuccess) { FailAfterSubmit(); return; }
      state_->writers.clear();
      if (cudaStreamSynchronize(state_->stream->value) != cudaSuccess) { FailAfterSubmit(); return; }
    }
#endif
    state_->terminal = true;
    state_->retained.reset();
  }
  void FailAfterSubmit() noexcept override
  {
    if (!state_->terminal) { state_->quarantine = state_; state_->terminal = true; }
  }
  void CopyFrom(const std::vector<OutputTensor> & tensors) override
  {
    if (tensors.size() != size()) { throw std::invalid_argument("Buffer copy count mismatch"); }
    auto sources = std::make_shared<std::vector<TensorStorage>>();
    sources->reserve(tensors.size());
    for (const auto & tensor : tensors) { sources->push_back(std::get<TensorStorage>(tensor.storage)); }
    state_->retained = sources;
    try {
      for (size_t i = 0; i < tensors.size(); ++i) {
        const auto & source = sources->at(i);
        if (source.byte_count != storage(i).byte_count) { throw std::invalid_argument("Buffer copy byte mismatch"); }
#ifdef GPU_ROS_ROSIDL_CUDA
        if (device_) {
          Check(cudaSetDevice(state_->stream->device));
          Check(cudaMemcpyAsync(pointer(i), source.data, source.byte_count,
            cudaMemcpyDeviceToDevice, state_->stream->value));
        } else
#endif
        { std::memcpy(pointer(i), source.data, source.byte_count); }
      }
      CompleteAfterSync();
    } catch (...) { FailAfterSubmit(); throw; }
  }
private:
  std::shared_ptr<BufferState> state_;
  bool device_;
};
class BufferAllocator final : public DeviceOutputAllocator
{
public:
  BufferAllocator(ExecutionProvider provider, int device) : provider_(provider), device_(device) {}
  std::unique_ptr<DeviceOutputBatch> Allocate(const std_msgs::msg::Header & header,
    const std::vector<DeviceOutputSpec> & specs) override
  { return std::make_unique<BufferBatch>(header, specs, provider_, device_); }
  Ort::MemoryInfo memory_info() const override { return ProviderMemoryInfo(provider_, device_); }
  bool SynchronizeAfterFailure() noexcept override
  {
    if (provider_ == ExecutionProvider::kCpu) { return true; }
#ifdef GPU_ROS_ROSIDL_CUDA
    return cudaSetDevice(device_) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
#else
    return false;
#endif
  }
private:
  ExecutionProvider provider_;
  int device_;
};
} // namespace
std::unique_ptr<DeviceOutputAllocator> CreateBufferOutputAllocator(ExecutionProvider provider, int device)
{
  if (provider != ExecutionProvider::kCpu && provider != ExecutionProvider::kCuda) {
    throw std::invalid_argument("only CPU and CUDA Buffer backends are implemented");
  }
#ifndef GPU_ROS_ROSIDL_CUDA
  if (provider == ExecutionProvider::kCuda) { throw std::invalid_argument("CUDA Buffer support is not built"); }
#endif
  if (device < 0) { throw std::invalid_argument("negative device ordinal"); }
  return std::make_unique<BufferAllocator>(provider, device);
}
std::shared_ptr<Message> BufferOutputMessage(const std::vector<OutputTensor> & output)
{
  if (output.empty()) { throw std::invalid_argument("empty Buffer output batch"); }
  auto message = std::static_pointer_cast<Message>(std::get<TensorStorage>(output.front().storage).envelope);
  if (!message || message->tensors.size() != output.size()) { throw std::invalid_argument("invalid Buffer envelope"); }
  for (const auto & tensor : output) {
    if (std::get<TensorStorage>(tensor.storage).envelope != message) {
      throw std::invalid_argument("mixed Buffer output envelopes");
    }
  }
  return message;
}
} // namespace gpu_ros::onnx_inference
