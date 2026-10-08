// Copyright 2026 gpu_ros_managed contributors
// Licensed under the Apache License, Version 2.0.
#include "gpu_ros_rosidl_buffer/cuda_buffer_access.hpp"

#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

#include "cuda_buffer/cuda_buffer_api.hpp"

namespace gpu_ros::rosidl_buffer
{
namespace
{
void Check(cudaError_t status)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

class DeviceScope
{
public:
  explicit DeviceScope(int device)
  {
    Check(cudaGetDevice(&previous_));
    Check(cudaSetDevice(device));
  }
  ~DeviceScope() { static_cast<void>(cudaSetDevice(previous_)); }

private:
  int previous_{};
};

struct ConsumerStream
{
  explicit ConsumerStream(int ordinal) : device(ordinal)
  {
    if (device < 0) {
      throw std::invalid_argument("CUDA Buffer access requires a nonnegative device ordinal");
    }
    DeviceScope scope(device);
    Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
  ~ConsumerStream()
  {
    int previous = device;
    static_cast<void>(cudaGetDevice(&previous));
    static_cast<void>(cudaSetDevice(device));
    static_cast<void>(cudaStreamDestroy(stream));
    static_cast<void>(cudaSetDevice(previous));
  }

  const int device;
  cudaStream_t stream{};
  std::mutex mutex;
  bool healthy{true};
};

struct ReadOwner
{
  ReadOwner(std::shared_ptr<ConsumerStream> consumer, std::shared_ptr<const void> message)
  : stream(std::move(consumer)), message_owner(std::move(message)) {}

  ~ReadOwner()
  {
    int previous = stream->device;
    static_cast<void>(cudaGetDevice(&previous));
    static_cast<void>(cudaSetDevice(stream->device));
    // The official handle refers into the Buffer. Release it before its message,
    // and with its consumer stream still alive, on the original CUDA device.
    reader.reset();
    message_owner.reset();
    static_cast<void>(cudaSetDevice(previous));
  }

  std::shared_ptr<ConsumerStream> stream;
  std::shared_ptr<const void> message_owner;
  std::optional<cuda_buffer_backend::ReadHandle> reader;
  // Failure-only self ownership retains uncertain GPU users until process exit,
  // without allocating during exception handling.
  std::shared_ptr<ReadOwner> quarantine;
};

class CudaBufferAccess final : public IBufferAccess
{
public:
  explicit CudaBufferAccess(int device) : stream_(std::make_shared<ConsumerStream>(device)) {}

  BufferReadLease AcquireRead(
    const rosidl::Buffer<uint8_t> & buffer,
    std::shared_ptr<const void> message_owner) override
  {
    if (!message_owner) {
      throw std::invalid_argument("CUDA Buffer read requires a message owner");
    }
    const auto * impl =
      dynamic_cast<const cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(buffer.get_impl());
    // from_input_buffer promotes other backends; reject them before calling it.
    if (buffer.get_backend_type() != "cuda" || !impl) {
      throw std::invalid_argument("CUDA Buffer access requires the official cuda_buffer backend");
    }
    if (buffer.size() == 0) {
      throw std::invalid_argument("CUDA Buffer access requires nonempty storage");
    }
    if (impl->get_device_id() != stream_->device) {
      throw std::invalid_argument("Buffer storage is on the wrong CUDA device");
    }
    std::lock_guard<std::mutex> lock(stream_->mutex);
    if (!stream_->healthy) {
      throw std::runtime_error("CUDA Buffer access is unhealthy after a synchronization failure");
    }
    DeviceScope scope(stream_->device);
    auto owner = std::make_shared<ReadOwner>(stream_, std::move(message_owner));
    try {
      owner->reader.emplace(cuda_buffer_backend::from_input_buffer(buffer, stream_->stream));
      Check(cudaStreamSynchronize(stream_->stream));
    } catch (...) {
      if (cudaStreamSynchronize(stream_->stream) != cudaSuccess) {
        stream_->healthy = false;
        owner->quarantine = owner;
      }
      throw;
    }
    const auto * pointer = owner->reader->get_ptr();
    cudaPointerAttributes attributes{};
    Check(cudaPointerGetAttributes(&attributes, pointer));
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != stream_->device) {
      throw std::invalid_argument("Buffer pointer is not device storage on the requested CUDA device");
    }
    return {pointer, buffer.size(), std::move(owner)};
  }

private:
  std::shared_ptr<ConsumerStream> stream_;
};
}  // namespace

std::unique_ptr<IBufferAccess> CreateCudaBufferAccess(int device_id)
{
  return std::make_unique<CudaBufferAccess>(device_id);
}

}  // namespace gpu_ros::rosidl_buffer
