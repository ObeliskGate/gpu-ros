// Copyright 2026 Boshen Chen
// Licensed under the Apache License, Version 2.0.
#include "gpu_ros_managed_cuda/cuda_backend.hpp"

#include <sstream>
#include <stdexcept>
#include <utility>

#include "gpu_ros_managed_core/detail/backend_ops.hpp"

namespace gpu_ros_managed::cuda
{
namespace
{
void check(cudaError_t result, const char * operation)
{
  if (result == cudaSuccess) {return;}
  std::ostringstream message;
  message << operation << ": " << cudaGetErrorName(result) << " (" <<
    cudaGetErrorString(result) << ")";
  throw std::runtime_error(message.str());
}

class CudaOps final : public detail::BackendOps
{
public:
  BackendKind kind() const noexcept override {return BackendKind::kCuda;}
  void select_device(int ordinal) override {check(cudaSetDevice(ordinal), "cudaSetDevice");}
  detail::Event create_event() override
  {
    cudaEvent_t event{};
    check(cudaEventCreateWithFlags(
        &event, cudaEventDisableTiming | cudaEventBlockingSync), "cudaEventCreateWithFlags");
    return reinterpret_cast<detail::Event>(event);
  }
  void record_event(detail::Event event, detail::NativeStream stream) override
  {
    check(cudaEventRecord(
        reinterpret_cast<cudaEvent_t>(event), reinterpret_cast<cudaStream_t>(stream)),
      "cudaEventRecord");
  }
  void wait_event(detail::NativeStream stream, detail::Event event) override
  {
    check(cudaStreamWaitEvent(
        reinterpret_cast<cudaStream_t>(stream), reinterpret_cast<cudaEvent_t>(event), 0),
      "cudaStreamWaitEvent");
  }
  void synchronize_event(detail::Event event) override
  {
    check(cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(event)), "cudaEventSynchronize");
  }
  void destroy_event(detail::Event event) noexcept override
  {
    if (event != 0) {static_cast<void>(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event)));}
  }
  std::shared_ptr<void> allocate_device(int ordinal, size_t bytes) override
  {
    select_device(ordinal);
    void * pointer{};
    check(cudaMalloc(&pointer, bytes), "cudaMalloc");
    return std::shared_ptr<void>(pointer, [ordinal](void * value) {
      if (value == nullptr) {return;}
      static_cast<void>(cudaSetDevice(ordinal));
      static_cast<void>(cudaFree(value));
    });
  }
  void copy_host_to_device(
    int ordinal, void * destination, const void * source, size_t bytes) override
  {
    select_device(ordinal);
    check(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
  }
  void copy_device_to_host(
    int ordinal, void * destination, const void * source, size_t bytes) override
  {
    select_device(ordinal);
    check(cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
  }
};

std::shared_ptr<detail::BackendOps> ops()
{
  static auto value = std::make_shared<CudaOps>();
  return value;
}
}  // namespace

cudaStream_t CudaStream::get() const
{
  const auto & state = detail::StreamAccess::get(stream_);
  return reinterpret_cast<cudaStream_t>(state.native);
}

CudaStream make_stream(int device_id)
{
  ops()->select_device(device_id);
  cudaStream_t native{};
  check(cudaStreamCreateWithFlags(&native, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
  auto owner = std::shared_ptr<void>(
    reinterpret_cast<void *>(native), [device_id](void * value) {
      static_cast<void>(cudaSetDevice(device_id));
      static_cast<void>(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(value)));
    });
  return CudaStream(detail::DeviceBufferFactory::make_stream(
      {BackendKind::kCuda, device_id}, reinterpret_cast<detail::NativeStream>(native),
      std::move(owner), ops()));
}
CudaStream wrap_borrowed_stream(
  cudaStream_t native, int device_id, std::shared_ptr<void> owner)
{
  if (native == nullptr) {throw std::invalid_argument("Borrowed CUDA stream is null");}
  return CudaStream(detail::DeviceBufferFactory::make_stream(
      {BackendKind::kCuda, device_id}, reinterpret_cast<detail::NativeStream>(native),
      std::move(owner), ops()));
}
std::shared_ptr<DeviceBuffer> allocate(size_t bytes, int device_id)
{
  auto owner = ops()->allocate_device(device_id, bytes);
  // Function argument evaluation order is not guaranteed.  Capture the
  // allocation before moving its owner into the DeviceBuffer.
  void * pointer = owner.get();
  return detail::DeviceBufferFactory::make_fresh(
    {BackendKind::kCuda, device_id}, pointer, bytes, std::move(owner), ops());
}
std::shared_ptr<DeviceBuffer> adopt_external(
  void * pointer, size_t bytes, int device_id, std::shared_ptr<void> owner)
{
  return detail::DeviceBufferFactory::make_fresh(
    {BackendKind::kCuda, device_id}, pointer, bytes, std::move(owner), ops());
}
std::shared_ptr<DeviceBuffer> adopt_synchronized_external(
  void * pointer, size_t bytes, int device_id, std::shared_ptr<void> owner)
{
  return detail::DeviceBufferFactory::make_ready(
    {BackendKind::kCuda, device_id}, pointer, bytes, std::move(owner), ops());
}
FixedDeviceMemoryPool make_fixed_device_pool(
  size_t block_size, size_t block_count, int device_id)
{
  return detail::PoolFactory::make(
    {BackendKind::kCuda, device_id}, block_size, block_count, ops());
}
bool wait_for_pending_releases(std::chrono::milliseconds timeout)
{
  return detail::wait_for_pending_releases(BackendKind::kCuda, timeout);
}
}  // namespace gpu_ros_managed::cuda
