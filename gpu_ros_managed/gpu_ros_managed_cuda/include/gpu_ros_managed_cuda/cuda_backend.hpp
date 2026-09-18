// Copyright 2026 Boshen Chen
// Licensed under the Apache License, Version 2.0.
#ifndef GPU_ROS_MANAGED_CUDA__CUDA_BACKEND_HPP_
#define GPU_ROS_MANAGED_CUDA__CUDA_BACKEND_HPP_

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>
#include <memory>

#include "gpu_ros_managed_core/buffer.hpp"
#include "gpu_ros_managed_core/fixed_device_memory_pool.hpp"

namespace gpu_ros_managed::cuda
{
class CudaStream
{
public:
  const DeviceStream & stream() const noexcept { return stream_; }
  cudaStream_t get() const;
  operator const DeviceStream &() const noexcept { return stream_; }

private:
  explicit CudaStream(DeviceStream stream) : stream_(std::move(stream)) {}
  DeviceStream stream_;
  friend CudaStream make_stream(int);
  friend CudaStream wrap_borrowed_stream(cudaStream_t, int, std::shared_ptr<void>);
};

CudaStream make_stream(int device_id);
CudaStream wrap_borrowed_stream(
  cudaStream_t stream, int device_id, std::shared_ptr<void> owner = {});
std::shared_ptr<DeviceBuffer> allocate(size_t bytes, int device_id);
std::shared_ptr<DeviceBuffer> adopt_external(
  void * pointer, size_t bytes, int device_id, std::shared_ptr<void> owner);
std::shared_ptr<DeviceBuffer> adopt_synchronized_external(
  void * pointer, size_t bytes, int device_id, std::shared_ptr<void> owner);
FixedDeviceMemoryPool make_fixed_device_pool(size_t block_size, size_t block_count, int device_id);
bool wait_for_pending_releases(std::chrono::milliseconds timeout);
} // namespace gpu_ros_managed::cuda
#endif
