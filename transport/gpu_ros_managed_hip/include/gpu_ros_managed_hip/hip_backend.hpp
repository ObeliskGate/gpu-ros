// Copyright 2026 Boshen Chen
// Licensed under the Apache License, Version 2.0.
#ifndef GPU_ROS_MANAGED_HIP__HIP_BACKEND_HPP_
#define GPU_ROS_MANAGED_HIP__HIP_BACKEND_HPP_
#include <hip/hip_runtime_api.h>
#include <chrono>
#include <cstddef>
#include <memory>
#include "gpu_ros_managed_core/buffer.hpp"
#include "gpu_ros_managed_core/fixed_device_memory_pool.hpp"
namespace gpu_ros_managed::hip
{
class HipStream
{
public:
  const DeviceStream & stream() const noexcept {return stream_;}
  hipStream_t get() const;
  operator const DeviceStream &() const noexcept {return stream_;}
private:
  explicit HipStream(DeviceStream stream) : stream_(std::move(stream)) {}
  DeviceStream stream_;
  friend HipStream make_stream(int);
  friend HipStream wrap_borrowed_stream(hipStream_t, int, std::shared_ptr<void>);
};
HipStream make_stream(int device_id);
HipStream wrap_borrowed_stream(hipStream_t, int, std::shared_ptr<void> owner = {});
std::shared_ptr<DeviceBuffer> allocate(size_t, int);
std::shared_ptr<DeviceBuffer> adopt_external(void *, size_t, int, std::shared_ptr<void>);
std::shared_ptr<DeviceBuffer> adopt_synchronized_external(void *, size_t, int, std::shared_ptr<void>);
FixedDeviceMemoryPool make_fixed_device_pool(size_t, size_t, int);
bool wait_for_pending_releases(std::chrono::milliseconds);
}  // namespace gpu_ros_managed::hip
#endif
