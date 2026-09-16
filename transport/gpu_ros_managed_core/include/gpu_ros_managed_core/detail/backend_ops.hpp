// Copyright 2026 Boshen Chen
// Licensed under the Apache License, Version 2.0.
#ifndef GPU_ROS_MANAGED_CORE__DETAIL__BACKEND_OPS_HPP_
#define GPU_ROS_MANAGED_CORE__DETAIL__BACKEND_OPS_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include "gpu_ros_managed_core/device.hpp"

namespace gpu_ros_managed::detail
{
using NativeStream = std::uintptr_t;
using Event = std::uintptr_t;

class BackendOps
{
public:
  virtual ~BackendOps() = default;
  virtual BackendKind kind() const noexcept = 0;
  virtual void select_device(int ordinal) = 0;
  virtual Event create_event() = 0;
  virtual void record_event(Event event, NativeStream stream) = 0;
  virtual void wait_event(NativeStream stream, Event event) = 0;
  virtual void synchronize_event(Event event) = 0;
  virtual void destroy_event(Event event) noexcept = 0;
  virtual std::shared_ptr<void> allocate_device(int ordinal, size_t bytes) = 0;
  virtual void copy_host_to_device(
    int ordinal, void * destination, const void * source, size_t bytes) = 0;
  virtual void copy_device_to_host(
    int ordinal, void * destination, const void * source, size_t bytes) = 0;
};

struct StreamState
{
  DeviceId device;
  NativeStream native{0};
  std::shared_ptr<void> owner;
  std::shared_ptr<BackendOps> ops;
};

struct StreamAccess
{
  static DeviceStream make(DeviceId device, NativeStream native, std::shared_ptr<void> owner,
    std::shared_ptr<BackendOps> ops)
  {
    auto state = std::make_shared<StreamState>();
    state->device = device;
    state->native = native;
    state->owner = std::move(owner);
    state->ops = std::move(ops);
    return DeviceStream(std::move(state));
  }
  static const StreamState & get(const DeviceStream & stream);
};

bool wait_for_pending_releases(BackendKind backend, std::chrono::milliseconds timeout);
size_t pending_release_count(BackendKind backend) noexcept;
} // namespace gpu_ros_managed::detail
#endif
