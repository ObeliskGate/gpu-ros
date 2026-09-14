// Copyright 2026 Boshen Chen
// Licensed under the Apache License, Version 2.0.
#ifndef GPU_ROS_MANAGED_CORE__DEVICE_HPP_
#define GPU_ROS_MANAGED_CORE__DEVICE_HPP_

#include <memory>

namespace gpu_ros_managed
{
enum class BackendKind {kCuda, kHip};

struct DeviceId
{
  BackendKind backend;
  int ordinal;
};

inline bool operator==(const DeviceId & a, const DeviceId & b) noexcept
{
  return a.backend == b.backend && a.ordinal == b.ordinal;
}
inline bool operator!=(const DeviceId & a, const DeviceId & b) noexcept {return !(a == b);}

namespace detail
{
struct StreamState;
struct StreamAccess;
class BackendOps;
class DeviceBufferFactory;
}  // namespace detail

class DeviceStream
{
public:
  DeviceStream() = default;
  BackendKind backend_kind() const;
  DeviceId device_id() const;
  explicit operator bool() const noexcept {return static_cast<bool>(state_);}

private:
  explicit DeviceStream(std::shared_ptr<detail::StreamState> state) : state_(std::move(state)) {}
  std::shared_ptr<detail::StreamState> state_;
  friend class detail::DeviceBufferFactory;
  friend struct detail::StreamAccess;
};
}  // namespace gpu_ros_managed
#endif
