// Copyright 2026 gpu_ros_managed contributors
// Licensed under the Apache License, Version 2.0.
#ifndef GPU_ROS_ROSIDL_BUFFER__BUFFER_ACCESS_HPP_
#define GPU_ROS_ROSIDL_BUFFER__BUFFER_ACCESS_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "rosidl_buffer/buffer.hpp"

namespace gpu_ros::rosidl_buffer
{

// The pointer aliases the original Buffer storage; acquiring a lease never clones
// or promotes it. Keep owner alive until all consumer work has completed, not
// merely until that work has been submitted. Do not mutate/replace the Buffer
// while a lease is live. Destroy dependent tensor views before releasing owner.
struct BufferReadLease
{
  const void * data{};
  size_t byte_count{};
  std::shared_ptr<const void> owner;
};

class IBufferAccess
{
public:
  virtual ~IBufferAccess() = default;

  // message_owner must own the message containing buffer (or buffer itself).
  // Returns storage ready for consumption, retaining both message and backend
  // read handle where applicable. Unsupported backends/devices are rejected;
  // there is no implicit host materialization or device promotion.
  virtual BufferReadLease AcquireRead(
    const rosidl::Buffer<uint8_t> & buffer,
    std::shared_ptr<const void> message_owner) = 0;
};

// Accepts only the real rosidl CPU backend. Empty CPU buffers are permitted.
std::unique_ptr<IBufferAccess> CreateCpuBufferAccess();

}  // namespace gpu_ros::rosidl_buffer

#endif  // GPU_ROS_ROSIDL_BUFFER__BUFFER_ACCESS_HPP_
