// Copyright 2026 gpu_ros_managed contributors
// Licensed under the Apache License, Version 2.0.
#include "gpu_ros_rosidl_buffer/buffer_access.hpp"

#include <stdexcept>
#include <utility>

namespace gpu_ros::rosidl_buffer
{
namespace
{
class CpuBufferAccess final : public IBufferAccess
{
public:
  BufferReadLease AcquireRead(
    const rosidl::Buffer<uint8_t> & buffer,
    std::shared_ptr<const void> message_owner) override
  {
    if (!message_owner) {
      throw std::invalid_argument("CPU Buffer read requires a message owner");
    }
    if (buffer.get_backend_type() != "cpu" ||
      !dynamic_cast<const rosidl::CpuBufferImpl<uint8_t> *>(buffer.get_impl()))
    {
      throw std::invalid_argument("CPU Buffer access requires the rosidl CPU backend");
    }
    return {buffer.data(), buffer.size(), std::move(message_owner)};
  }
};
}  // namespace

std::unique_ptr<IBufferAccess> CreateCpuBufferAccess()
{
  return std::make_unique<CpuBufferAccess>();
}

}  // namespace gpu_ros::rosidl_buffer
