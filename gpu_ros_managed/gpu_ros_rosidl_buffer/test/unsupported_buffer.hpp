// Copyright 2026 gpu_ros_managed contributors
// Licensed under the Apache License, Version 2.0.
#ifndef GPU_ROS_ROSIDL_BUFFER__TEST__UNSUPPORTED_BUFFER_HPP_
#define GPU_ROS_ROSIDL_BUFFER__TEST__UNSUPPORTED_BUFFER_HPP_

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "rosidl_buffer/buffer.hpp"

namespace gpu_ros::rosidl_buffer::test
{
struct CopyAttempts
{
  size_t host{};
  size_t clone{};
};

// Deliberately not a supported storage implementation: verifies that neither a
// backend identifier nor host conversion can masquerade as direct access.
class UnsupportedBuffer final : public rosidl::BufferImplBase<uint8_t>
{
public:
  UnsupportedBuffer(std::string name, std::shared_ptr<CopyAttempts> attempts)
  : name_(std::move(name)), attempts_(std::move(attempts)) {}

  std::string get_backend_type() const override { return name_; }
  size_t size() const override { return 4; }
  std::unique_ptr<rosidl::BufferImplBase<uint8_t>> to_cpu() const override
  {
    ++attempts_->host;
    throw std::runtime_error("unexpected host materialization");
  }
  std::unique_ptr<rosidl::BufferImplBase<uint8_t>> clone() const override
  {
    ++attempts_->clone;
    throw std::runtime_error("unexpected storage clone");
  }

private:
  std::string name_;
  std::shared_ptr<CopyAttempts> attempts_;
};
}  // namespace gpu_ros::rosidl_buffer::test

#endif  // GPU_ROS_ROSIDL_BUFFER__TEST__UNSUPPORTED_BUFFER_HPP_
