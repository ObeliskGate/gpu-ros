// Copyright 2026 Boshen Chen
// Licensed under the Apache License, Version 2.0.
#ifndef GPU_ROS_MANAGED_CORE__HOST_BUFFER_HPP_
#define GPU_ROS_MANAGED_CORE__HOST_BUFFER_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gpu_ros_managed
{
class HostBuffer
{
public:
  HostBuffer() = default;
  HostBuffer(std::shared_ptr<const void> owner, const void * data, size_t size)
  : owner_(std::move(owner)), data_(static_cast<const uint8_t *>(data)), size_(size)
  {
    if (size_ != 0 && data_ == nullptr) {
      throw std::invalid_argument("HostBuffer data is null");
    }
    if (size_ != 0 && !owner_) {
      throw std::invalid_argument("HostBuffer external storage requires an owner");
    }
  }
  static HostBuffer copy(const void * data, size_t size)
  {
    if (size != 0 && data == nullptr) {throw std::invalid_argument("HostBuffer data is null");}
    auto bytes = std::make_shared<std::vector<uint8_t>>();
    if (size != 0) {
      const auto * begin = static_cast<const uint8_t *>(data);
      bytes->assign(begin, begin + size);
    }
    return HostBuffer(bytes, bytes->data(), bytes->size());
  }
  const uint8_t * data() const noexcept {return data_;}
  size_t size() const noexcept {return size_;}
  const std::shared_ptr<const void> & owner() const noexcept {return owner_;}

private:
  std::shared_ptr<const void> owner_;
  const uint8_t * data_{nullptr};
  size_t size_{0};
};
}  // namespace gpu_ros_managed
#endif
