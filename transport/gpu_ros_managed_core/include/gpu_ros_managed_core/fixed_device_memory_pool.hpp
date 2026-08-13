// Modified from NVIDIA Isaac ROS NITROS memory-pool sources for the managed backend-neutral API.

// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright 2026 gpu_ros_managed contributors
// Licensed under the Apache License, Version 2.0.
#ifndef GPU_ROS_MANAGED_CORE__FIXED_DEVICE_MEMORY_POOL_HPP_
#define GPU_ROS_MANAGED_CORE__FIXED_DEVICE_MEMORY_POOL_HPP_

#include <chrono>
#include <cstddef>
#include <memory>
#include <utility>

#include "gpu_ros_managed_core/buffer.hpp"

namespace gpu_ros_managed
{
namespace detail {struct PoolState; class PoolFactory;}

struct PoolBlock
{
  std::shared_ptr<DeviceBuffer> buffer;
  WriteHandle writer;
  PoolBlock(std::shared_ptr<DeviceBuffer> b, WriteHandle w)
  : buffer(std::move(b)), writer(std::move(w)) {}
  PoolBlock(PoolBlock &&) noexcept = default;
  PoolBlock & operator=(PoolBlock &&) = delete;
};

struct SynchronizedPoolBlock
{
  std::shared_ptr<DeviceBuffer> buffer;
  SynchronizedWriteHandle writer;
  SynchronizedPoolBlock(std::shared_ptr<DeviceBuffer> b, SynchronizedWriteHandle w)
  : buffer(std::move(b)), writer(std::move(w)) {}
  SynchronizedPoolBlock(SynchronizedPoolBlock &&) noexcept = default;
  SynchronizedPoolBlock & operator=(SynchronizedPoolBlock &&) = delete;
};

class FixedDeviceMemoryPool
{
public:
  FixedDeviceMemoryPool(FixedDeviceMemoryPool &&) noexcept = default;
  FixedDeviceMemoryPool(const FixedDeviceMemoryPool &) = delete;
  FixedDeviceMemoryPool & operator=(const FixedDeviceMemoryPool &) = delete;
  FixedDeviceMemoryPool & operator=(FixedDeviceMemoryPool &&) = delete;
  PoolBlock acquire(const DeviceStream & stream);
  std::unique_ptr<PoolBlock> acquire_for(
    const DeviceStream & stream, std::chrono::milliseconds timeout);
  SynchronizedPoolBlock acquire_synchronized();
  std::unique_ptr<SynchronizedPoolBlock> acquire_synchronized_for(
    std::chrono::milliseconds timeout);
  size_t available() const noexcept;
  size_t capacity() const noexcept;
  bool shutdown(std::chrono::milliseconds timeout);

private:
  explicit FixedDeviceMemoryPool(std::shared_ptr<detail::PoolState> state)
  : state_(std::move(state)) {}
  std::shared_ptr<detail::PoolState> state_;
  friend class detail::PoolFactory;
};

namespace detail
{
class PoolFactory
{
public:
  static FixedDeviceMemoryPool make(
    DeviceId, size_t block_size, size_t block_count, std::shared_ptr<BackendOps>);
};
}
}  // namespace gpu_ros_managed
#endif
