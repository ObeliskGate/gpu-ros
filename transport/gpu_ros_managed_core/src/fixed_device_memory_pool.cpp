// Modified from NVIDIA Isaac ROS NITROS memory-pool sources for the managed backend-neutral API.

// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright 2026 gpu_ros_managed contributors
// Licensed under the Apache License, Version 2.0.
#include "gpu_ros_managed_core/fixed_device_memory_pool.hpp"

#include <condition_variable>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "gpu_ros_managed_core/detail/backend_ops.hpp"

namespace gpu_ros_managed::detail
{
struct PoolState
{
  DeviceId device;
  size_t block_size{0};
  size_t block_count{0};
  std::shared_ptr<void> allocation;
  uint8_t * base{nullptr};
  std::shared_ptr<BackendOps> ops;
  mutable std::mutex mutex;
  std::condition_variable cv;
  std::vector<size_t> free_blocks;
  bool stopping{false};
};
}  // namespace gpu_ros_managed::detail

namespace gpu_ros_managed
{
namespace
{
std::unique_ptr<PoolBlock> acquire_impl(
  const std::shared_ptr<detail::PoolState> & state, const DeviceStream & stream,
  std::chrono::milliseconds * timeout)
{
  if (stream.device_id() != state->device) {
    throw std::invalid_argument("Pool and stream backend/device mismatch");
  }
  size_t index;
  {
    std::unique_lock<std::mutex> lock(state->mutex);
    const auto ready = [&] {return state->stopping || !state->free_blocks.empty();};
    if (timeout) {
      if (!state->cv.wait_for(lock, *timeout, ready)) {return nullptr;}
    } else {
      state->cv.wait(lock, ready);
    }
    if (state->stopping) {throw std::runtime_error("Pool is shutting down");}
    index = state->free_blocks.back();
    state->free_blocks.pop_back();
  }

  auto block_token = std::shared_ptr<void>(
    state->base + index * state->block_size,
    [state, index](void *) {
      {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->free_blocks.push_back(index);
      }
      state->cv.notify_all();
    });
  auto * pointer = static_cast<uint8_t *>(block_token.get());
  auto buffer = detail::DeviceBufferFactory::make_fresh(
    state->device, pointer, state->block_size, std::move(block_token), state->ops);
  return std::make_unique<PoolBlock>(buffer, buffer->get_write_handle(stream));
}

std::unique_ptr<SynchronizedPoolBlock> acquire_synchronized_impl(
  const std::shared_ptr<detail::PoolState> & state,
  std::chrono::milliseconds * timeout)
{
  size_t index;
  {
    std::unique_lock<std::mutex> lock(state->mutex);
    const auto ready = [&] {return state->stopping || !state->free_blocks.empty();};
    if (timeout) {
      if (!state->cv.wait_for(lock, *timeout, ready)) {return nullptr;}
    } else {
      state->cv.wait(lock, ready);
    }
    if (state->stopping) {throw std::runtime_error("Pool is shutting down");}
    index = state->free_blocks.back();
    state->free_blocks.pop_back();
  }

  auto block_token = std::shared_ptr<void>(
    state->base + index * state->block_size,
    [state, index](void *) {
      {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->free_blocks.push_back(index);
      }
      state->cv.notify_all();
    });
  auto * pointer = static_cast<uint8_t *>(block_token.get());
  auto buffer = detail::DeviceBufferFactory::make_fresh(
    state->device, pointer, state->block_size, std::move(block_token), state->ops);
  return std::make_unique<SynchronizedPoolBlock>(
    buffer, buffer->get_synchronized_write_handle());
}
}  // namespace

PoolBlock FixedDeviceMemoryPool::acquire(const DeviceStream & stream)
{
  auto result = acquire_impl(state_, stream, nullptr);
  return std::move(*result);
}
std::unique_ptr<PoolBlock> FixedDeviceMemoryPool::acquire_for(
  const DeviceStream & stream, std::chrono::milliseconds timeout)
{
  return acquire_impl(state_, stream, &timeout);
}
SynchronizedPoolBlock FixedDeviceMemoryPool::acquire_synchronized()
{
  auto result = acquire_synchronized_impl(state_, nullptr);
  return std::move(*result);
}
std::unique_ptr<SynchronizedPoolBlock> FixedDeviceMemoryPool::acquire_synchronized_for(
  std::chrono::milliseconds timeout)
{
  return acquire_synchronized_impl(state_, &timeout);
}
size_t FixedDeviceMemoryPool::available() const noexcept
{
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->free_blocks.size();
}
size_t FixedDeviceMemoryPool::capacity() const noexcept {return state_->block_count;}
bool FixedDeviceMemoryPool::shutdown(std::chrono::milliseconds timeout)
{
  std::unique_lock<std::mutex> lock(state_->mutex);
  state_->stopping = true;
  state_->cv.notify_all();
  const bool drained = state_->cv.wait_for(
    lock, timeout, [&] {return state_->free_blocks.size() == state_->block_count;});
  if (drained) {
    state_->allocation.reset();
    state_->base = nullptr;
  }
  return drained;
}

FixedDeviceMemoryPool detail::PoolFactory::make(
  DeviceId device, size_t block_size, size_t block_count,
  std::shared_ptr<BackendOps> ops)
{
  if (block_size == 0 || block_count == 0) {
    throw std::invalid_argument("Pool block size and count must be non-zero");
  }
  if (block_size > std::numeric_limits<size_t>::max() / block_count) {
    throw std::overflow_error("Pool capacity overflows size_t");
  }
  if (!ops || ops->kind() != device.backend) {
    throw std::invalid_argument("Pool backend does not match DeviceId");
  }
  auto state = std::make_shared<PoolState>();
  state->device = device;
  state->block_size = block_size;
  state->block_count = block_count;
  state->ops = std::move(ops);
  state->allocation = state->ops->allocate_device(device.ordinal, block_size * block_count);
  state->base = static_cast<uint8_t *>(state->allocation.get());
  state->free_blocks.reserve(block_count);
  for (size_t i = 0; i < block_count; ++i) {state->free_blocks.push_back(i);}
  return FixedDeviceMemoryPool(std::move(state));
}
}  // namespace gpu_ros_managed
