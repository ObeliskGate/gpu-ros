// Modified from NVIDIA Isaac ROS NITROS buffer sources for the managed backend-neutral API.

// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright 2026 gpu_ros_managed contributors
// Licensed under the Apache License, Version 2.0.
#ifndef GPU_ROS_MANAGED_CORE__BUFFER_HPP_
#define GPU_ROS_MANAGED_CORE__BUFFER_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "gpu_ros_managed_core/detail/backend_ops.hpp"
#include "gpu_ros_managed_core/device.hpp"

namespace gpu_ros_managed
{
namespace detail
{
struct BufferState;
class DeviceBufferFactory;
}  // namespace detail

class WriteHandle
{
public:
  WriteHandle(WriteHandle && other) noexcept;
  WriteHandle(const WriteHandle &) = delete;
  WriteHandle & operator=(const WriteHandle &) = delete;
  WriteHandle & operator=(WriteHandle &&) = delete;
  ~WriteHandle() noexcept;
  uint8_t * data() const noexcept;
  size_t size() const noexcept;
  void finalize();

private:
  explicit WriteHandle(std::shared_ptr<detail::BufferState> state);
  std::shared_ptr<detail::BufferState> state_;
  bool responsible_{true};
  friend class DeviceBuffer;
};

class ReadHandle
{
public:
  ReadHandle(ReadHandle && other) noexcept;
  ReadHandle(const ReadHandle &) = delete;
  ReadHandle & operator=(const ReadHandle &) = delete;
  ReadHandle & operator=(ReadHandle &&) = delete;
  ~ReadHandle() noexcept;
  const uint8_t * data() const noexcept;
  size_t size() const noexcept;
  void finish();

private:
  ReadHandle(std::shared_ptr<detail::BufferState> state, DeviceStream stream);
  std::shared_ptr<detail::BufferState> state_;
  DeviceStream stream_;
  bool responsible_{true};
  friend class DeviceBuffer;
};

class BlockingReadyLease
{
public:
  BlockingReadyLease(BlockingReadyLease &&) noexcept = default;
  BlockingReadyLease(const BlockingReadyLease &) = delete;
  BlockingReadyLease & operator=(const BlockingReadyLease &) = delete;
  BlockingReadyLease & operator=(BlockingReadyLease &&) = delete;
  const uint8_t * data() const noexcept;
  size_t size() const noexcept;

private:
  explicit BlockingReadyLease(std::shared_ptr<detail::BufferState> state);
  std::shared_ptr<detail::BufferState> state_;
  friend class DeviceBuffer;
};

class DeviceBuffer
{
public:
  // Internal construction point used by backend factories. BufferState is not
  // a public constructible type.
  explicit DeviceBuffer(std::shared_ptr<detail::BufferState> state) : state_(std::move(state)) {}
  size_t size() const noexcept;
  DeviceId device_id() const noexcept;
  WriteHandle get_write_handle(const DeviceStream & producer_stream);
  ReadHandle get_read_handle(const DeviceStream & consumer_stream) const;
  BlockingReadyLease get_blocking_ready_lease() const;

private:
  std::shared_ptr<detail::BufferState> state_;
  friend class detail::DeviceBufferFactory;
};

namespace detail
{
class DeviceBufferFactory
{
public:
  static DeviceStream make_stream(
    DeviceId, NativeStream, std::shared_ptr<void>, std::shared_ptr<BackendOps>);
  static std::shared_ptr<DeviceBuffer> make_fresh(
    DeviceId, void *, size_t, std::shared_ptr<void>, std::shared_ptr<BackendOps>);
  static std::shared_ptr<DeviceBuffer> make_ready(
    DeviceId, void *, size_t, std::shared_ptr<void>, std::shared_ptr<BackendOps>);
  static void copy_to_host_blocking(
    const DeviceBuffer &, void * destination, size_t bytes);
};
}  // namespace detail
}  // namespace gpu_ros_managed
#endif
