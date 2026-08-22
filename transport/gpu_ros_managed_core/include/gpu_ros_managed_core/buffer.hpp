// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright 2026 gpu_ros_managed contributors
// Licensed under the Apache License, Version 2.0.
// Modified from NVIDIA Isaac ROS NITROS 4.5 for a backend-neutral CUDA/HIP
// buffer handle API.
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
enum class BufferReadiness
{
  kNotReady,
  kSynchronouslyReady,
  kEventBackedReady
};

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
  // Retain a source owner (for example a ROS Image) until this producer's
  // completion event has made the device buffer safe to release.
  void retain_owner(std::shared_ptr<const void> owner);
  void finalize();
  // Mark a producer operation failed when the caller knows that no
  // trustworthy completion event can be recorded. The allocation is then
  // orphan-safe and cannot be returned to a pool.
  void fail() noexcept;
  // Cancel a reservation before any producer work is submitted. The fresh
  // buffer can then be returned to a fixed pool for reuse.
  void cancel();

private:
  explicit WriteHandle(std::shared_ptr<detail::BufferState> state);
  std::shared_ptr<detail::BufferState> state_;
  bool responsible_{true};
  friend class DeviceBuffer;
};

class SynchronizedWriteHandle
{
public:
  SynchronizedWriteHandle(SynchronizedWriteHandle && other) noexcept;
  SynchronizedWriteHandle(const SynchronizedWriteHandle &) = delete;
  SynchronizedWriteHandle & operator=(const SynchronizedWriteHandle &) = delete;
  SynchronizedWriteHandle & operator=(SynchronizedWriteHandle &&) = delete;
  ~SynchronizedWriteHandle() noexcept;
  uint8_t * data() const noexcept;
  size_t size() const noexcept;
  // ORT-style writers must call finalize_synchronously only after the
  // external operation has synchronously completed. Destruction without an
  // explicit finalize/cancel marks the buffer failed and non-recyclable.
  void finalize_synchronously();
  void cancel();
  void fail() noexcept;

private:
  explicit SynchronizedWriteHandle(std::shared_ptr<detail::BufferState> state);
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
  BufferReadiness readiness() const noexcept;
  bool failed() const noexcept;
  WriteHandle get_write_handle(const DeviceStream & producer_stream);
  SynchronizedWriteHandle get_synchronized_write_handle();
  ReadHandle get_read_handle(const DeviceStream & consumer_stream) const;
  BlockingReadyLease get_blocking_ready_lease() const;
  // A successful blocking H2D copy is the complete producer operation and
  // transitions a fresh buffer to ready without creating a producer event.
  void copy_from_host_blocking(const void * source, size_t bytes);
  // Copies a ready buffer to host memory after synchronizing its producer.
  void copy_to_host_blocking(void * destination, size_t bytes) const;

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
};
}  // namespace detail
}  // namespace gpu_ros_managed
#endif
