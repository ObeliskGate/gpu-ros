// Modified from NVIDIA Isaac ROS NITROS buffer sources for the managed backend-neutral API.

// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright 2026 gpu_ros_managed contributors
// Licensed under the Apache License, Version 2.0.
#include "gpu_ros_managed_core/buffer.hpp"

#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "gpu_ros_managed_core/detail/backend_ops.hpp"

namespace gpu_ros_managed::detail
{
enum class BufferPhase {kFresh, kWriting, kReady, kFailed, kReleasing, kReleased};

struct BufferState
{
  DeviceId device;
  uint8_t * data{nullptr};
  size_t size{0};
  std::shared_ptr<void> allocation_owner;
  std::shared_ptr<BackendOps> ops;
  std::mutex mutex;
  BufferPhase phase{BufferPhase::kFresh};
  NativeStream writer_stream{0};
  Event producer_event{0};
  std::vector<Event> reader_events;
  // Set when an operation using the allocation may have been submitted but no
  // trustworthy completion event exists. Such allocations must never be
  // returned to an allocator or pool.
  bool release_must_orphan{false};

  ~BufferState();
};

namespace
{
struct ReleaseTracker
{
  std::mutex mutex;
  std::condition_variable cv;
  size_t cuda{0};
  size_t hip{0};
};

ReleaseTracker & tracker()
{
  static auto * value = new ReleaseTracker;
  return *value;
}

size_t & count_for(ReleaseTracker & value, BackendKind backend)
{
  return backend == BackendKind::kCuda ? value.cuda : value.hip;
}

std::vector<std::shared_ptr<void>> & orphan_storage()
{
  static auto * value = new std::vector<std::shared_ptr<void>>;
  return *value;
}
std::mutex & orphan_mutex()
{
  static auto * value = new std::mutex;
  return *value;
}

void validate_stream(const BufferState & state, const StreamState & stream)
{
  if (state.device != stream.device) {
    throw std::invalid_argument("DeviceBuffer stream backend/device mismatch");
  }
  if (state.ops.get() != stream.ops.get()) {
    throw std::invalid_argument("DeviceBuffer and stream use different backend instances");
  }
}
}  // namespace

BufferState::~BufferState()
{
  std::vector<Event> events;
  if (producer_event != 0) {events.push_back(producer_event);}
  events.insert(events.end(), reader_events.begin(), reader_events.end());
  auto owner = std::move(allocation_owner);
  auto backend_ops = ops;
  const auto allocation_device = device;
  const bool must_orphan = release_must_orphan;
  phase = BufferPhase::kReleasing;
  if (!owner && events.empty()) {return;}

  auto & value = tracker();
  {
    std::lock_guard<std::mutex> lock(value.mutex);
    ++count_for(value, allocation_device.backend);
  }
  try {
    std::thread(
      [events, owner, backend_ops, allocation_device, must_orphan]() mutable {
        bool safe = !must_orphan;
        bool device_selected = false;
        try {
          backend_ops->select_device(allocation_device.ordinal);
          device_selected = true;
        } catch (...) {
          safe = false;
        }
        for (const Event event : events) {
          if (event == 0) {continue;}
          if (device_selected) {
            try {
              backend_ops->synchronize_event(event);
            } catch (...) {
              safe = false;
            }
          }
          // BackendOps requires this operation to be noexcept. Keep it outside
          // the synchronization try block so every event is destroyed exactly
          // once, including after an earlier synchronization failure.
          backend_ops->destroy_event(event);
        }
        if (safe || !owner) {
          owner.reset();
        } else {
          std::lock_guard<std::mutex> lock(orphan_mutex());
          orphan_storage().push_back(std::move(owner));
        }
        auto & release_tracker = tracker();
        {
          std::lock_guard<std::mutex> lock(release_tracker.mutex);
          --count_for(release_tracker, allocation_device.backend);
        }
        release_tracker.cv.notify_all();
      }).detach();
    // Keep a local owner until std::thread has successfully copied its
    // callable. If thread construction throws, the catch path can still
    // safe-orphan the allocation.
    owner.reset();
  } catch (...) {
    for (const Event event : events) {
      if (event != 0) {backend_ops->destroy_event(event);}
    }
    if (owner) {
      std::lock_guard<std::mutex> lock(orphan_mutex());
      orphan_storage().push_back(std::move(owner));
    }
    {
      std::lock_guard<std::mutex> tracker_lock(value.mutex);
      --count_for(value, allocation_device.backend);
    }
    value.cv.notify_all();
  }
}

const StreamState & StreamAccess::get(const DeviceStream & stream)
{
  if (!stream.state_) {throw std::invalid_argument("DeviceStream is empty");}
  return *stream.state_;
}

bool wait_for_pending_releases(BackendKind backend, std::chrono::milliseconds timeout)
{
  auto & value = tracker();
  std::unique_lock<std::mutex> lock(value.mutex);
  return value.cv.wait_for(lock, timeout, [&] {return count_for(value, backend) == 0;});
}

size_t pending_release_count(BackendKind backend) noexcept
{
  auto & value = tracker();
  std::lock_guard<std::mutex> lock(value.mutex);
  return count_for(value, backend);
}
}  // namespace gpu_ros_managed::detail

namespace gpu_ros_managed
{
BackendKind DeviceStream::backend_kind() const {return detail::StreamAccess::get(*this).device.backend;}
DeviceId DeviceStream::device_id() const {return detail::StreamAccess::get(*this).device;}

WriteHandle::WriteHandle(std::shared_ptr<detail::BufferState> state) : state_(std::move(state)) {}
WriteHandle::WriteHandle(WriteHandle && other) noexcept
: state_(std::move(other.state_)), responsible_(other.responsible_)
{
  other.responsible_ = false;
}
WriteHandle::~WriteHandle() noexcept
{
  if (!responsible_ || !state_) {return;}
  try {finalize();} catch (...) {}
}
uint8_t * WriteHandle::data() const noexcept {return state_ ? state_->data : nullptr;}
size_t WriteHandle::size() const noexcept {return state_ ? state_->size : 0;}
void WriteHandle::finalize()
{
  if (!responsible_ || !state_) {return;}
  std::lock_guard<std::mutex> lock(state_->mutex);
  if (state_->phase == detail::BufferPhase::kReady) {
    responsible_ = false;
    return;
  }
  if (state_->phase != detail::BufferPhase::kWriting) {
    throw std::logic_error("WriteHandle cannot finalize a buffer that is not being written");
  }
  detail::Event event = 0;
  try {
    event = state_->ops->create_event();
    state_->ops->record_event(event, state_->writer_stream);
    state_->producer_event = event;
    state_->phase = detail::BufferPhase::kReady;
    responsible_ = false;
  } catch (...) {
    if (event != 0) {state_->ops->destroy_event(event);}
    state_->phase = detail::BufferPhase::kFailed;
    state_->release_must_orphan = true;
    responsible_ = false;
    throw;
  }
}

ReadHandle::ReadHandle(
  std::shared_ptr<detail::BufferState> state, DeviceStream stream)
: state_(std::move(state)), stream_(std::move(stream))
{
  const auto & native = detail::StreamAccess::get(stream_);
  std::lock_guard<std::mutex> lock(state_->mutex);
  detail::validate_stream(*state_, native);
  if (state_->phase == detail::BufferPhase::kWriting) {
    throw std::logic_error("Reader acquisition rejected while writer is active");
  }
  if (state_->phase != detail::BufferPhase::kReady) {
    throw std::logic_error("Reader acquisition requires a ready buffer");
  }
  if (state_->producer_event != 0) {
    try {
      state_->ops->wait_event(native.native, state_->producer_event);
    } catch (...) {
      state_->phase = detail::BufferPhase::kFailed;
      throw;
    }
  }
}
ReadHandle::ReadHandle(ReadHandle && other) noexcept
: state_(std::move(other.state_)), stream_(std::move(other.stream_)),
  responsible_(other.responsible_)
{
  other.responsible_ = false;
}
ReadHandle::~ReadHandle() noexcept
{
  if (!responsible_ || !state_) {return;}
  try {finish();} catch (...) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->phase = detail::BufferPhase::kFailed;
  }
}
const uint8_t * ReadHandle::data() const noexcept {return state_ ? state_->data : nullptr;}
size_t ReadHandle::size() const noexcept {return state_ ? state_->size : 0;}
void ReadHandle::finish()
{
  if (!responsible_ || !state_) {return;}
  const auto & native = detail::StreamAccess::get(stream_);
  detail::Event event = 0;
  try {
    event = state_->ops->create_event();
    state_->ops->record_event(event, native.native);
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->reader_events.push_back(event);
    responsible_ = false;
  } catch (...) {
    if (event != 0) {state_->ops->destroy_event(event);}
    {
      std::lock_guard<std::mutex> lock(state_->mutex);
      state_->phase = detail::BufferPhase::kFailed;
      state_->release_must_orphan = true;
    }
    responsible_ = false;
    throw;
  }
}

BlockingReadyLease::BlockingReadyLease(std::shared_ptr<detail::BufferState> state)
: state_(std::move(state))
{
  detail::Event event = 0;
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->phase == detail::BufferPhase::kWriting) {
      throw std::logic_error("BlockingReadyLease rejected while writer is active");
    }
    if (state_->phase != detail::BufferPhase::kReady) {
      throw std::logic_error("BlockingReadyLease requires a ready buffer");
    }
    event = state_->producer_event;
  }
  if (event != 0) {
    try {
      state_->ops->synchronize_event(event);
    } catch (...) {
      std::lock_guard<std::mutex> lock(state_->mutex);
      state_->phase = detail::BufferPhase::kFailed;
      state_->release_must_orphan = true;
      throw;
    }
  }
}
const uint8_t * BlockingReadyLease::data() const noexcept {return state_ ? state_->data : nullptr;}
size_t BlockingReadyLease::size() const noexcept {return state_ ? state_->size : 0;}

size_t DeviceBuffer::size() const noexcept {return state_->size;}
DeviceId DeviceBuffer::device_id() const noexcept {return state_->device;}
WriteHandle DeviceBuffer::get_write_handle(const DeviceStream & stream)
{
  const auto & native = detail::StreamAccess::get(stream);
  std::lock_guard<std::mutex> lock(state_->mutex);
  detail::validate_stream(*state_, native);
  if (state_->phase != detail::BufferPhase::kFresh) {
    throw std::logic_error("DeviceBuffer writer can only be acquired once");
  }
  state_->phase = detail::BufferPhase::kWriting;
  state_->writer_stream = native.native;
  return WriteHandle(state_);
}
ReadHandle DeviceBuffer::get_read_handle(const DeviceStream & stream) const
{
  return ReadHandle(state_, stream);
}
BlockingReadyLease DeviceBuffer::get_blocking_ready_lease() const
{
  return BlockingReadyLease(state_);
}
void DeviceBuffer::copy_from_host_blocking(const void * source, size_t bytes)
{
  std::lock_guard<std::mutex> lock(state_->mutex);
  if (state_->phase != detail::BufferPhase::kFresh) {
    throw std::logic_error("DeviceBuffer H2D copy can only write a fresh buffer");
  }
  if (bytes > state_->size) {
    throw std::out_of_range("H2D copy exceeds DeviceBuffer size");
  }
  if (bytes != 0 && source == nullptr) {
    throw std::invalid_argument("H2D copy source is null for a non-empty copy");
  }
  try {
    if (bytes != 0) {
      state_->ops->copy_host_to_device(
        state_->device.ordinal, state_->data, source, bytes);
    }
    state_->phase = detail::BufferPhase::kReady;
  } catch (...) {
    state_->phase = detail::BufferPhase::kFailed;
    state_->release_must_orphan = true;
    throw;
  }
}
void DeviceBuffer::copy_to_host_blocking(void * destination, size_t bytes) const
{
  if (bytes > state_->size) {
    throw std::out_of_range("D2H copy exceeds DeviceBuffer size");
  }
  if (bytes != 0 && destination == nullptr) {
    throw std::invalid_argument("D2H copy destination is null for a non-empty copy");
  }
  auto lease = get_blocking_ready_lease();
  if (bytes != 0) {
    state_->ops->copy_device_to_host(
      state_->device.ordinal, destination, lease.data(), bytes);
  }
}

DeviceStream detail::DeviceBufferFactory::make_stream(
  DeviceId device, detail::NativeStream native, std::shared_ptr<void> owner,
  std::shared_ptr<detail::BackendOps> ops)
{
  if (!ops || ops->kind() != device.backend) {
    throw std::invalid_argument("Stream backend does not match DeviceId");
  }
  return detail::StreamAccess::make(device, native, std::move(owner), std::move(ops));
}

namespace
{
std::shared_ptr<DeviceBuffer> make_buffer(
  DeviceId device, void * data, size_t size, std::shared_ptr<void> owner,
  std::shared_ptr<detail::BackendOps> ops, detail::BufferPhase phase)
{
  if (!ops || ops->kind() != device.backend) {
    throw std::invalid_argument("Allocation backend does not match DeviceId");
  }
  if (size != 0 && data == nullptr) {throw std::invalid_argument("Device allocation is null");}
  if (size != 0 && !owner) {
    throw std::invalid_argument("Device allocation requires an owner");
  }
  auto state = std::make_shared<detail::BufferState>();
  state->device = device;
  state->data = static_cast<uint8_t *>(data);
  state->size = size;
  state->allocation_owner = std::move(owner);
  state->ops = std::move(ops);
  state->phase = phase;
  return std::shared_ptr<DeviceBuffer>(new DeviceBuffer(std::move(state)));
}
}  // namespace

std::shared_ptr<DeviceBuffer> detail::DeviceBufferFactory::make_fresh(
  DeviceId device, void * data, size_t size, std::shared_ptr<void> owner,
  std::shared_ptr<detail::BackendOps> ops)
{
  return make_buffer(device, data, size, std::move(owner), std::move(ops),
    detail::BufferPhase::kFresh);
}
std::shared_ptr<DeviceBuffer> detail::DeviceBufferFactory::make_ready(
  DeviceId device, void * data, size_t size, std::shared_ptr<void> owner,
  std::shared_ptr<detail::BackendOps> ops)
{
  return make_buffer(device, data, size, std::move(owner), std::move(ops),
    detail::BufferPhase::kReady);
}
}  // namespace gpu_ros_managed
