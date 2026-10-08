// Copyright 2026 gpu_ros_managed contributors
// Licensed under the Apache License, Version 2.0.
#include "gpu_ros_nvidia_tensor_bundle_compat/native_tensor_list.hpp"
#include "native_tensor_list_internal.hpp"
#include "tensor_metadata.hpp"

#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <utility>
#include "cuda_buffer/cuda_buffer_api.hpp"

namespace gpu_ros::nvidia_tensor_bundle_compat::native
{
namespace metadata = gpu_ros::nvidia_tensor_bundle_compat::detail;
namespace
{
void Check(cudaError_t status)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}
class DeviceScope
{
public:
  explicit DeviceScope(int device)
  {
    Check(cudaGetDevice(&previous_));
    Check(cudaSetDevice(device));
  }
  ~DeviceScope() { static_cast<void>(cudaSetDevice(previous_)); }
private:
  int previous_;
};
struct Stream
{
  explicit Stream(int ordinal) : device(ordinal)
  {
    DeviceScope scope(device);
    Check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking));
  }
  ~Stream()
  {
    int previous = device;
    static_cast<void>(cudaGetDevice(&previous));
    static_cast<void>(cudaSetDevice(device));
    static_cast<void>(cudaStreamDestroy(value));
    static_cast<void>(cudaSetDevice(previous));
  }
  const int device;
  cudaStream_t value{};
};
bool Drain(int device, cudaStream_t stream) noexcept
{
  int previous = device;
  if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(device) != cudaSuccess) {
    return false;
  }
  const bool completed = cudaStreamSynchronize(stream) == cudaSuccess;
  static_cast<void>(cudaSetDevice(previous));
  return completed;
}
} // namespace
namespace detail
{
enum class Status { Pending, Ready, Cancelled, Failed };
struct State
{
  ~State()
  {
    int previous = device;
    static_cast<void>(cudaGetDevice(&previous));
    static_cast<void>(cudaSetDevice(device));
    writers.clear();
    readers.clear();
    message.reset();
    static_cast<void>(cudaSetDevice(previous));
  }
  int device{};
  cudaStream_t stream{};
  std::shared_ptr<const void> stream_owner;
  Message::ConstSharedPtr message;
  std::vector<TensorSpec> specs;
  std::vector<size_t> bytes;
  std::vector<const uint8_t *> pointers;
  std::vector<cuda_buffer_backend::ReadHandle> readers;
  std::vector<cuda_buffer_backend::WriteHandle> writers;
  std::shared_ptr<const void> retained_owner;
  std::shared_ptr<const void> copy_owner;
  std::shared_ptr<std::atomic<bool>> poisoned;
  std::atomic<Status> status{Status::Pending};
  // A failure-only self reference uses the already allocated control block.
  // Deliberately never broken: uncertain GPU users survive until process exit.
  std::shared_ptr<State> quarantine;
};
struct Access
{
  static TensorList Wrap(std::shared_ptr<State> state) { return TensorList(std::move(state)); }
  static const std::shared_ptr<State> & Get(const TensorList & value) { return value.state_; }
};
void Fail(const std::shared_ptr<State> & state) noexcept
{
  if (!state || state->status.load(std::memory_order_acquire) != Status::Pending) {
    return;
  }
  state->quarantine = state;
  if (state->poisoned) {
    state->poisoned->store(true, std::memory_order_release);
  }
  state->status.store(Status::Failed, std::memory_order_release);
}
void RequirePending(const std::shared_ptr<State> & state)
{
  if (!state || state->status.load(std::memory_order_acquire) != Status::Pending) {
    throw std::logic_error("native output batch is not pending");
  }
}
static TensorList ImportImpl(Message::ConstSharedPtr message, int device,
  cudaStream_t stream, std::shared_ptr<const void> stream_owner,
  std::shared_ptr<std::atomic<bool>> poisoned = {})
{
  if (!message || device < 0) {
    throw std::invalid_argument("native import requires message and CUDA device");
  }
  metadata::ValidateNames(*message);
  auto state = std::make_shared<State>();
  state->poisoned = std::move(poisoned);
  state->device = device;
  state->stream = stream;
  state->stream_owner = std::move(stream_owner);
  state->message = std::move(message);
  const size_t count = state->message->tensors.size();
  state->specs.reserve(count);
  state->bytes.reserve(count);
  state->pointers.reserve(count);
  state->readers.reserve(count);
  // Validate every field before any promotion or backend handle acquisition.
  for (size_t i = 0; i < count; ++i) {
    const auto & tensor = state->message->tensors[i];
    state->bytes.push_back(metadata::ValidateTensor(tensor));
    state->specs.push_back({state->message->names[i], tensor.dtype_code,
      tensor.dtype_bits, tensor.dtype_lanes, tensor.shape});
  }
  if (!state->stream_owner) {
    auto owned_stream = std::make_shared<Stream>(device);
    state->stream = owned_stream->value;
    state->stream_owner = std::move(owned_stream);
    stream = state->stream;
  }
  DeviceScope scope(device);
  try {
    for (const auto & tensor : state->message->tensors) {
      state->readers.push_back(cuda_buffer_backend::from_input_buffer(tensor.data, stream));
      state->pointers.push_back(state->readers.back().get_ptr());
    }
    Check(cudaStreamSynchronize(stream));
  } catch (...) {
    if (!Drain(device, stream)) {
      Fail(state);
    }
    throw;
  }
  for (const auto * pointer : state->pointers) {
    cudaPointerAttributes attributes{};
    Check(cudaPointerGetAttributes(&attributes, pointer));
    if (attributes.device != device || attributes.type != cudaMemoryTypeDevice) {
      throw std::invalid_argument("TensorList storage is on the wrong CUDA device");
    }
  }
  state->status.store(Status::Ready, std::memory_order_release);
  return Access::Wrap(std::move(state));
}
TensorList ImportTensorList(Message::ConstSharedPtr message, int device,
  cudaStream_t stream, std::shared_ptr<const void> stream_owner)
{
  if (!stream_owner) {
    throw std::invalid_argument("borrowed native stream requires an owner");
  }
  return ImportImpl(std::move(message), device, stream, std::move(stream_owner));
}
TensorList ImportTensorList(Message::ConstSharedPtr message, int device)
{
  return ImportImpl(std::move(message), device, nullptr, {});
}
Message::ConstSharedPtr RawMessage(const TensorList & value)
{
  const auto & state = Access::Get(value);
  if (state->status.load(std::memory_order_acquire) != Status::Ready) {
    throw std::logic_error("native TensorList is not ready");
  }
  return state->message;
}
} // namespace detail

TensorList::TensorList(std::shared_ptr<detail::State> state) : state_(std::move(state)) {}
const std_msgs::msg::Header & TensorList::header() const noexcept { return state_->message->header; }
const std::vector<TensorSpec> & TensorList::tensors() const noexcept { return state_->specs; }
int TensorList::device_id() const noexcept { return state_->device; }
std::shared_ptr<const void> TensorList::owner() const noexcept { return state_; }
bool TensorList::same_message(const TensorList & other) const noexcept
{
  return state_->message == other.state_->message;
}
const uint8_t * TensorList::data(size_t index) const
{
  if (state_->status.load(std::memory_order_acquire) != detail::Status::Ready) {
    throw std::logic_error("native TensorList is not ready");
  }
  return state_->pointers.at(index);
}
OutputBatch::OutputBatch(std::shared_ptr<detail::State> state) : state_(std::move(state)) {}
OutputBatch::~OutputBatch() { FailAfterSubmit(); }
OutputBatch::OutputBatch(OutputBatch &&) noexcept = default;
OutputBatch & OutputBatch::operator=(OutputBatch && other) noexcept
{
  if (this != &other) {
    FailAfterSubmit();
    state_ = std::move(other.state_);
  }
  return *this;
}
const std::vector<TensorSpec> & OutputBatch::tensors() const noexcept
{
  if (!state_) { std::terminate(); }
  return state_->specs;
}
void * OutputBatch::data(size_t index) const
{
  if (!state_) { throw std::logic_error("native batch was moved"); }
  const auto status = state_->status.load(std::memory_order_acquire);
  if (status != detail::Status::Pending && status != detail::Status::Ready) {
    throw std::logic_error("native batch was cancelled or failed");
  }
  return const_cast<uint8_t *>(state_->pointers.at(index));
}
std::shared_ptr<const void> OutputBatch::owner() const noexcept
{
  if (!state_) { std::terminate(); }
  return state_;
}
TensorList OutputBatch::message() const
{
  if (!state_) { throw std::logic_error("native batch was moved"); }
  return TensorList(state_);
}
void OutputBatch::RetainOwner(std::shared_ptr<const void> owner)
{
  detail::RequirePending(state_);
  state_->retained_owner = std::move(owner);
}
void OutputBatch::CompleteAfterSync()
{
  detail::RequirePending(state_);
  try {
    DeviceScope scope(state_->device);
    state_->writers.clear();
    Check(cudaStreamSynchronize(state_->stream));
  } catch (...) {
    if (!Drain(state_->device, state_->stream)) {
      FailAfterSubmit();
    } else {
      CancelBeforeSubmit();
    }
    throw;
  }
  state_->retained_owner.reset();
  state_->copy_owner.reset();
  state_->status.store(detail::Status::Ready, std::memory_order_release);
}
void OutputBatch::CancelBeforeSubmit() noexcept
{
  if (!state_ || state_->status.load(std::memory_order_acquire) != detail::Status::Pending) {
    return;
  }
  try {
    DeviceScope scope(state_->device);
    state_->writers.clear();
    if (!Drain(state_->device, state_->stream)) {
      FailAfterSubmit();
      return;
    }
    state_->retained_owner.reset();
    state_->copy_owner.reset();
    state_->status.store(detail::Status::Cancelled, std::memory_order_release);
  } catch (...) {
    FailAfterSubmit();
  }
}
void OutputBatch::FailAfterSubmit() noexcept { detail::Fail(state_); }
void OutputBatch::CopyFrom(const std::vector<CopySource> & sources,
  std::shared_ptr<const void> source_owner)
{
  detail::RequirePending(state_);
  bool submitted = false;
  try {
    if (sources.size() != state_->specs.size() || (!sources.empty() && !source_owner)) {
      throw std::invalid_argument("native copy source count or owner mismatch");
    }
    DeviceScope scope(state_->device);
    for (size_t i = 0; i < sources.size(); ++i) {
      const auto & source = sources[i];
      if (!source.data || source.bytes != state_->bytes[i]) {
        throw std::invalid_argument("native copy source pointer or byte size mismatch");
      }
      if (source.on_device) {
        cudaPointerAttributes attributes{};
        Check(cudaPointerGetAttributes(&attributes, source.data));
        if (attributes.type != cudaMemoryTypeDevice || attributes.device != state_->device) {
          throw std::invalid_argument("native copy requires a matching CUDA device");
        }
      }
    }
    state_->copy_owner = std::move(source_owner);
    for (size_t i = 0; i < sources.size(); ++i) {
      const auto & source = sources[i];
      submitted = true;
      cuda_buffer_backend::to_buffer(source.data, source.bytes, state_->writers[i], state_->stream,
        source.on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice);
    }
    Check(cudaStreamSynchronize(state_->stream));
    CompleteAfterSync();
  } catch (...) {
    if (submitted && !Drain(state_->device, state_->stream)) {
      FailAfterSubmit();
    } else {
      CancelBeforeSubmit();
    }
    throw;
  }
}

struct TensorListTransport::Impl
{
  rclcpp::Node * node;
  int device;
  rclcpp::Publisher<detail::Message>::SharedPtr publisher;
  rclcpp::Subscription<detail::Message>::SharedPtr subscription;
  std::shared_ptr<std::atomic<bool>> poisoned{std::make_shared<std::atomic<bool>>(false)};
  std::mutex mutex;
  std::condition_variable cv;
  size_t active{0};
  bool stopping{false};
};
TensorListTransport::TensorListTransport(rclcpp::Node * node, int device_id,
  const std::string & output_topic, bool publish_output) : impl_(std::make_shared<Impl>())
{
  if (!node || device_id < 0) {
    throw std::invalid_argument("native transport requires node and nonnegative device");
  }
  impl_->node = node;
  impl_->device = device_id;
  if (publish_output) {
    rclcpp::PublisherOptions options;
    options.use_intra_process_comm = rclcpp::IntraProcessSetting::Disable;
    impl_->publisher = node->create_publisher<detail::Message>(output_topic, rclcpp::QoS(10), options);
  }
}
TensorListTransport::~TensorListTransport() { Unsubscribe(); }
void TensorListTransport::Unsubscribe()
{
  {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    impl_->stopping = true;
    impl_->cv.wait(lock, [this] { return impl_->active == 0; });
  }
  impl_->subscription.reset();
}
void TensorListTransport::Subscribe(Callback callback, const std::string & input_topic)
{
  Unsubscribe();
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->poisoned->load(std::memory_order_acquire)) {
      throw std::logic_error("native transport has unknown GPU completion");
    }
    impl_->stopping = false;
  }
  impl_->subscription = impl_->node->create_subscription<detail::Message>(input_topic,
    rclcpp::QoS(10), [weak = std::weak_ptr<Impl>(impl_), callback = std::move(callback),
      logger = impl_->node->get_logger()](detail::Message::ConstSharedPtr message) {
      auto state = weak.lock();
      if (!state) { return; }
      {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->stopping || state->poisoned->load(std::memory_order_acquire)) { return; }
        ++state->active;
      }
      struct Guard
      {
        std::shared_ptr<Impl> state;
        ~Guard()
        {
          std::lock_guard<std::mutex> lock(state->mutex);
          --state->active;
          state->cv.notify_all();
        }
      } guard{state};
      try {
        callback(detail::ImportImpl(std::move(message), state->device, nullptr, {}, state->poisoned));
      } catch (const std::exception & error) {
        RCLCPP_ERROR(logger, "%s native TensorList frame: %s",
          state->poisoned->load(std::memory_order_acquire) ? "Stopping after" : "Dropping", error.what());
      }
    });
}
OutputBatch TensorListTransport::Allocate(const std_msgs::msg::Header & header,
  const std::vector<TensorSpec> & specs)
{
  if (impl_->poisoned->load(std::memory_order_acquire)) {
    throw std::logic_error("native transport has unknown GPU completion");
  }
  auto state = std::make_shared<detail::State>();
  state->device = impl_->device;
  state->poisoned = impl_->poisoned;
  state->specs = specs;
  state->bytes.reserve(specs.size());
  auto message = std::make_unique<detail::Message>();
  message->header = header;
  message->names.reserve(specs.size());
  message->tensors.reserve(specs.size());
  for (const auto & spec : specs) {
    auto & tensor = message->tensors.emplace_back();
    state->bytes.push_back(metadata::SetTensorMetadata(tensor,
      spec.dtype_code, spec.dtype_bits, spec.dtype_lanes, spec.shape));
    message->names.push_back(spec.name);
  }
  metadata::ValidateNames(*message);
  state->writers.reserve(specs.size());
  state->pointers.reserve(specs.size());
  auto stream = std::make_shared<Stream>(state->device);
  state->stream = stream->value;
  state->stream_owner = stream;
  // A downstream raw ROS message can outlive every facade and its transport.
  // Retain the stream only, never the state that itself owns this message.
  auto shared_message = std::shared_ptr<detail::Message>(message.release(),
    [stream](detail::Message * value) noexcept {
      int previous = stream->device;
      static_cast<void>(cudaGetDevice(&previous));
      static_cast<void>(cudaSetDevice(stream->device));
      delete value;
      static_cast<void>(cudaSetDevice(previous));
    });
  state->message = shared_message;
  OutputBatch batch(state);
  try {
    DeviceScope scope(state->device);
    for (size_t i = 0; i < specs.size(); ++i) {
      auto & tensor = shared_message->tensors[i];
      tensor.data = cuda_buffer_backend::allocate_buffer(state->bytes[i]);
      state->writers.push_back(cuda_buffer_backend::from_output_buffer(tensor.data, state->stream));
      state->pointers.push_back(state->writers.back().get_ptr());
    }
  } catch (...) {
    if (Drain(state->device, state->stream)) { batch.CancelBeforeSubmit(); }
    else { batch.FailAfterSubmit(); }
    throw;
  }
  return batch;
}
void TensorListTransport::Publish(const TensorList & message)
{
  if (!impl_->publisher) { throw std::logic_error("native publisher disabled"); }
  if (message.device_id() != impl_->device) {
    throw std::invalid_argument("native message is on the wrong CUDA device");
  }
  // With IPC disabled rclcpp takes its inter-process const-reference path,
  // avoiding the Buffer clone in the intra-process const-reference overload.
  impl_->publisher->publish(*detail::RawMessage(message));
}
} // namespace gpu_ros::nvidia_tensor_bundle_compat::native
