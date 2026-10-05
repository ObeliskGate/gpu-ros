// Copyright 2026 gpu_ros_managed contributors
// Licensed under the Apache License, Version 2.0.
#include "gpu_ros_nvidia_tensor_bundle_compat/tensor_list_buffer_adapter.hpp"
#include "tensor_list_buffer_adapter_detail.hpp"
#include "host_buffer_access.hpp"
#include <condition_variable>
#include <mutex>

#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include "cuda_buffer/cuda_buffer_api.hpp"
#include "gpu_ros_managed_cuda/cuda_backend.hpp"
#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"

namespace gpu_ros::nvidia_tensor_bundle_compat
{
namespace
{
using Message = isaac_ros_tensor_list_interfaces::msg::TensorList;
using namespace gpu_ros_managed;
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
struct NativeTensorListEnvelope
{
  explicit NativeTensorListEnvelope(int ordinal)
      : device(ordinal), stream(cuda::make_stream(ordinal))
  {
  }
  ~NativeTensorListEnvelope()
  {
    int previous = device;
    static_cast<void>(cudaGetDevice(&previous));
    static_cast<void>(cudaSetDevice(device));
    writers.clear();
    readers.clear();
    static_cast<void>(cudaSetDevice(previous));
  }
  int device;
  cuda::CudaStream stream;
  Message::ConstSharedPtr message;
  std::optional<ManagedTensorBundle> copy_source;
  std::shared_ptr<const void> input_owner;
  std::vector<cuda_buffer_backend::ReadHandle> readers;
  std::vector<cuda_buffer_backend::WriteHandle> writers;
};
struct NativeAttachment final : DeviceBufferAttachment
{
  NativeAttachment(
    std::shared_ptr<NativeTensorListEnvelope> value, size_t tensor_index, const void * data)
      : envelope(std::move(value)), index(tensor_index), device(envelope->device), pointer(data)
  {
  }
  const std::shared_ptr<NativeTensorListEnvelope> envelope;
  const size_t index;
  const int device;
  const void * const pointer;
};
NativeTensorSpec Spec(const Message::_tensors_type::value_type & tensor)
{
  NativeTensorSpec spec{tensor.name, static_cast<TensorDataType>(tensor.data_type), {}};
  std::vector<uint32_t> materialized_dims;
  const auto & dims = detail::HostValues(tensor.shape.dims, materialized_dims);
  if (dims.empty() || tensor.shape.rank != dims.size()) {
    throw std::invalid_argument("TensorList rank does not match positive dimensions");
  }
  spec.shape.assign(dims.begin(), dims.end());
  const auto bytes = tensor_byte_size(spec.shape, spec.dtype);
  std::vector<uint64_t> materialized_strides;
  const auto & strides = detail::HostValues(tensor.strides, materialized_strides);
  if (bytes != tensor.data.size() ||
      (!strides.empty() && strides != contiguous_strides(spec.shape, spec.dtype)))
  {
    throw std::invalid_argument("TensorList byte size or contiguous strides mismatch");
  }
  return spec;
}
std::shared_ptr<NativeTensorListEnvelope> Reusable(const ManagedTensorBundle & bundle, int device)
{
  std::shared_ptr<NativeTensorListEnvelope> envelope;
  for (size_t i = 0; i < bundle.tensors().size(); ++i) {
    const auto & tensor = bundle.tensors()[i];
    const auto * buffer = std::get_if<std::shared_ptr<DeviceBuffer>>(&tensor.storage());
    if (!buffer || !*buffer) {
      return {};
    }
    const auto attachment =
      std::dynamic_pointer_cast<const NativeAttachment>((*buffer)->attachment());
    if (!attachment || attachment->index != i || attachment->device != device ||
        (*buffer)->device_id().backend != BackendKind::kCuda ||
        (*buffer)->device_id().ordinal != device)
    {
      return {};
    }
    if (!envelope) {
      envelope = attachment->envelope;
    }
    if (envelope != attachment->envelope || envelope->message->header != bundle.header() ||
        envelope->message->tensors.size() != bundle.tensors().size())
    {
      return {};
    }
    const auto spec = Spec(envelope->message->tensors[i]);
    if (spec.name != tensor.name() || spec.dtype != tensor.data_type() ||
        spec.shape != tensor.shape() ||
        tensor.strides() != contiguous_strides(spec.shape, spec.dtype))
    {
      return {};
    }
    const auto lease = (*buffer)->get_blocking_ready_lease();
    if (lease.data() != attachment->pointer ||
        lease.size() != envelope->message->tensors[i].data.size())
    {
      return {};
    }
  }
  return envelope;
}
} // namespace
namespace detail
{
Message::ConstSharedPtr ReusableTensorList(const ManagedTensorBundle & bundle, int device)
{
  auto envelope = Reusable(bundle, device);
  return envelope ? envelope->message : Message::ConstSharedPtr{};
}
ManagedTensorBundle ImportTensorList(Message::ConstSharedPtr message, int device)
{
  DeviceScope scope(device);
  auto envelope = std::make_shared<NativeTensorListEnvelope>(device);
  envelope->message = std::move(message);
  envelope->readers.reserve(envelope->message->tensors.size());
  std::vector<NativeTensorSpec> specs;
  specs.reserve(envelope->message->tensors.size());
  for (const auto & tensor : envelope->message->tensors) {
    specs.push_back(Spec(tensor));
  }
  const auto retain_uncertain_owner = [&] {
    // No payload allocation: use Managed's orphan protocol for the message,
    // including any CPU sources of partially submitted promotion copies.
    auto orphan = cuda::adopt_external(nullptr, 0, device, envelope);
    orphan->get_synchronized_write_handle().fail();
  };
  try {
    for (const auto & tensor : envelope->message->tensors) {
      envelope->readers.push_back(
        cuda_buffer_backend::from_input_buffer(tensor.data, envelope->stream.get()));
    }
  } catch (...) {
    if (cudaStreamSynchronize(envelope->stream.get()) != cudaSuccess) {
      retain_uncertain_owner();
    }
    throw;
  }
  const auto status = cudaStreamSynchronize(envelope->stream.get());
  if (status != cudaSuccess) {
    retain_uncertain_owner();
  }
  Check(status);
  std::vector<ManagedTensor> tensors;
  tensors.reserve(specs.size());
  for (size_t i = 0; i < specs.size(); ++i) {
    cudaPointerAttributes attributes{};
    Check(cudaPointerGetAttributes(&attributes, envelope->readers[i].get_ptr()));
    if (attributes.device != device || attributes.type != cudaMemoryTypeDevice) {
      throw std::invalid_argument("TensorList storage is on the wrong CUDA device");
    }
    auto attachment =
      std::make_shared<NativeAttachment>(envelope, i, envelope->readers[i].get_ptr());
    auto buffer =
      cuda::adopt_synchronized_external(const_cast<uint8_t *>(envelope->readers[i].get_ptr()),
        envelope->message->tensors[i].data.size(), device, envelope, std::move(attachment));
    tensors.emplace_back(
      std::move(specs[i].name), specs[i].dtype, std::move(specs[i].shape), std::move(buffer));
  }
  return ManagedTensorBundle(envelope->message->header, std::move(tensors));
}
} // namespace detail

struct NativeOutputBatch::Impl
{
  std::shared_ptr<NativeTensorListEnvelope> envelope;
  std::vector<std::shared_ptr<DeviceBuffer>> buffers;
  std::vector<SynchronizedWriteHandle> writers;
  std::vector<void *> pointers;
  bool finished{false};
  ~Impl()
  {
    if (!finished) {
      for (auto & writer : writers) {
        writer.fail();
      }
    }
  }
};
NativeOutputBatch::NativeOutputBatch(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
NativeOutputBatch::~NativeOutputBatch() = default;
NativeOutputBatch::NativeOutputBatch(NativeOutputBatch &&) noexcept = default;
NativeOutputBatch & NativeOutputBatch::operator=(NativeOutputBatch &&) noexcept = default;
const std::vector<std::shared_ptr<DeviceBuffer>> & NativeOutputBatch::buffers() const
{
  return impl_->buffers;
}
void * NativeOutputBatch::pointer(size_t index) const
{
  return impl_->pointers.at(index);
}
void NativeOutputBatch::RetainOwner(std::shared_ptr<const void> owner)
{
  if (impl_->finished) {
    throw std::logic_error("native batch already completed");
  }
  impl_->envelope->input_owner = std::move(owner);
}
void NativeOutputBatch::CompleteAfterSync()
{
  if (impl_->finished) {
    throw std::logic_error("native batch already completed");
  }
  DeviceScope scope(impl_->envelope->device);
  impl_->envelope->writers.clear();
  Check(cudaStreamSynchronize(impl_->envelope->stream.get()));
  for (auto & writer : impl_->writers) {
    writer.finalize_synchronously();
  }
  impl_->envelope->input_owner.reset();
  impl_->finished = true;
}
void NativeOutputBatch::CancelBeforeSubmit() noexcept
{
  if (!impl_ || impl_->finished) {
    return;
  }
  for (auto & writer : impl_->writers) {
    try {
      writer.cancel();
    } catch (...) {
      writer.fail();
    }
  }
  impl_->envelope->writers.clear();
  impl_->envelope->copy_source.reset();
  impl_->envelope->input_owner.reset();
  impl_->finished = true;
}
void NativeOutputBatch::FailAfterSubmit() noexcept
{
  if (!impl_ || impl_->finished) {
    return;
  }
  // Failed Managed writers orphan their state, retaining the typed attachment,
  // native writer, message and stream. Unknown GPU completion is never guessed.
  for (auto & writer : impl_->writers) {
    writer.fail();
  }
  impl_->finished = true;
}
void NativeOutputBatch::CopyFrom(const ManagedTensorBundle & source)
{
  if (source.tensors().size() != impl_->buffers.size()) {
    CancelBeforeSubmit();
    throw std::invalid_argument("native copy tensor count mismatch");
  }
  DeviceScope scope(impl_->envelope->device);
  bool submitted = false;
  try {
    impl_->envelope->copy_source = source;
    std::vector<BlockingReadyLease> leases;
    leases.reserve(source.tensors().size());
    for (size_t i = 0; i < source.tensors().size(); ++i) {
      const auto & tensor = source.tensors()[i];
      const auto spec = Spec(impl_->envelope->message->tensors[i]);
      if (tensor.name() != spec.name || tensor.data_type() != spec.dtype ||
          tensor.shape() != spec.shape)
      {
        throw std::invalid_argument("native copy metadata mismatch");
      }
      const void * pointer;
      cudaMemcpyKind kind;
      if (const auto * host = std::get_if<HostBuffer>(&tensor.storage())) {
        pointer = host->data();
        kind = cudaMemcpyHostToDevice;
      } else {
        const auto & buffer = std::get<std::shared_ptr<DeviceBuffer>>(tensor.storage());
        if (!buffer || buffer->device_id().backend != BackendKind::kCuda ||
            buffer->device_id().ordinal != impl_->envelope->device)
        {
          throw std::invalid_argument("native copy requires a host or matching CUDA tensor");
        }
        leases.push_back(buffer->get_blocking_ready_lease());
        pointer = leases.back().data();
        kind = cudaMemcpyDeviceToDevice;
      }
      submitted = true;
      cuda_buffer_backend::to_buffer(pointer, tensor.byte_size(), impl_->envelope->writers[i],
        impl_->envelope->stream.get(), kind);
    }
    Check(cudaStreamSynchronize(impl_->envelope->stream.get()));
    CompleteAfterSync();
    impl_->envelope->copy_source.reset();
  } catch (...) {
    if (submitted) {
      // A blocking synchronization retains source owners until every submitted
      // copy is done. If CUDA cannot establish completion, retain the source too.
      static_cast<void>(cudaStreamSynchronize(impl_->envelope->stream.get()));
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
  rclcpp::Publisher<Message>::SharedPtr publisher;
  rclcpp::Subscription<Message>::SharedPtr subscription;
  std::mutex callback_mutex;
  std::condition_variable callback_cv;
  size_t active_callbacks{0};
  bool stopping{false};
};
TensorListTransport::TensorListTransport(
  rclcpp::Node * node, int device_id, const std::string & output_topic, bool publish_output)
    : impl_(std::make_shared<Impl>())
{
  impl_->node = node;
  impl_->device = device_id;
  if (publish_output) {
    rclcpp::PublisherOptions options;
    // Publishing a shared immutable envelope through intra-process const&
    // publishing clones GPU Buffer fields. Use the RMW Buffer backend instead.
    options.use_intra_process_comm = rclcpp::IntraProcessSetting::Disable;
    impl_->publisher = node->create_publisher<Message>(output_topic, rclcpp::QoS(10), options);
  }
}
TensorListTransport::~TensorListTransport()
{
  Unsubscribe();
}
void TensorListTransport::Unsubscribe()
{
  {
    std::unique_lock<std::mutex> lock(impl_->callback_mutex);
    impl_->stopping = true;
    impl_->callback_cv.wait(lock, [this] { return impl_->active_callbacks == 0; });
  }
  impl_->subscription.reset();
}
void TensorListTransport::Subscribe(Callback callback, const std::string & input_topic)
{
  Unsubscribe();
  {
    std::lock_guard<std::mutex> lock(impl_->callback_mutex);
    impl_->stopping = false;
  }
  const int device = impl_->device;
  impl_->subscription = impl_->node->create_subscription<Message>(input_topic, rclcpp::QoS(10),
    [weak = std::weak_ptr<Impl>(impl_), device, logger = impl_->node->get_logger(),
      callback = std::move(callback)](Message::ConstSharedPtr message) {
      auto state = weak.lock();
      if (!state) {
        return;
      }
      {
        std::lock_guard<std::mutex> lock(state->callback_mutex);
        if (state->stopping) {
          return;
        }
        ++state->active_callbacks;
      }
      struct Guard
      {
        std::shared_ptr<Impl> state;
        ~Guard()
        {
          std::lock_guard<std::mutex> lock(state->callback_mutex);
          --state->active_callbacks;
          state->callback_cv.notify_all();
        }
      } guard{state};
      try {
        auto bundle = std::make_shared<ManagedTensorBundle>(
          detail::ImportTensorList(std::move(message), device));
        callback(ManagedTensorBundleView(std::move(bundle)));
      } catch (const std::exception & error) {
        RCLCPP_ERROR(logger, "Dropping native TensorList frame: %s", error.what());
      }
    });
}
NativeOutputBatch TensorListTransport::Allocate(
  const std_msgs::msg::Header & header, const std::vector<NativeTensorSpec> & specs)
{
  DeviceScope scope(impl_->device);
  auto state = std::make_unique<NativeOutputBatch::Impl>();
  state->envelope = std::make_shared<NativeTensorListEnvelope>(impl_->device);
  // CudaBufferImpl remembers the output stream for later host materialization.
  // A ROS consumer may retain the message after all Managed views are gone.
  // Retain only the stream in its deleter, not the envelope (which owns the message).
  auto message =
    std::shared_ptr<Message>(new Message, [stream = state->envelope->stream](Message * value) {
      static_cast<void>(stream);
      delete value;
    });
  message->header = header;
  message->tensors.reserve(specs.size());
  state->envelope->message = message;
  state->envelope->writers.reserve(specs.size());
  state->buffers.reserve(specs.size());
  state->writers.reserve(specs.size());
  state->pointers.reserve(specs.size());
  NativeOutputBatch batch(std::move(state));
  try {
    for (size_t i = 0; i < specs.size(); ++i) {
      const auto & spec = specs[i];
      const auto bytes = tensor_byte_size(spec.shape, spec.dtype);
      if (spec.shape.empty() || spec.shape.size() > std::numeric_limits<uint8_t>::max()) {
        throw std::invalid_argument("native output rank out of range");
      }
      auto & tensor = message->tensors.emplace_back();
      tensor.name = spec.name;
      tensor.data_type = static_cast<int32_t>(spec.dtype);
      tensor.shape.rank = static_cast<decltype(tensor.shape.rank)>(spec.shape.size());
      for (const auto dim : spec.shape) {
        if (dim <= 0 || static_cast<uint64_t>(dim) > std::numeric_limits<uint32_t>::max()) {
          throw std::invalid_argument("native output dimension out of range");
        }
        tensor.shape.dims.push_back(static_cast<uint32_t>(dim));
      }
      tensor.strides = contiguous_strides(spec.shape, spec.dtype);
      tensor.data = cuda_buffer_backend::allocate_buffer(bytes);
      auto & envelope = batch.impl_->envelope;
      envelope->writers.push_back(
        cuda_buffer_backend::from_output_buffer(tensor.data, envelope->stream.get()));
      auto * pointer = envelope->writers.back().get_ptr();
      auto buffer = cuda::adopt_external(pointer, bytes, impl_->device, envelope,
        std::make_shared<NativeAttachment>(envelope, i, pointer));
      batch.impl_->writers.push_back(buffer->get_synchronized_write_handle());
      if (batch.impl_->writers.back().data() != pointer) {
        throw std::runtime_error("native and Managed output pointers differ");
      }
      batch.impl_->buffers.push_back(std::move(buffer));
      batch.impl_->pointers.push_back(pointer);
    }
  } catch (...) {
    batch.CancelBeforeSubmit();
    throw;
  }
  return batch;
}
void TensorListTransport::Publish(const ManagedTensorBundle & bundle)
{
  if (!impl_->publisher) {
    throw std::logic_error("native publisher disabled");
  }
  if (auto envelope = Reusable(bundle, impl_->device)) {
    impl_->publisher->publish(*envelope->message);
    return;
  }
  std::vector<NativeTensorSpec> specs;
  specs.reserve(bundle.tensors().size());
  for (const auto & tensor : bundle.tensors()) {
    specs.push_back({tensor.name(), tensor.data_type(), tensor.shape()});
  }
  auto batch = Allocate(bundle.header(), specs);
  batch.CopyFrom(bundle);
  impl_->publisher->publish(*batch.impl_->envelope->message);
}
} // namespace gpu_ros::nvidia_tensor_bundle_compat
