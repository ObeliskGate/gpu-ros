// Copyright 2026 gpu_ros_managed contributors
// Licensed under the Apache License, Version 2.0.
#include "gpu_ros_nvidia_tensor_bundle_compat/tensor_list_buffer_adapter.hpp"
#include "tensor_list_buffer_adapter_detail.hpp"
#include "native_tensor_list_internal.hpp"

#include <optional>
#include <stdexcept>
#include <utility>
#include "gpu_ros_managed_cuda/cuda_backend.hpp"

namespace gpu_ros::nvidia_tensor_bundle_compat
{
namespace
{
using namespace gpu_ros_managed;
struct ScalarMapping
{
  TensorDataType dtype;
  uint8_t code;
  uint8_t bits;
};
constexpr ScalarMapping kScalarMappings[]{
  {TensorDataType::kInt8, 0, 8}, {TensorDataType::kUInt8, 1, 8},
  {TensorDataType::kInt16, 0, 16}, {TensorDataType::kUInt16, 1, 16},
  {TensorDataType::kInt32, 0, 32}, {TensorDataType::kUInt32, 1, 32},
  {TensorDataType::kInt64, 0, 64}, {TensorDataType::kUInt64, 1, 64},
  {TensorDataType::kFloat32, 2, 32}, {TensorDataType::kFloat64, 2, 64}};
TensorDataType ManagedType(const native::TensorSpec & spec)
{
  for (const auto & mapping : kScalarMappings) {
    if (spec.dtype_lanes == 1 && spec.dtype_code == mapping.code && spec.dtype_bits == mapping.bits) {
      return mapping.dtype;
    }
  }
  throw std::invalid_argument("TensorList has unsupported scalar dtype");
}
native::TensorSpec NativeSpec(const NativeTensorSpec & spec)
{
  for (const auto & mapping : kScalarMappings) {
    if (spec.dtype == mapping.dtype) {
      return {spec.name, mapping.code, mapping.bits, 1, spec.shape};
    }
  }
  throw std::invalid_argument("TensorList has unsupported project dtype");
}
struct NativeAttachment final : DeviceBufferAttachment
{
  NativeAttachment(native::TensorList value, size_t tensor_index, const void * data)
      : message(std::move(value)), index(tensor_index), device(message.device_id()), pointer(data) {}
  const native::TensorList message;
  const size_t index;
  const int device;
  const void * const pointer;
};
std::optional<native::TensorList> Reusable(const ManagedTensorBundle & bundle, int device)
{
  std::optional<native::TensorList> message;
  for (size_t i = 0; i < bundle.tensors().size(); ++i) {
    const auto & tensor = bundle.tensors()[i];
    const auto * buffer = std::get_if<std::shared_ptr<DeviceBuffer>>(&tensor.storage());
    if (!buffer || !*buffer) { return {}; }
    const auto attachment =
      std::dynamic_pointer_cast<const NativeAttachment>((*buffer)->attachment());
    if (!attachment || attachment->index != i || attachment->device != device ||
        (*buffer)->device_id().backend != BackendKind::kCuda ||
        (*buffer)->device_id().ordinal != device) {
      return {};
    }
    if (!message) { message = attachment->message; }
    if (!message->same_message(attachment->message) || message->header() != bundle.header() ||
        message->tensors().size() != bundle.tensors().size()) {
      return {};
    }
    const auto & spec = message->tensors().at(i);
    const auto dtype = ManagedType(spec);
    if (spec.name != tensor.name() || dtype != tensor.data_type() || spec.shape != tensor.shape() ||
        tensor.strides() != contiguous_strides(spec.shape, dtype)) {
      return {};
    }
    const auto lease = (*buffer)->get_blocking_ready_lease();
    if (lease.data() != attachment->pointer || lease.data() != message->data(i) ||
        lease.size() != tensor_byte_size(spec.shape, dtype)) {
      return {};
    }
  }
  return message;
}
ManagedTensorBundle Import(native::TensorList message)
{
  std::vector<ManagedTensor> tensors;
  tensors.reserve(message.tensors().size());
  for (size_t i = 0; i < message.tensors().size(); ++i) {
    const auto & spec = message.tensors()[i];
    const auto dtype = ManagedType(spec);
    auto attachment = std::make_shared<NativeAttachment>(message, i, message.data(i));
    auto buffer = cuda::adopt_synchronized_external(const_cast<uint8_t *>(message.data(i)),
      tensor_byte_size(spec.shape, dtype), message.device_id(),
      std::const_pointer_cast<void>(message.owner()), std::move(attachment));
    tensors.emplace_back(spec.name, dtype, spec.shape, std::move(buffer));
  }
  return ManagedTensorBundle(message.header(), std::move(tensors));
}
} // namespace
namespace detail
{
isaac_ros_tensor_msgs::msg::TensorList::ConstSharedPtr ReusableTensorList(
  const ManagedTensorBundle & bundle, int device)
{
  auto message = Reusable(bundle, device);
  return message ? native::detail::RawMessage(*message) : nullptr;
}
ManagedTensorBundle ImportTensorList(
  isaac_ros_tensor_msgs::msg::TensorList::ConstSharedPtr message, int device)
{
  auto stream = std::make_shared<cuda::CudaStream>(cuda::make_stream(device));
  return Import(native::detail::ImportTensorList(std::move(message), device, stream->get(), stream));
}
} // namespace detail

struct NativeOutputBatch::Impl
{
  explicit Impl(native::OutputBatch value) : batch(std::move(value)) {}
  native::OutputBatch batch;
  std::vector<std::shared_ptr<DeviceBuffer>> buffers;
  std::vector<SynchronizedWriteHandle> writers;
  bool finished{false};
  ~Impl()
  {
    if (!finished) {
      batch.FailAfterSubmit();
      for (auto & writer : writers) { writer.fail(); }
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
void * NativeOutputBatch::pointer(size_t index) const { return impl_->batch.data(index); }
void NativeOutputBatch::RetainOwner(std::shared_ptr<const void> owner)
{
  if (impl_->finished) { throw std::logic_error("native batch already completed"); }
  impl_->batch.RetainOwner(std::move(owner));
}
void NativeOutputBatch::CompleteAfterSync()
{
  if (impl_->finished) { throw std::logic_error("native batch already completed"); }
  impl_->batch.CompleteAfterSync();
  for (auto & writer : impl_->writers) { writer.finalize_synchronously(); }
  impl_->finished = true;
}
void NativeOutputBatch::CancelBeforeSubmit() noexcept
{
  if (!impl_ || impl_->finished) { return; }
  impl_->batch.CancelBeforeSubmit();
  for (auto & writer : impl_->writers) {
    try { writer.cancel(); }
    catch (...) { writer.fail(); }
  }
  impl_->finished = true;
}
void NativeOutputBatch::FailAfterSubmit() noexcept
{
  if (!impl_ || impl_->finished) { return; }
  impl_->batch.FailAfterSubmit();
  for (auto & writer : impl_->writers) { writer.fail(); }
  impl_->finished = true;
}
void NativeOutputBatch::CopyFrom(const ManagedTensorBundle & source)
{
  if (impl_->finished) { throw std::logic_error("native batch already completed"); }
  // Leases (not merely DeviceBuffer references) exclude reuse until copying ends.
  struct CopyOwner
  {
    explicit CopyOwner(const ManagedTensorBundle & value) : bundle(value) {}
    ManagedTensorBundle bundle;
    std::vector<BlockingReadyLease> leases;
  };
  bool copying = false;
  try {
    if (source.tensors().size() != impl_->buffers.size()) {
      throw std::invalid_argument("native copy tensor count mismatch");
    }
    auto owner = std::make_shared<CopyOwner>(source);
    owner->leases.reserve(source.tensors().size());
    std::vector<native::CopySource> sources;
    sources.reserve(source.tensors().size());
    const auto message = impl_->batch.message();
    for (size_t i = 0; i < source.tensors().size(); ++i) {
      const auto & tensor = source.tensors()[i];
      const auto & spec = impl_->batch.tensors()[i];
      if (tensor.name() != spec.name || tensor.data_type() != ManagedType(spec) ||
          tensor.shape() != spec.shape) {
        throw std::invalid_argument("native copy metadata mismatch");
      }
      if (const auto * host = std::get_if<HostBuffer>(&tensor.storage())) {
        sources.push_back({host->data(), tensor.byte_size(), false});
      } else {
        const auto & buffer = std::get<std::shared_ptr<DeviceBuffer>>(tensor.storage());
        if (!buffer || buffer->device_id().backend != BackendKind::kCuda ||
            buffer->device_id().ordinal != message.device_id()) {
          throw std::invalid_argument("native copy requires a host or matching CUDA tensor");
        }
        owner->leases.push_back(buffer->get_blocking_ready_lease());
        sources.push_back({owner->leases.back().data(), tensor.byte_size(), true});
      }
    }
    copying = true;
    impl_->batch.CopyFrom(sources, owner);
    for (auto & writer : impl_->writers) { writer.finalize_synchronously(); }
    impl_->finished = true;
  } catch (...) {
    if (copying) { FailAfterSubmit(); }
    else { CancelBeforeSubmit(); }
    throw;
  }
}
struct TensorListTransport::Impl
{
  Impl(rclcpp::Node * node, int device, const std::string & topic, bool publish)
      : transport(node, device, topic, publish), device(device) {}
  native::TensorListTransport transport;
  int device;
};
TensorListTransport::TensorListTransport(
  rclcpp::Node * node, int device_id, const std::string & output_topic, bool publish_output)
    : impl_(std::make_shared<Impl>(node, device_id, output_topic, publish_output)) {}
TensorListTransport::~TensorListTransport() { Unsubscribe(); }
void TensorListTransport::Unsubscribe() { impl_->transport.Unsubscribe(); }
void TensorListTransport::Subscribe(Callback callback, const std::string & input_topic)
{
  impl_->transport.Subscribe([callback = std::move(callback)](native::TensorList message) {
    auto bundle = std::make_shared<ManagedTensorBundle>(Import(std::move(message)));
    callback(ManagedTensorBundleView(std::move(bundle)));
  }, input_topic);
}
NativeOutputBatch TensorListTransport::Allocate(
  const std_msgs::msg::Header & header, const std::vector<NativeTensorSpec> & specs)
{
  std::vector<native::TensorSpec> native_specs;
  native_specs.reserve(specs.size());
  for (const auto & spec : specs) { native_specs.push_back(NativeSpec(spec)); }
  auto native_batch = impl_->transport.Allocate(header, native_specs);
  // Reserve downstream control storage before handing out any Managed writer.
  std::unique_ptr<NativeOutputBatch::Impl> state;
  try {
    state = std::make_unique<NativeOutputBatch::Impl>(std::move(native_batch));
    state->buffers.reserve(specs.size());
    state->writers.reserve(specs.size());
  } catch (...) {
    if (state) { state->batch.CancelBeforeSubmit(); state->finished = true; }
    else { native_batch.CancelBeforeSubmit(); }
    throw;
  }
  NativeOutputBatch batch(std::move(state));
  try {
    auto & output = batch.impl_->batch;
    const auto message = output.message();
    for (size_t i = 0; i < specs.size(); ++i) {
      auto * pointer = output.data(i);
      auto buffer = cuda::adopt_external(pointer, tensor_byte_size(specs[i].shape, specs[i].dtype),
        impl_->device, std::const_pointer_cast<void>(output.owner()),
        std::make_shared<NativeAttachment>(message, i, pointer));
      batch.impl_->writers.push_back(buffer->get_synchronized_write_handle());
      if (batch.impl_->writers.back().data() != pointer) {
        throw std::runtime_error("native and Managed output pointers differ");
      }
      batch.impl_->buffers.push_back(std::move(buffer));
    }
  } catch (...) {
    batch.CancelBeforeSubmit();
    throw;
  }
  return batch;
}
void TensorListTransport::Publish(const ManagedTensorBundle & bundle)
{
  if (auto message = Reusable(bundle, impl_->device)) {
    impl_->transport.Publish(*message);
    return;
  }
  std::vector<NativeTensorSpec> specs;
  specs.reserve(bundle.tensors().size());
  for (const auto & tensor : bundle.tensors()) {
    specs.push_back({tensor.name(), tensor.data_type(), tensor.shape()});
  }
  auto batch = Allocate(bundle.header(), specs);
  batch.CopyFrom(bundle);
  impl_->transport.Publish(batch.impl_->batch.message());
}
} // namespace gpu_ros::nvidia_tensor_bundle_compat
