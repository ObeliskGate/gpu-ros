// Copyright 2026 gpu_ros_managed contributors
// Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <type_traits>
#include "cuda_buffer/cuda_buffer_api.hpp"
#include "gpu_ros_nvidia_tensor_bundle_compat/native_tensor_list.hpp"
#include "../src/native_tensor_list_internal.hpp"
#include "../src/tensor_metadata.hpp"

namespace native = gpu_ros::nvidia_tensor_bundle_compat::native;
namespace metadata = gpu_ros::nvidia_tensor_bundle_compat::detail;
namespace
{
using Message = native::detail::Message;
static_assert(!std::is_default_constructible_v<native::TensorList>);
static_assert(!std::is_copy_constructible_v<native::OutputBatch>);
const native::TensorSpec kFloatSpec{"values", 2, 32, 1, {1, 4}};
class NativeTensorListTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      GTEST_SKIP() << "Native TensorList acceptance requires a real CUDA device";
    }
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    if (!rclcpp::ok()) { rclcpp::init(0, nullptr); }
    node = std::make_shared<rclcpp::Node>("pure_native_tensor_list_test");
  }
  void TearDown() override
  {
    node.reset();
    if (rclcpp::ok()) { rclcpp::shutdown(); }
  }
  std::shared_ptr<rclcpp::Node> node;
};
struct Stream
{
  Stream() { EXPECT_EQ(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), cudaSuccess); }
  ~Stream() { if (value) { static_cast<void>(cudaStreamDestroy(value)); } }
  cudaStream_t value{};
};
struct Event
{
  Event() { EXPECT_EQ(cudaEventCreateWithFlags(&value, cudaEventDisableTiming), cudaSuccess); }
  ~Event() { if (value) { static_cast<void>(cudaEventDestroy(value)); } }
  cudaEvent_t value{};
};
struct Gate
{
  explicit Gate(cudaStream_t value) : stream(value) {}
  cudaStream_t stream;
  std::mutex mutex;
  std::condition_variable cv;
  bool released{false};
  static void CUDART_CB Wait(void * opaque)
  {
    auto & gate = *static_cast<Gate *>(opaque);
    std::unique_lock<std::mutex> lock(gate.mutex);
    gate.cv.wait(lock, [&] { return gate.released; });
  }
  void Release()
  {
    std::lock_guard<std::mutex> lock(mutex);
    released = true;
    cv.notify_all();
  }
  ~Gate()
  {
    Release();
    static_cast<void>(cudaStreamSynchronize(stream));
  }
};
std::shared_ptr<Message> CpuMessage()
{
  auto message = std::make_shared<Message>();
  message->header.frame_id = "camera";
  message->header.stamp.sec = 42;
  message->names = {"values"};
  auto & tensor = message->tensors.emplace_back();
  metadata::SetTensorMetadata(tensor, 2, 32, 1, {1, 4});
  const float values[]{2, 4, 6, 8};
  tensor.data.resize(sizeof(values));
  std::memcpy(tensor.data.data(), values, sizeof(values));
  return message;
}
void ExpectValues(const void * pointer, const std::array<float, 4> & expected)
{
  std::array<float, 4> restored{};
  ASSERT_EQ(cudaMemcpy(restored.data(), pointer, sizeof(restored), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(restored, expected);
}
TEST_F(NativeTensorListTest, CpuPromotionPreservesMetadataHeaderAndRetainsInput)
{
  auto raw = CpuMessage();
  const auto header = raw->header;
  std::weak_ptr<const Message> weak = raw;
  {
    auto input = native::detail::ImportTensorList(raw, 0);
    raw.reset();
    ASSERT_FALSE(weak.expired());
    EXPECT_EQ(input.header(), header);
    EXPECT_EQ(input.device_id(), 0);
    ASSERT_EQ(input.tensors().size(), 1U);
    EXPECT_EQ(input.tensors()[0].name, "values");
    EXPECT_EQ(input.tensors()[0].shape, (std::vector<int64_t>{1, 4}));
    EXPECT_EQ(input.tensors()[0].dtype_code, 2);
    EXPECT_EQ(input.tensors()[0].dtype_bits, 32);
    EXPECT_EQ(input.tensors()[0].dtype_lanes, 1);
    cudaPointerAttributes attributes{};
    ASSERT_EQ(cudaPointerGetAttributes(&attributes, input.data(0)), cudaSuccess);
    EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);
    EXPECT_EQ(attributes.device, 0);
    ExpectValues(input.data(0), {2, 4, 6, 8});
    EXPECT_THROW(input.data(1), std::out_of_range);
  }
  EXPECT_TRUE(weak.expired());
}
TEST_F(NativeTensorListTest, DirectCudaInputHasBackendPointerWithoutPayloadClone)
{
  native::TensorListTransport transport(node.get(), 0, "unused", false);
  auto batch = transport.Allocate(CpuMessage()->header, {kFloatSpec});
  const float values[]{3, 5, 7, 9};
  void * pointer = batch.data(0);
  auto message = batch.message();
  EXPECT_THROW(message.data(0), std::logic_error);
  ASSERT_EQ(cudaMemcpy(pointer, values, sizeof(values), cudaMemcpyHostToDevice), cudaSuccess);
  batch.CompleteAfterSync();
  auto raw = native::detail::RawMessage(message);
  auto stream = std::make_shared<Stream>();
  auto read = cuda_buffer_backend::from_input_buffer(raw->tensors[0].data, stream->value);
  ASSERT_EQ(cudaStreamSynchronize(stream->value), cudaSuccess);
  auto input = native::detail::ImportTensorList(raw, 0, stream->value, stream);
  EXPECT_EQ(input.data(0), pointer);
  EXPECT_EQ(input.data(0), read.get_ptr());
  EXPECT_TRUE(input.same_message(message));
  ExpectValues(input.data(0), {3, 5, 7, 9});
  EXPECT_THROW(batch.CompleteAfterSync(), std::logic_error);
}
TEST_F(NativeTensorListTest, EverySupportedScalarTypeUsesDLPackNotWireOrdinals)
{
  native::TensorListTransport transport(node.get(), 0, "unused", false);
  struct Scalar { uint8_t code; uint8_t bits; };
  constexpr Scalar scalar_types[]{
    {0, 8}, {0, 16}, {0, 32}, {0, 64},
    {1, 8}, {1, 16}, {1, 32}, {1, 64},
    {2, 32}, {2, 64}};
  for (const auto & scalar : scalar_types) {
    const size_t bytes = 6 * (scalar.bits / 8);
    auto source = std::make_shared<std::vector<uint8_t>>(bytes, 0x2a);
    auto batch = transport.Allocate({}, {{"scalar", scalar.code, scalar.bits, 1, {2, 3}}});
    batch.CopyFrom({{source->data(), bytes, false}}, source);
    auto raw = native::detail::RawMessage(batch.message());
    EXPECT_EQ(raw->tensors[0].dtype_code, scalar.code);
    EXPECT_EQ(raw->tensors[0].dtype_bits, scalar.bits);
    EXPECT_EQ(raw->tensors[0].data.size(), bytes);
    auto imported = native::detail::ImportTensorList(raw, 0);
    std::vector<uint8_t> actual(bytes);
    ASSERT_EQ(cudaMemcpy(actual.data(), imported.data(0), bytes, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(actual, *source);
  }
}
TEST_F(NativeTensorListTest, InvalidMetadataIsRejectedBeforeBackendPromotion)
{
  auto message = CpuMessage();
  auto & tensor = message->tensors[0];
  tensor.strides = {16, 4};
  EXPECT_THROW(native::detail::ImportTensorList(message, 0), std::invalid_argument);
  tensor.strides = {4, 1};
  tensor.dtype_lanes = 2;
  EXPECT_THROW(native::detail::ImportTensorList(message, 0), std::invalid_argument);
  tensor.dtype_lanes = 1;
  tensor.dtype_code = 9;
  EXPECT_THROW(native::detail::ImportTensorList(message, 0), std::invalid_argument);
  tensor.dtype_code = 2;
  tensor.byte_offset = 4;
  EXPECT_THROW(native::detail::ImportTensorList(message, 0), std::invalid_argument);
  tensor.byte_offset = 0;
  tensor.data.resize(15);
  EXPECT_THROW(native::detail::ImportTensorList(message, 0), std::invalid_argument);
  tensor.data.resize(16);
  tensor.shape = {0, 4};
  EXPECT_THROW(native::detail::ImportTensorList(message, 0), std::invalid_argument);
  tensor.shape = {};
  EXPECT_THROW(native::detail::ImportTensorList(message, 0), std::invalid_argument);
  tensor.shape = {std::numeric_limits<int64_t>::max(), 4};
  EXPECT_THROW(native::detail::ImportTensorList(message, 0), std::overflow_error);
  tensor.shape = {1, 4};
  message->names.clear();
  EXPECT_THROW(native::detail::ImportTensorList(message, 0), std::invalid_argument);
}
TEST_F(NativeTensorListTest, OutputNamesAreValidatedBeforeAllocation)
{
  native::TensorListTransport transport(node.get(), 0, "unused", false);
  auto invalid = kFloatSpec;
  invalid.name.clear();
  EXPECT_THROW(transport.Allocate({}, {invalid}), std::invalid_argument);
  EXPECT_THROW(transport.Allocate({}, {kFloatSpec, kFloatSpec}), std::invalid_argument);
}
TEST_F(NativeTensorListTest, ReadyMessageAndBackendReadersOutliveTransport)
{
  std::optional<native::TensorList> retained;
  Message::ConstSharedPtr raw;
  std::weak_ptr<const void> state;
  {
    auto transport = std::make_unique<native::TensorListTransport>(node.get(), 0, "native_lifetime");
    auto batch = transport->Allocate({}, {kFloatSpec});
    auto values = std::make_shared<std::array<float, 4>>(std::array<float, 4>{2, 4, 6, 8});
    batch.CopyFrom({{values->data(), sizeof(*values), false}}, values);
    retained = batch.message();
    raw = native::detail::RawMessage(*retained);
    state = retained->owner();
    transport->Publish(*retained);
  }
  node.reset();
  ExpectValues(retained->data(0), {2, 4, 6, 8});
  retained.reset();
  EXPECT_TRUE(state.expired());
  // Raw ROS ownership alone must keep the allocation's original stream alive.
  Stream reader_stream;
  auto read = cuda_buffer_backend::from_input_buffer(raw->tensors[0].data, reader_stream.value);
  ASSERT_EQ(cudaStreamSynchronize(reader_stream.value), cudaSuccess);
  ExpectValues(read.get_ptr(), {2, 4, 6, 8});
}
TEST_F(NativeTensorListTest, D2DCopyIsExplicitAndReleasesSourceAfterCompletion)
{
  native::TensorListTransport transport(node.get(), 0, "unused", false);
  auto source = transport.Allocate({}, {kFloatSpec});
  const float values[]{1, 3, 5, 7};
  ASSERT_EQ(cudaMemcpy(source.data(0), values, sizeof(values), cudaMemcpyHostToDevice), cudaSuccess);
  source.CompleteAfterSync();
  auto destination = transport.Allocate({}, {kFloatSpec});
  auto owner = std::make_shared<native::TensorList>(source.message());
  std::weak_ptr<const void> weak = owner;
  const void * original = source.data(0);
  destination.CopyFrom({{original, sizeof(values), true}}, owner);
  owner.reset();
  EXPECT_TRUE(weak.expired());
  EXPECT_NE(destination.data(0), original);
  ExpectValues(destination.message().data(0), {1, 3, 5, 7});
}
TEST_F(NativeTensorListTest, CancelRejectsReadAndPublishAndReleasesOwners)
{
  native::TensorListTransport transport(node.get(), 0, "native_cancel");
  auto owner = std::make_shared<int>(7);
  std::weak_ptr<const void> weak = owner;
  std::weak_ptr<const void> storage;
  {
    auto batch = transport.Allocate({}, {kFloatSpec});
    batch.RetainOwner(owner);
    owner.reset();
    auto message = batch.message();
    storage = batch.owner();
    EXPECT_THROW(transport.Publish(message), std::logic_error);
    batch.CancelBeforeSubmit();
    EXPECT_TRUE(weak.expired());
    EXPECT_THROW(message.data(0), std::logic_error);
    EXPECT_THROW(transport.Publish(message), std::logic_error);
    EXPECT_THROW(batch.data(0), std::logic_error);
    batch.FailAfterSubmit();
  }
  EXPECT_TRUE(storage.expired());
  auto next = transport.Allocate({}, {kFloatSpec});
  next.CancelBeforeSubmit();
}
TEST_F(NativeTensorListTest, InvalidCopyCancelsWithoutSubmittingAndMovePreservesState)
{
  native::TensorListTransport transport(node.get(), 0, "unused", false);
  auto batch = transport.Allocate({}, {kFloatSpec});
  auto source = std::make_shared<std::array<float, 4>>();
  auto message = batch.message();
  EXPECT_THROW(batch.CopyFrom({{source->data(), 15, false}}, source), std::invalid_argument);
  EXPECT_THROW(message.data(0), std::logic_error);
  auto original = transport.Allocate({}, {kFloatSpec});
  void * pointer = original.data(0);
  auto moved = std::move(original);
  EXPECT_EQ(moved.data(0), pointer);
  EXPECT_THROW(original.data(0), std::logic_error);
  EXPECT_THROW(moved.data(1), std::out_of_range);
  moved.CancelBeforeSubmit();
}
TEST_F(NativeTensorListTest, SubmittedFailureRetainsOwnersAcrossRealPendingGpuWork)
{
  native::TensorListTransport transport(node.get(), 0, "unused", false);
  Stream work;
  Event finished;
  Gate gate(work.value);
  std::weak_ptr<const void> storage;
  auto source_owner = std::make_shared<int>(42);
  std::weak_ptr<const void> source = source_owner;
  {
    auto batch = transport.Allocate({}, {kFloatSpec});
    storage = batch.owner();
    batch.RetainOwner(source_owner);
    source_owner.reset();
    ASSERT_EQ(cudaLaunchHostFunc(work.value, Gate::Wait, &gate), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(batch.data(0), 0, 16, work.value), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(finished.value, work.value), cudaSuccess);
    EXPECT_EQ(cudaEventQuery(finished.value), cudaErrorNotReady);
    batch.FailAfterSubmit();
    EXPECT_THROW(batch.message().data(0), std::logic_error);
  }
  EXPECT_FALSE(storage.expired());
  EXPECT_FALSE(source.expired());
  EXPECT_THROW(transport.Allocate({}, {kFloatSpec}), std::logic_error);
  gate.Release();
  ASSERT_EQ(cudaEventSynchronize(finished.value), cudaSuccess);
  // This tests conservative unknown-completion state, not a fabricated driver fault.
  EXPECT_FALSE(storage.expired());
  EXPECT_FALSE(source.expired());
}
TEST_F(NativeTensorListTest, SynchronizedCompletionReleasesRetainedOwners)
{
  native::TensorListTransport transport(node.get(), 0, "unused", false);
  Stream work;
  Event finished;
  std::weak_ptr<const void> storage;
  auto owner = std::make_shared<int>(42);
  std::weak_ptr<const void> weak = owner;
  {
    auto batch = transport.Allocate({}, {kFloatSpec});
    storage = batch.owner();
    batch.RetainOwner(owner);
    owner.reset();
    ASSERT_EQ(cudaMemsetAsync(batch.data(0), 0, 16, work.value), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(finished.value, work.value), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(finished.value), cudaSuccess);
    batch.CompleteAfterSync();
    EXPECT_TRUE(weak.expired());
    ExpectValues(batch.message().data(0), {0, 0, 0, 0});
  }
  EXPECT_TRUE(storage.expired());
}
TEST_F(NativeTensorListTest, UnfinishedDestructorFailsSafeInsteadOfCancelling)
{
  native::TensorListTransport transport(node.get(), 0, "unused", false);
  std::weak_ptr<const void> storage;
  {
    auto batch = transport.Allocate({}, {kFloatSpec});
    storage = batch.owner();
  }
  EXPECT_FALSE(storage.expired());
  EXPECT_THROW(transport.Allocate({}, {kFloatSpec}), std::logic_error);
}
TEST_F(NativeTensorListTest, RmwSubscriptionUsesFacadeAndStopsAfterUnsubscribe)
{
  using namespace std::chrono_literals;
  native::TensorListTransport publisher(node.get(), 0, "pure_native_wire");
  auto receiver = std::make_unique<native::TensorListTransport>(node.get(), 0, "unused", false);
  std::optional<native::TensorList> received;
  size_t callbacks = 0;
  receiver->Subscribe([&](native::TensorList message) {
    received = message;
    ++callbacks;
  }, "pure_native_wire");
  auto batch = publisher.Allocate(CpuMessage()->header, {kFloatSpec});
  auto source = std::make_shared<std::array<float, 4>>(std::array<float, 4>{2, 4, 6, 8});
  batch.CopyFrom({{source->data(), sizeof(*source), false}}, source);
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (!received && std::chrono::steady_clock::now() < deadline) {
    publisher.Publish(batch.message());
    executor.spin_once(20ms);
  }
  ASSERT_TRUE(received);
  EXPECT_EQ(received->header(), batch.message().header());
  ExpectValues(received->data(0), {2, 4, 6, 8});
  receiver->Unsubscribe();
  const size_t stopped_count = callbacks;
  receiver.reset();
  publisher.Publish(batch.message());
  executor.spin_some();
  EXPECT_EQ(callbacks, stopped_count);
  ExpectValues(received->data(0), {2, 4, 6, 8});
}
} // namespace
