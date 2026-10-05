// Copyright 2026 gpu_ros_managed contributors
// Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include "cuda_buffer/cuda_buffer_api.hpp"
#include "gpu_ros_managed_cuda/cuda_backend.hpp"
#include "gpu_ros_nvidia_tensor_bundle_compat/tensor_bundle_conversion.hpp"
#include "gpu_ros_nvidia_tensor_bundle_compat/tensor_list_buffer_adapter.hpp"
#include "../src/tensor_list_buffer_adapter_detail.hpp"

namespace compat = gpu_ros::nvidia_tensor_bundle_compat;
using namespace gpu_ros_managed;
using namespace std::chrono_literals;
namespace
{
class NativeAdapterTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    int count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
    ASSERT_GT(count, 0) << "Native adapter acceptance requires a real CUDA device";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }
    node = std::make_shared<rclcpp::Node>("native_adapter_test");
  }
  void TearDown() override
  {
    node.reset();
    EXPECT_TRUE(cuda::wait_for_pending_releases(5s));
  }
  std::shared_ptr<rclcpp::Node> node;
};
ManagedTensorBundle Bundle(
  const std_msgs::msg::Header & header, const std::shared_ptr<DeviceBuffer> & buffer)
{
  return ManagedTensorBundle(
    header, {ManagedTensor("values", TensorDataType::kFloat32, {1, 4}, buffer)});
}
TEST_F(NativeAdapterTest, NativeManagedRoundTripPreservesEnvelopePayloadAndLocalPointer)
{
  compat::TensorListTransport transport(node.get(), 0, "unused", false);
  std_msgs::msg::Header header;
  header.frame_id = "camera";
  header.stamp.sec = 42;
  auto batch = transport.Allocate(header, {{"values", TensorDataType::kFloat32, {1, 4}}});
  const float values[]{1, 2, 3, 4};
  ASSERT_EQ(
    cudaMemcpy(batch.pointer(0), values, sizeof(values), cudaMemcpyHostToDevice), cudaSuccess);
  batch.CompleteAfterSync();
  auto managed = Bundle(header, batch.buffers()[0]);
  auto message = compat::detail::ReusableTensorList(managed, 0);
  ASSERT_TRUE(message);
  auto stream = cuda::make_stream(0);
  auto native_read = cuda_buffer_backend::from_input_buffer(message->tensors[0].data, stream.get());
  EXPECT_EQ(native_read.get_ptr(), batch.pointer(0));
  auto imported = compat::detail::ImportTensorList(message, 0);
  EXPECT_EQ(compat::detail::ReusableTensorList(imported, 0), message);
  auto buffer = std::get<std::shared_ptr<DeviceBuffer>>(imported.tensors()[0].storage());
  EXPECT_EQ(buffer->get_blocking_ready_lease().data(), batch.pointer(0));
  const auto host = compat::ToTensorBundle(*message);
  EXPECT_EQ(host.header, header);
  ASSERT_EQ(host.tensors[0].data.get_backend_type(), "cpu");
  EXPECT_EQ(std::memcmp(host.tensors[0].data.data(), values, sizeof(values)), 0);
}
TEST_F(NativeAdapterTest, HeaderChangeAndUnattachedInputUseExplicitCopy)
{
  compat::TensorListTransport transport(node.get(), 0, "unused", false);
  auto source = cuda::allocate(16, 0);
  const float values[]{5, 6, 7, 8};
  source->copy_from_host_blocking(values, sizeof(values));
  std_msgs::msg::Header header;
  auto original = Bundle(header, source);
  EXPECT_FALSE(compat::detail::ReusableTensorList(original, 0));
  auto batch = transport.Allocate(header, {{"values", TensorDataType::kFloat32, {1, 4}}});
  batch.CopyFrom(original);
  EXPECT_NE(batch.pointer(0), source->get_blocking_ready_lease().data());
  auto copied = Bundle(header, batch.buffers()[0]);
  auto message = compat::detail::ReusableTensorList(copied, 0);
  ASSERT_TRUE(message);
  header.frame_id = "changed";
  auto changed = Bundle(header, batch.buffers()[0]);
  EXPECT_FALSE(compat::detail::ReusableTensorList(changed, 0));
  auto replacement = transport.Allocate(header, {{"values", TensorDataType::kFloat32, {1, 4}}});
  replacement.CopyFrom(changed);
  auto replacement_message =
    compat::detail::ReusableTensorList(Bundle(header, replacement.buffers()[0]), 0);
  ASSERT_TRUE(replacement_message);
  EXPECT_NE(replacement_message, message);
  const auto host = compat::ToTensorBundle(*replacement_message);
  EXPECT_EQ(host.header, header);
  EXPECT_EQ(std::memcmp(host.tensors[0].data.data(), values, sizeof(values)), 0);
}
TEST_F(NativeAdapterTest, CopiedAttachmentCannotSubstituteAnotherAllocation)
{
  compat::TensorListTransport transport(node.get(), 0, "unused", false);
  auto original = transport.Allocate({}, {{"values", TensorDataType::kFloat32, {1, 4}}});
  ASSERT_EQ(cudaMemset(original.pointer(0), 0, 16), cudaSuccess);
  original.CompleteAfterSync();
  const float values[]{9, 8, 7, 6};
  auto source = cuda::allocate(sizeof(values), 0);
  source->copy_from_host_blocking(values, sizeof(values));
  auto lease = source->get_blocking_ready_lease();
  auto substituted = cuda::adopt_synchronized_external(const_cast<uint8_t *>(lease.data()),
    lease.size(), 0, source, original.buffers()[0]->attachment());
  auto bundle = Bundle({}, substituted);
  EXPECT_FALSE(compat::detail::ReusableTensorList(bundle, 0));
  auto copied = transport.Allocate({}, {{"values", TensorDataType::kFloat32, {1, 4}}});
  copied.CopyFrom(bundle);
  auto message = compat::detail::ReusableTensorList(Bundle({}, copied.buffers()[0]), 0);
  ASSERT_TRUE(message);
  const auto host = compat::ToTensorBundle(*message);
  EXPECT_EQ(std::memcmp(host.tensors[0].data.data(), values, sizeof(values)), 0);
}
TEST_F(NativeAdapterTest, CpuPromotionAndHostConversionHaveExactContent)
{
  auto message = std::make_shared<compat::NvidiaTensorList>();
  auto & tensor = message->tensors.emplace_back();
  tensor.name = "values";
  tensor.data_type = 9;
  tensor.shape.rank = 2;
  tensor.shape.dims = {1, 4};
  tensor.strides = {16, 4};
  const float values[]{1, 3, 5, 7};
  tensor.data.resize(sizeof(values));
  std::memcpy(tensor.data.data(), values, sizeof(values));
  auto managed = compat::detail::ImportTensorList(message, 0);
  auto buffer = std::get<std::shared_ptr<DeviceBuffer>>(managed.tensors()[0].storage());
  float restored[4]{};
  ASSERT_EQ(cudaMemcpy(restored, buffer->get_blocking_ready_lease().data(), sizeof(restored),
              cudaMemcpyDeviceToHost),
    cudaSuccess);
  EXPECT_EQ(std::memcmp(restored, values, sizeof(values)), 0);
}
struct Gate
{
  std::mutex mutex;
  std::condition_variable cv;
  bool released{false};
  static void CUDART_CB Wait(void * pointer)
  {
    auto & gate = *static_cast<Gate *>(pointer);
    std::unique_lock<std::mutex> lock(gate.mutex);
    gate.cv.wait(lock, [&] { return gate.released; });
  }
  void Release()
  {
    std::lock_guard<std::mutex> lock(mutex);
    released = true;
    cv.notify_all();
  }
  ~Gate() { Release(); }
};
TEST_F(NativeAdapterTest, MultipleReadersRetainEnvelopeAfterPublisherDestruction)
{
  auto transport = std::make_unique<compat::TensorListTransport>(node.get(), 0, "unused", false);
  auto batch = transport->Allocate({}, {{"values", TensorDataType::kFloat32, {1, 4}}});
  const float values[]{2, 4, 6, 8};
  ASSERT_EQ(
    cudaMemcpy(batch.pointer(0), values, sizeof(values), cudaMemcpyHostToDevice), cudaSuccess);
  batch.CompleteAfterSync();
  auto message = compat::detail::ReusableTensorList(Bundle({}, batch.buffers()[0]), 0);
  std::weak_ptr<const compat::NvidiaTensorList> weak = message;
  auto imported =
    std::make_unique<ManagedTensorBundle>(compat::detail::ImportTensorList(message, 0));
  auto buffer = std::get<std::shared_ptr<DeviceBuffer>>(imported->tensors()[0].storage());
  auto early = cuda::make_stream(0);
  auto late = cuda::make_stream(0);
  float * restored = nullptr;
  ASSERT_EQ(cudaMallocHost(reinterpret_cast<void **>(&restored), sizeof(values)), cudaSuccess);
  auto pinned = std::shared_ptr<float>(restored, [stream = late.get()](float * pointer) {
    static_cast<void>(cudaStreamSynchronize(stream));
    static_cast<void>(cudaFreeHost(pointer));
  });
  Gate gate;
  {
    auto reader = buffer->get_read_handle(early);
    ASSERT_EQ(cudaStreamSynchronize(early.get()), cudaSuccess);
    reader.finish();
  }
  {
    auto reader = buffer->get_read_handle(late);
    ASSERT_EQ(cudaLaunchHostFunc(late.get(), Gate::Wait, &gate), cudaSuccess);
    ASSERT_EQ(
      cudaMemcpyAsync(restored, reader.data(), sizeof(values), cudaMemcpyDeviceToHost, late.get()),
      cudaSuccess);
    reader.finish();
  }
  // Also release the allocation transaction, otherwise it is itself an owner.
  batch = transport->Allocate({}, {});
  batch.CancelBeforeSubmit();
  transport.reset();
  node.reset();
  message.reset();
  imported.reset();
  buffer.reset();
  EXPECT_FALSE(weak.expired());
  gate.Release();
  ASSERT_EQ(cudaStreamSynchronize(late.get()), cudaSuccess);
  EXPECT_TRUE(cuda::wait_for_pending_releases(5s));
  EXPECT_TRUE(weak.expired());
  EXPECT_EQ(std::memcmp(restored, values, sizeof(values)), 0);
}
TEST_F(NativeAdapterTest, CancelReclaimsAndSubmittedFailureRetainsAttachment)
{
  compat::TensorListTransport transport(node.get(), 0, "unused", false);
  std::weak_ptr<const DeviceBufferAttachment> cancelled;
  {
    auto batch = transport.Allocate({}, {{"values", TensorDataType::kFloat32, {1, 4}}});
    cancelled = batch.buffers()[0]->attachment();
    batch.CancelBeforeSubmit();
    auto writer = batch.buffers()[0]->get_synchronized_write_handle();
    writer.cancel();
  }
  EXPECT_TRUE(cuda::wait_for_pending_releases(5s));
  EXPECT_TRUE(cancelled.expired());
  std::weak_ptr<const DeviceBufferAttachment> failed;
  {
    auto batch = transport.Allocate({}, {{"values", TensorDataType::kFloat32, {1, 4}}});
    failed = batch.buffers()[0]->attachment();
    batch.FailAfterSubmit();
    EXPECT_TRUE(batch.buffers()[0]->failed());
  }
  EXPECT_FALSE(failed.expired());
}

TEST_F(NativeAdapterTest, ReorderedFilteredMixedAndReshapedTensorsRemainConvertible)
{
  compat::TensorListTransport transport(node.get(), 0, "unused", false);
  const float a[]{1, 2, 3, 4};
  const float b[]{5, 6, 7, 8};
  ManagedTensorBundle host(
    {}, {ManagedTensor::from_host_copy("a", TensorDataType::kFloat32, {1, 4}, a, sizeof(a)),
          ManagedTensor::from_host_copy("b", TensorDataType::kFloat32, {1, 4}, b, sizeof(b))});
  auto first = transport.Allocate(
    {}, {{"a", TensorDataType::kFloat32, {1, 4}}, {"b", TensorDataType::kFloat32, {1, 4}}});
  first.CopyFrom(host);
  auto second = transport.Allocate(
    {}, {{"a", TensorDataType::kFloat32, {1, 4}}, {"b", TensorDataType::kFloat32, {1, 4}}});
  second.CopyFrom(host);
  const std::vector<ManagedTensorBundle> variants{
    ManagedTensorBundle(
      {}, {ManagedTensor("b", TensorDataType::kFloat32, {1, 4}, first.buffers()[1]),
            ManagedTensor("a", TensorDataType::kFloat32, {1, 4}, first.buffers()[0])}),
    ManagedTensorBundle(
      {}, {ManagedTensor("a", TensorDataType::kFloat32, {1, 4}, first.buffers()[0])}),
    ManagedTensorBundle(
      {}, {ManagedTensor("a", TensorDataType::kFloat32, {1, 4}, first.buffers()[0]),
            ManagedTensor("b", TensorDataType::kFloat32, {1, 4}, second.buffers()[1])}),
    ManagedTensorBundle(
      {}, {ManagedTensor("a", TensorDataType::kFloat32, {2, 2}, first.buffers()[0])})};
  for (const auto & input : variants) {
    EXPECT_FALSE(compat::detail::ReusableTensorList(input, 0));
    std::vector<compat::NativeTensorSpec> specs;
    for (const auto & tensor : input.tensors()) {
      specs.push_back({tensor.name(), tensor.data_type(), tensor.shape()});
    }
    auto replacement = transport.Allocate(input.header(), specs);
    replacement.CopyFrom(input);
    std::vector<ManagedTensor> tensors;
    for (size_t i = 0; i < specs.size(); ++i) {
      tensors.emplace_back(specs[i].name, specs[i].dtype, specs[i].shape, replacement.buffers()[i]);
    }
    auto message = compat::detail::ReusableTensorList(
      ManagedTensorBundle(input.header(), std::move(tensors)), 0);
    ASSERT_TRUE(message);
    const auto materialized = compat::ToTensorBundle(*message);
    for (const auto & tensor : materialized.tensors) {
      EXPECT_EQ(std::memcmp(tensor.data.data(), tensor.name == "a" ? a : b, sizeof(a)), 0);
    }
  }
}

TEST_F(NativeAdapterTest, StandardConversionMaterializesProjectCudaPayload)
{
  compat::TensorBundle input;
  input.header.frame_id = "host_boundary";
  auto & tensor = input.tensors.emplace_back();
  tensor.name = "values";
  tensor.data_type = 9;
  tensor.shape = {1, 4};
  tensor.data = cuda_buffer_backend::allocate_buffer(16);
  auto stream = cuda::make_stream(0);
  const float values[]{1, 2, 3, 4};
  {
    auto writer = cuda_buffer_backend::from_output_buffer(tensor.data, stream.get());
    cuda_buffer_backend::to_buffer(
      values, sizeof(values), writer, stream.get(), cudaMemcpyHostToDevice);
  }
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  const auto output = compat::ToNvidiaTensorList(input);
  EXPECT_EQ(output.header, input.header);
  EXPECT_EQ(output.tensors[0].data.get_backend_type(), "cpu");
  EXPECT_EQ(std::memcmp(output.tensors[0].data.data(), values, sizeof(values)), 0);
}

TEST_F(NativeAdapterTest, RmwConsumerReadsPublishedNativeOutput)
{
  compat::TensorListTransport transport(node.get(), 0, "native_output");
  std_msgs::msg::Header header;
  header.stamp.sec = 19;
  header.frame_id = "wire";
  const float values[]{9, 8, 7, 6};
  auto batch = transport.Allocate(header, {{"values", TensorDataType::kFloat32, {1, 4}}});
  ASSERT_EQ(
    cudaMemcpy(batch.pointer(0), values, sizeof(values), cudaMemcpyHostToDevice), cudaSuccess);
  batch.CompleteAfterSync();
  auto bundle = Bundle(header, batch.buffers()[0]);
  compat::NvidiaTensorList::ConstSharedPtr received;
  auto subscriber = node->create_subscription<compat::NvidiaTensorList>("native_output",
    rclcpp::QoS(10), [&received](compat::NvidiaTensorList::ConstSharedPtr message) {
      received = std::move(message);
    });
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (!received && std::chrono::steady_clock::now() < deadline) {
    transport.Publish(bundle);
    executor.spin_once(20ms);
  }
  ASSERT_TRUE(received);
  const auto host = compat::ToTensorBundle(*received);
  EXPECT_EQ(host.header, header);
  ASSERT_EQ(host.tensors.size(), 1U);
  EXPECT_EQ(std::memcmp(host.tensors[0].data.data(), values, sizeof(values)), 0);
}

TEST_F(NativeAdapterTest, NativeMessageRetainsMaterializationStreamWithoutManagedViews)
{
  compat::NvidiaTensorList::ConstSharedPtr message;
  const float values[]{4, 3, 2, 1};
  {
    compat::TensorListTransport transport(node.get(), 0, "unused", false);
    auto batch = transport.Allocate({}, {{"values", TensorDataType::kFloat32, {1, 4}}});
    ASSERT_EQ(
      cudaMemcpy(batch.pointer(0), values, sizeof(values), cudaMemcpyHostToDevice), cudaSuccess);
    batch.CompleteAfterSync();
    message = compat::detail::ReusableTensorList(Bundle({}, batch.buffers()[0]), 0);
  }
  node.reset();
  ASSERT_TRUE(message);
  const auto host = compat::ToTensorBundle(*message);
  EXPECT_EQ(std::memcmp(host.tensors[0].data.data(), values, sizeof(values)), 0);
}
} // namespace
