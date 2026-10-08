// Copyright 2026 gpu_ros_managed contributors
// Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "gpu_ros_rosidl_buffer/cuda_buffer_access.hpp"
#include "unsupported_buffer.hpp"

namespace buffer_access = gpu_ros::rosidl_buffer;
namespace
{
void Check(cudaError_t status)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

struct Stream
{
  Stream() { Check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { static_cast<void>(cudaStreamDestroy(value)); }
  cudaStream_t value{};
};

struct Gate
{
  explicit Gate(cudaStream_t value) : stream(value)
  {
    Check(cudaLaunchHostFunc(stream, Wait, this));
  }
  ~Gate()
  {
    Release();
    static_cast<void>(cudaStreamSynchronize(stream));
  }
  void Release()
  {
    std::lock_guard<std::mutex> lock(mutex);
    released = true;
    cv.notify_all();
  }
  static void CUDART_CB Wait(void * opaque)
  {
    auto & gate = *static_cast<Gate *>(opaque);
    std::unique_lock<std::mutex> lock(gate.mutex);
    gate.cv.wait(lock, [&gate] { return gate.released; });
  }
  cudaStream_t stream;
  std::mutex mutex;
  std::condition_variable cv;
  bool released{false};
};

class RosidlBufferCuda : public ::testing::Test
{
protected:
  void SetUp() override
  {
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "Buffer CUDA acceptance requires a real CUDA device";
    }
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  }
  int device_count{};
};

TEST(RosidlBufferCudaFactory, RejectsNegativeDevice)
{
  EXPECT_THROW(buffer_access::CreateCudaBufferAccess(-1), std::invalid_argument);
}

TEST_F(RosidlBufferCuda, RetainsOfficialStorageAndReadHandleBeyondOriginalOwners)
{
  auto message = std::make_shared<rosidl::Buffer<uint8_t>>(
    cuda_buffer_backend::allocate_buffer(4));
  std::weak_ptr<const void> weak = message;
  Stream producer;
  const std::array<uint8_t, 4> expected{2, 4, 6, 8};
  const void * original = nullptr;
  {
    auto writer = cuda_buffer_backend::from_output_buffer(*message, producer.value);
    original = writer.get_ptr();
    cuda_buffer_backend::to_buffer(expected.data(), expected.size(), writer,
      producer.value, cudaMemcpyHostToDevice);
  }
  auto access = buffer_access::CreateCudaBufferAccess(0);
  auto lease = access->AcquireRead(*message, message);
  EXPECT_EQ(lease.data, original);
  EXPECT_EQ(lease.byte_count, expected.size());
  message.reset();
  access.reset();
  ASSERT_FALSE(weak.expired());
  Stream consumer;
  std::array<uint8_t, 4> actual{};
  ASSERT_EQ(cudaMemcpyAsync(actual.data(), lease.data, actual.size(),
      cudaMemcpyDeviceToHost, consumer.value), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(consumer.value), cudaSuccess);
  EXPECT_EQ(actual, expected);
  auto retained = lease;
  lease = {};
  EXPECT_FALSE(weak.expired());
  retained = {};
  EXPECT_TRUE(weak.expired());
}

TEST_F(RosidlBufferCuda, WaitsForProducerBeforeReturningToIndependentConsumer)
{
  auto message = std::make_shared<rosidl::Buffer<uint8_t>>(
    cuda_buffer_backend::allocate_buffer(4));
  auto access = buffer_access::CreateCudaBufferAccess(0);
  Stream producer;
  // Declare the future before the gate so exception unwinding opens the gate
  // before waiting for the asynchronous AcquireRead to finish.
  std::future<buffer_access::BufferReadLease> pending;
  Gate gate(producer.value);
  const void * original = nullptr;
  {
    auto writer = cuda_buffer_backend::from_output_buffer(*message, producer.value);
    original = writer.get_ptr();
    Check(cudaMemsetAsync(writer.get_ptr(), 0x2a, 4, producer.value));
  }
  std::promise<void> started;
  auto entered = started.get_future();
  pending = std::async(std::launch::async, [&] {
      started.set_value();
      return access->AcquireRead(*message, message);
    });
  entered.wait();
  EXPECT_EQ(pending.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
  gate.Release();
  auto lease = pending.get();
  EXPECT_EQ(lease.data, original);
  Stream consumer;
  std::array<uint8_t, 4> actual{};
  ASSERT_EQ(cudaMemcpyAsync(actual.data(), lease.data, actual.size(),
      cudaMemcpyDeviceToHost, consumer.value), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(consumer.value), cudaSuccess);
  EXPECT_EQ(actual, (std::array<uint8_t, 4>{0x2a, 0x2a, 0x2a, 0x2a}));
}

TEST_F(RosidlBufferCuda, RejectsCpuWithoutPromotingOrChangingStorage)
{
  auto access = buffer_access::CreateCudaBufferAccess(0);
  auto message = std::make_shared<rosidl::Buffer<uint8_t>>(4, uint8_t{7});
  const void * pointer = message->data();
  EXPECT_THROW(access->AcquireRead(*message, message), std::invalid_argument);
  EXPECT_EQ(message->get_backend_type(), "cpu");
  EXPECT_EQ(message->data(), pointer);
  for (const auto value : *message) {
    EXPECT_EQ(value, 7U);
  }
}

TEST_F(RosidlBufferCuda, RejectsUnknownAndImpersonatingBackendsWithoutCopying)
{
  auto access = buffer_access::CreateCudaBufferAccess(0);
  for (const char * backend : {"unsupported", "cuda"}) {
    auto attempts = std::make_shared<buffer_access::test::CopyAttempts>();
    auto message = std::make_shared<rosidl::Buffer<uint8_t>>(
      std::make_unique<buffer_access::test::UnsupportedBuffer>(backend, attempts));
    EXPECT_THROW(access->AcquireRead(*message, message), std::invalid_argument);
    EXPECT_EQ(attempts->host, 0U);
    EXPECT_EQ(attempts->clone, 0U);
  }
}

TEST_F(RosidlBufferCuda, RejectsMissingOwnerAndEmptyCudaStorage)
{
  auto access = buffer_access::CreateCudaBufferAccess(0);
  auto message = std::make_shared<rosidl::Buffer<uint8_t>>(
    cuda_buffer_backend::allocate_buffer(4));
  EXPECT_THROW(access->AcquireRead(*message, {}), std::invalid_argument);
  auto empty = std::make_shared<rosidl::Buffer<uint8_t>>(
    cuda_buffer_backend::allocate_buffer(0));
  EXPECT_THROW(access->AcquireRead(*empty, empty), std::invalid_argument);
  auto cpu_access = buffer_access::CreateCpuBufferAccess();
  EXPECT_THROW(cpu_access->AcquireRead(*message, message), std::invalid_argument);
}

TEST_F(RosidlBufferCuda, RejectsUnavailableDeviceOrdinal)
{
  EXPECT_THROW(buffer_access::CreateCudaBufferAccess(device_count), std::runtime_error);
  int current = -1;
  ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
  EXPECT_EQ(current, 0);
}

TEST_F(RosidlBufferCuda, RejectsStorageOnAnotherRealDeviceAndRestoresCallingDevice)
{
  if (device_count < 2) {
    GTEST_SKIP() << "Cross-device Buffer rejection requires two real CUDA devices";
  }
  auto message = std::make_shared<rosidl::Buffer<uint8_t>>(
    cuda_buffer_backend::allocate_buffer(4));
  auto other_access = buffer_access::CreateCudaBufferAccess(1);
  EXPECT_THROW(other_access->AcquireRead(*message, message), std::invalid_argument);
  int current = -1;
  ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
  EXPECT_EQ(current, 0);
}
}  // namespace
