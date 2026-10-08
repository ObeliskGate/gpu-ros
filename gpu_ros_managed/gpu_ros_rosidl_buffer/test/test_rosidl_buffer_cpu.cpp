// Copyright 2026 gpu_ros_managed contributors
// Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include "gpu_ros_rosidl_buffer/buffer_access.hpp"
#include "unsupported_buffer.hpp"

namespace buffer_access = gpu_ros::rosidl_buffer;
namespace
{
struct Message
{
  rosidl::Buffer<uint8_t> data{2, 4, 6, 8};
};
static_assert(std::has_virtual_destructor_v<buffer_access::IBufferAccess>);

TEST(RosidlBufferCpu, AliasesOriginalStorageAndRetainsMessageAfterAccessDestruction)
{
  auto access = buffer_access::CreateCpuBufferAccess();
  auto message = std::make_shared<Message>();
  std::weak_ptr<const Message> weak = message;
  const auto * pointer = message->data.data();
  auto lease = access->AcquireRead(message->data, message);
  EXPECT_EQ(lease.data, pointer);
  EXPECT_EQ(lease.byte_count, 4U);
  EXPECT_EQ(lease.owner.get(), message.get());
  EXPECT_FALSE(lease.owner.owner_before(message));
  EXPECT_FALSE(message.owner_before(lease.owner));
  auto retained = lease;
  message.reset();
  access.reset();
  lease = {};
  ASSERT_FALSE(weak.expired());
  const auto * actual = static_cast<const uint8_t *>(retained.data);
  const std::array<uint8_t, 4> expected{2, 4, 6, 8};
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual[i], expected[i]);
  }
  retained = {};
  EXPECT_TRUE(weak.expired());
}

TEST(RosidlBufferCpu, EmptyBufferStillRetainsItsOwner)
{
  auto access = buffer_access::CreateCpuBufferAccess();
  auto buffer = std::make_shared<rosidl::Buffer<uint8_t>>();
  std::weak_ptr<const void> weak = buffer;
  auto lease = access->AcquireRead(*buffer, buffer);
  EXPECT_EQ(lease.byte_count, 0U);
  EXPECT_EQ(lease.data, buffer->data());
  buffer.reset();
  EXPECT_FALSE(weak.expired());
  lease = {};
  EXPECT_TRUE(weak.expired());
}

TEST(RosidlBufferCpu, RejectsMissingMessageOwner)
{
  auto access = buffer_access::CreateCpuBufferAccess();
  const Message message;
  EXPECT_THROW(access->AcquireRead(message.data, {}), std::invalid_argument);
}

TEST(RosidlBufferCpu, RejectsOtherAndImpersonatingBackendsWithoutMaterialization)
{
  auto access = buffer_access::CreateCpuBufferAccess();
  for (const char * backend : {"unsupported", "cuda", "cpu"}) {
    auto attempts = std::make_shared<buffer_access::test::CopyAttempts>();
    auto buffer = std::make_shared<rosidl::Buffer<uint8_t>>(
      std::make_unique<buffer_access::test::UnsupportedBuffer>(backend, attempts));
    EXPECT_THROW(access->AcquireRead(*buffer, buffer), std::invalid_argument);
    EXPECT_EQ(attempts->host, 0U);
    EXPECT_EQ(attempts->clone, 0U);
  }
}
}  // namespace
