// Copyright 2026 Boshen Chen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gpu_ros_nvidia_tensor_bundle_compat/tensor_bundle_conversion.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "gpu_ros_tensor_bundle_msgs/msg/tensor.hpp"

namespace compat = gpu_ros::nvidia_tensor_bundle_compat;

namespace
{
compat::NvidiaTensorList FloatMessage()
{
  compat::NvidiaTensorList message;
  message.names = {"values"};
  auto & tensor = message.tensors.emplace_back();
  tensor.dtype_code = 2;
  tensor.dtype_bits = 32;
  tensor.dtype_lanes = 1;
  tensor.shape = {2, 3};
  tensor.strides = {3, 1};
  tensor.data.resize(6 * sizeof(float));
  return message;
}
} // namespace

TEST(TensorBundleConversion, RoundTripPreservesNamesOrderHeaderAndPayload)
{
  compat::TensorBundle bundle;
  bundle.header.frame_id = "camera";
  bundle.header.stamp.sec = 42;
  const float values[] = {1.0F, 2.0F};
  for (const auto * name : {"second", "first"}) {
    auto & tensor = bundle.tensors.emplace_back();
    tensor.name = name;
    tensor.data_type = gpu_ros_tensor_bundle_msgs::msg::Tensor::FLOAT32;
    tensor.shape = {1, 2};
    tensor.data.resize(sizeof(values));
    std::memcpy(tensor.data.data(), values, sizeof(values));
  }

  const auto native = compat::ToNvidiaTensorList(bundle);
  EXPECT_EQ(native.names, (std::vector<std::string>{"second", "first"}));
  ASSERT_EQ(native.tensors.size(), 2U);
  for (const auto & tensor : native.tensors) {
    EXPECT_EQ(tensor.shape, (std::vector<int64_t>{1, 2}));
    EXPECT_EQ(tensor.strides, (std::vector<int64_t>{2, 1}));
    EXPECT_EQ(tensor.dtype_code, 2);
    EXPECT_EQ(tensor.dtype_bits, 32);
    EXPECT_EQ(tensor.dtype_lanes, 1);
    EXPECT_EQ(tensor.byte_offset, 0U);
    EXPECT_EQ(tensor.data.get_backend_type(), "cpu");
  }
  const auto restored = compat::ToTensorBundle(native);
  EXPECT_EQ(restored, bundle);
  for (const auto & tensor : restored.tensors) {
    EXPECT_EQ(tensor.data.get_backend_type(), "cpu");
  }
}

TEST(TensorBundleConversion, MapsEverySupportedScalarDtypeExplicitly)
{
  using Tensor = gpu_ros_tensor_bundle_msgs::msg::Tensor;
  struct Expected
  {
    uint8_t project;
    uint8_t code;
    uint8_t bits;
  };
  const Expected types[]{
    {Tensor::INT8, 0, 8}, {Tensor::UINT8, 1, 8},
    {Tensor::INT16, 0, 16}, {Tensor::UINT16, 1, 16},
    {Tensor::INT32, 0, 32}, {Tensor::UINT32, 1, 32},
    {Tensor::INT64, 0, 64}, {Tensor::UINT64, 1, 64},
    {Tensor::FLOAT32, 2, 32}, {Tensor::FLOAT64, 2, 64}};
  for (const auto & type : types) {
    SCOPED_TRACE(static_cast<int>(type.project));
    compat::TensorBundle bundle;
    auto & tensor = bundle.tensors.emplace_back();
    tensor.name = "scalar";
    tensor.data_type = type.project;
    tensor.shape = {1};
    tensor.data.resize(type.bits / 8);
    const auto native = compat::ToNvidiaTensorList(bundle);
    ASSERT_EQ(native.tensors.size(), 1U);
    EXPECT_EQ(native.tensors[0].dtype_code, type.code);
    EXPECT_EQ(native.tensors[0].dtype_bits, type.bits);
    EXPECT_EQ(native.tensors[0].dtype_lanes, 1);
    EXPECT_EQ(compat::ToTensorBundle(native), bundle);
  }
}

TEST(TensorBundleConversion, RejectsNamesTensorCountMismatch)
{
  auto message = FloatMessage();
  message.names.clear();
  EXPECT_THROW(compat::ToTensorBundle(message), std::invalid_argument);
  message.names = {"one", "two"};
  EXPECT_THROW(compat::ToTensorBundle(message), std::invalid_argument);
}

TEST(TensorBundleConversion, AcceptsInferredStridesButRejectsByteStridesAndViews)
{
  auto message = FloatMessage();
  auto & tensor = message.tensors[0];
  EXPECT_NO_THROW(compat::ToTensorBundle(message));
  tensor.strides.clear();
  EXPECT_NO_THROW(compat::ToTensorBundle(message));
  for (const auto & strides : std::vector<std::vector<int64_t>>{
      {12, 4}, {4, 1}, {3}, {3, -1}, {3, 0}}) {
    tensor.strides = strides;
    EXPECT_THROW(compat::ToTensorBundle(message), std::invalid_argument);
  }
  tensor.strides = {3, 1};
  tensor.byte_offset = sizeof(float);
  EXPECT_THROW(compat::ToTensorBundle(message), std::invalid_argument);
  tensor.byte_offset = 0;
  tensor.data.resize(7 * sizeof(float));
  EXPECT_THROW(compat::ToTensorBundle(message), std::invalid_argument);
}

TEST(TensorBundleConversion, RejectsUnknownDtypeWidthsAndVectorLanes)
{
  auto message = FloatMessage();
  auto & tensor = message.tensors[0];
  tensor.dtype_code = 255;
  EXPECT_THROW(compat::ToTensorBundle(message), std::invalid_argument);
  tensor.dtype_code = 2;
  tensor.dtype_bits = 16; // float16 is not in the project enum.
  EXPECT_THROW(compat::ToTensorBundle(message), std::invalid_argument);
  tensor.dtype_bits = 32;
  tensor.dtype_lanes = 0;
  EXPECT_THROW(compat::ToTensorBundle(message), std::invalid_argument);
  tensor.dtype_lanes = 2;
  EXPECT_THROW(compat::ToTensorBundle(message), std::invalid_argument);
  compat::TensorBundle bundle;
  auto & project = bundle.tensors.emplace_back();
  project.data_type = 255;
  project.shape = {1};
  project.data.resize(1);
  EXPECT_THROW(compat::ToNvidiaTensorList(bundle), std::invalid_argument);
}

TEST(TensorBundleConversion, RejectsInvalidShapeSizeAndOverflow)
{
  auto message = FloatMessage();
  auto & tensor = message.tensors[0];
  for (const auto & shape : std::vector<std::vector<int64_t>>{{}, {2, 0}, {2, -1}}) {
    tensor.shape = shape;
    EXPECT_THROW(compat::ToTensorBundle(message), std::invalid_argument);
  }
  tensor.shape = {2, 3};
  tensor.data.resize(23);
  EXPECT_THROW(compat::ToTensorBundle(message), std::invalid_argument);
  tensor.shape = {std::numeric_limits<int64_t>::max(), 4};
  EXPECT_THROW(compat::ToTensorBundle(message), std::overflow_error);
  compat::TensorBundle bundle;
  auto & huge = bundle.tensors.emplace_back();
  huge.data_type = gpu_ros_tensor_bundle_msgs::msg::Tensor::FLOAT32;
  huge.shape = {std::numeric_limits<int64_t>::max(), 4};
  EXPECT_THROW(compat::ToNvidiaTensorList(bundle), std::overflow_error);
  huge.shape = {std::numeric_limits<int64_t>::max()};
  EXPECT_THROW(compat::ToNvidiaTensorList(bundle), std::overflow_error);
  // Total bytes fit size_t on the target, but a DLPack stride cannot fit int64_t.
  huge.data_type = gpu_ros_tensor_bundle_msgs::msg::Tensor::UINT8;
  huge.shape = {1, std::numeric_limits<int64_t>::max(), 2};
  EXPECT_THROW(compat::ToNvidiaTensorList(bundle), std::overflow_error);
}
