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
#include <vector>

#include <gtest/gtest.h>

#include "gpu_ros_tensor_bundle_msgs/msg/tensor.hpp"

namespace compat = gpu_ros::nvidia_tensor_bundle_compat;

TEST(TensorBundleConversion, RoundTripPreservesContiguousPayload)
{
  compat::TensorBundle bundle;
  bundle.header.frame_id = "camera";
  auto & tensor = bundle.tensors.emplace_back();
  tensor.name = "input";
  tensor.data_type = gpu_ros_tensor_bundle_msgs::msg::Tensor::FLOAT32;
  tensor.shape = {1, 2};
  tensor.data.resize(2U * sizeof(float));
  const float values[] = {1.0F, 2.0F};
  std::memcpy(tensor.data.data(), values, sizeof(values));

  const auto old = compat::ToNvidiaTensorList(bundle);
  ASSERT_EQ(old.tensors.size(), 1U);
  EXPECT_EQ(old.tensors[0].shape.rank, 2U);
  EXPECT_EQ(old.tensors[0].shape.dims, (std::vector<uint32_t>{1, 2}));
  EXPECT_EQ(old.tensors[0].strides, (std::vector<uint64_t>{8, 4}));

  const auto restored = compat::ToTensorBundle(old);
  EXPECT_EQ(restored.header.frame_id, "camera");
  ASSERT_EQ(restored.tensors.size(), 1U);
  EXPECT_EQ(restored.tensors[0].shape, (std::vector<int64_t>{1, 2}));
  EXPECT_EQ(restored.tensors[0].data_type, gpu_ros_tensor_bundle_msgs::msg::Tensor::FLOAT32);
  EXPECT_EQ(restored.tensors[0].data, tensor.data);
}

TEST(TensorBundleConversion, RejectsNonContiguousOldTensor)
{
  compat::NvidiaTensorList old;
  auto & tensor = old.tensors.emplace_back();
  tensor.name = "strided";
  tensor.data_type = 9;
  tensor.shape.rank = 2;
  tensor.shape.dims = {1, 2};
  tensor.strides = {16, 4};
  tensor.data.resize(8);

  EXPECT_THROW(compat::ToTensorBundle(old), std::invalid_argument);
}
