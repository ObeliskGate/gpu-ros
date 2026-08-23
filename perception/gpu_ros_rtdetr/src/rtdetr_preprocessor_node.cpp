// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
//
// SPDX-License-Identifier: Apache-2.0
// Modified from NVIDIA Isaac ROS RT-DETR preprocessor sources for standard ROS
// 2 integration; see THIRD_PARTY_NOTICES.md for the exact pinned revision.

#include "gpu_ros_rtdetr/rtdetr_preprocessor_node.hpp"
#include "gpu_ros_tensor_bundle_msgs/msg/tensor.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "rclcpp_components/register_node_macro.hpp"

namespace gpu_ros::rtdetr
{

namespace
{
using Tensor = gpu_ros_tensor_bundle_msgs::msg::Tensor;
}  // namespace

RtDetrPreprocessorNode::RtDetrPreprocessorNode(const rclcpp::NodeOptions options)
: rclcpp::Node("rtdetr_preprocessor_node", options),
  input_image_tensor_name_{declare_parameter<std::string>(
      "input_image_tensor_name", "input_tensor")},
  output_image_tensor_name_{declare_parameter<std::string>(
      "output_image_tensor_name", "images")},
  output_size_tensor_name_{declare_parameter<std::string>(
      "output_size_tensor_name", "orig_target_sizes")},
  image_height_{declare_parameter<int64_t>("image_height", 480)},
  image_width_{declare_parameter<int64_t>("image_width", 640)},
  use_max_dim_for_orig_size_{declare_parameter<bool>("use_max_dim_for_orig_size", true)}
{
  pub_ = create_publisher<TensorBundle>("tensor_pub", 10);
  sub_ = create_subscription<TensorBundle>(
    "encoded_tensor", 10,
    std::bind(&RtDetrPreprocessorNode::InputCallback, this, std::placeholders::_1));
}

void RtDetrPreprocessorNode::InputCallback(TensorBundle::UniquePtr msg)
{
  // Locate the encoded image tensor by name.
  size_t image_tensor_index = msg->tensors.size();
  for (size_t i = 0; i < msg->tensors.size(); ++i) {
    if (msg->tensors[i].name == input_image_tensor_name_) {
      image_tensor_index = i;
      break;
    }
  }
  if (image_tensor_index == msg->tensors.size()) {
    RCLCPP_WARN(
      get_logger(), "Input tensor '%s' not found; dropping message.",
      input_image_tensor_name_.c_str());
    return;
  }

  const int64_t orig_width = use_max_dim_for_orig_size_ ?
    std::max(image_height_, image_width_) : image_width_;
  const int64_t orig_height = use_max_dim_for_orig_size_ ?
    std::max(image_height_, image_width_) : image_height_;
  const int64_t output_size[2]{orig_width, orig_height};

  auto loaned_output = pub_->borrow_loaned_message();
  auto & out_msg = loaned_output.get();
  out_msg.header = std::move(msg->header);

  // Forward the image tensor unchanged, renamed to the model's input binding.
  auto image_out = std::move(msg->tensors[image_tensor_index]);
  image_out.name = output_image_tensor_name_;
  out_msg.tensors.push_back(std::move(image_out));

  // Append the orig_target_sizes int64 [1, 2] tensor.
  Tensor size_out;
  size_out.name = output_size_tensor_name_;
  size_out.data_type = Tensor::INT64;
  size_out.shape = {1, 2};
  size_out.data.resize(sizeof(output_size));
  std::memcpy(size_out.data.data(), output_size, sizeof(output_size));
  out_msg.tensors.push_back(std::move(size_out));

  pub_->publish(std::move(loaned_output));
}

}  // namespace gpu_ros::rtdetr

RCLCPP_COMPONENTS_REGISTER_NODE(gpu_ros::rtdetr::RtDetrPreprocessorNode)
