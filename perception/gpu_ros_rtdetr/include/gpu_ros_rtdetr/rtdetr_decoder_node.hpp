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
// Modified from NVIDIA Isaac ROS RT-DETR decoder sources for standard ROS 2
// integration; see THIRD_PARTY_NOTICES.md for the exact pinned revision.

#ifndef GPU_ROS_RTDETR__RTDETR_DECODER_NODE_HPP_
#define GPU_ROS_RTDETR__RTDETR_DECODER_NODE_HPP_

#include <string>

#include "rclcpp/rclcpp.hpp"
#include "gpu_ros_tensor_bundle_msgs/msg/tensor_bundle.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"

namespace gpu_ros::rtdetr
{

// Standard-ROS2 (NITROS-free) RT-DETR decoder.
// Reads labels/boxes/scores host tensors and publishes a Detection2DArray.
class RtDetrDecoderNode : public rclcpp::Node
{
public:
  explicit RtDetrDecoderNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());

private:
  using TensorBundle = gpu_ros_tensor_bundle_msgs::msg::TensorBundle;

  void InputCallback(const TensorBundle::SharedPtr msg);

  rclcpp::Subscription<TensorBundle>::SharedPtr sub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr pub_;

  std::string labels_tensor_name_;
  std::string boxes_tensor_name_;
  std::string scores_tensor_name_;
  double confidence_threshold_;
};

}  // namespace gpu_ros::rtdetr

#endif  // GPU_ROS_RTDETR__RTDETR_DECODER_NODE_HPP_
