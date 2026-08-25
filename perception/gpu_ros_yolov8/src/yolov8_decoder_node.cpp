// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// Modified from NVIDIA Isaac ROS YOLOv8 decoder sources for standard ROS 2
// integration; see THIRD_PARTY_NOTICES.md for the exact pinned revision.

#include "gpu_ros_yolov8/yolov8_decoder_node.hpp"

#include <exception>
#include <string>
#include <utility>

#include "rclcpp_components/register_node_macro.hpp"

namespace gpu_ros::yolov8
{

YoloV8DecoderNode::YoloV8DecoderNode(const rclcpp::NodeOptions options)
: rclcpp::Node("yolov8_decoder_node", options)
{
  config_.tensor_name = declare_parameter<std::string>("tensor_name", "output_tensor");
  config_.confidence_threshold = declare_parameter<double>("confidence_threshold", 0.25);
  config_.nms_threshold = declare_parameter<double>("nms_threshold", 0.45);
  config_.num_classes = declare_parameter<int64_t>("num_classes", 80);

  pub_ = create_publisher<vision_msgs::msg::Detection2DArray>("detections_output", 10);
  sub_ = create_subscription<TensorBundle>(
    "tensor_sub", rclcpp::QoS(10),
    [this](TensorBundle::SharedPtr msg) {InputCallback(std::move(msg));});
}

void YoloV8DecoderNode::InputCallback(const TensorBundle::SharedPtr msg)
{
  try {
    pub_->publish(DecodeYoloV8TensorBundle(*msg, config_));
  } catch (const std::exception & error) {
    RCLCPP_ERROR(get_logger(), "Failed to decode YOLOv8 TensorBundle: %s", error.what());
  }
}

}  // namespace gpu_ros::yolov8

RCLCPP_COMPONENTS_REGISTER_NODE(gpu_ros::yolov8::YoloV8DecoderNode)
