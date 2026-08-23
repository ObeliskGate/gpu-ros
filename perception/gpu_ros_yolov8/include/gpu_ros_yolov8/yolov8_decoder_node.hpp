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

#ifndef GPU_ROS_YOLOV8__YOLOV8_DECODER_NODE_HPP_
#define GPU_ROS_YOLOV8__YOLOV8_DECODER_NODE_HPP_

#include <cstdint>

#include "gpu_ros_yolov8/yolov8_decoder.hpp"

#include "gpu_ros_tensor_bundle_msgs/msg/tensor_bundle.hpp"
#include "rclcpp/rclcpp.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"

namespace gpu_ros::yolov8
{

class YoloV8DecoderNode : public rclcpp::Node
{
public:
  explicit YoloV8DecoderNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());

private:
  using TensorBundle = gpu_ros_tensor_bundle_msgs::msg::TensorBundle;

  void InputCallback(const TensorBundle::SharedPtr msg);
  void TrackMessageId(int64_t message_id);

  rclcpp::Subscription<TensorBundle>::SharedPtr sub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr pub_;
  YoloV8DecoderConfig config_;
  bool debug_message_flow_{false};
  bool has_last_message_id_{false};
  int64_t last_message_id_{0};
};

}  // namespace gpu_ros::yolov8

#endif  // GPU_ROS_YOLOV8__YOLOV8_DECODER_NODE_HPP_
