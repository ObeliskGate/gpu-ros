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
// Modified from NVIDIA Isaac ROS YOLOv8 decoder sources for a standard ROS 2
// core; see THIRD_PARTY_NOTICES.md for the exact pinned revision.

#ifndef ISAAC_ROS_YOLOV8_STD__YOLOV8_DECODER_HPP_
#define ISAAC_ROS_YOLOV8_STD__YOLOV8_DECODER_HPP_

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"
#include "std_msgs/msg/header.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace yolov8_std
{

constexpr int kGxfFloat32 = 9;

struct YoloV8DecoderConfig
{
  std::string tensor_name{"output_tensor"};
  double confidence_threshold{0.25};
  double nms_threshold{0.45};
  int64_t num_classes{80};
};

vision_msgs::msg::Detection2DArray DecodeYoloV8TensorList(
  const isaac_ros_tensor_list_interfaces::msg::TensorList & msg,
  const YoloV8DecoderConfig & config);

vision_msgs::msg::Detection2DArray DecodeYoloV8Values(
  const std_msgs::msg::Header & header,
  const float * values,
  size_t value_count,
  const std::vector<int64_t> & shape,
  const YoloV8DecoderConfig & config);

}  // namespace yolov8_std
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_YOLOV8_STD__YOLOV8_DECODER_HPP_
