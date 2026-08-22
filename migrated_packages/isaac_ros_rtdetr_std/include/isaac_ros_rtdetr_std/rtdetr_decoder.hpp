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
// Modified from NVIDIA Isaac ROS RT-DETR decoder sources for a standard ROS 2
// core; see THIRD_PARTY_NOTICES.md for the exact pinned revision.

#ifndef ISAAC_ROS_RTDETR_STD__RTDETR_DECODER_HPP_
#define ISAAC_ROS_RTDETR_STD__RTDETR_DECODER_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "std_msgs/msg/header.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"

namespace nvidia::isaac_ros::rtdetr_std
{

struct RtDetrDecoderConfig
{
  std::string labels_tensor_name{"labels"};
  std::string boxes_tensor_name{"boxes"};
  std::string scores_tensor_name{"scores"};
  double confidence_threshold{0.9};
};

vision_msgs::msg::Detection2DArray DecodeRtDetrValues(
  const std_msgs::msg::Header & header,
  const int64_t * labels,
  size_t label_count,
  const float * boxes,
  size_t box_value_count,
  const float * scores,
  size_t score_count,
  const RtDetrDecoderConfig & config);

}  // namespace nvidia::isaac_ros::rtdetr_std

#endif  // ISAAC_ROS_RTDETR_STD__RTDETR_DECODER_HPP_
