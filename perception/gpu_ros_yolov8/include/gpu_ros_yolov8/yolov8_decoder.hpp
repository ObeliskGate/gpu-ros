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

#ifndef GPU_ROS_YOLOV8__YOLOV8_DECODER_HPP_
#define GPU_ROS_YOLOV8__YOLOV8_DECODER_HPP_

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

#include "gpu_ros_tensor_bundle_msgs/msg/tensor.hpp"
#include "gpu_ros_tensor_bundle_msgs/msg/tensor_bundle.hpp"
#include "std_msgs/msg/header.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"

namespace gpu_ros::yolov8
{

constexpr uint8_t kTensorBundleFloat32 =
  gpu_ros_tensor_bundle_msgs::msg::Tensor::FLOAT32;

struct YoloV8DecoderConfig
{
  std::string tensor_name{"output_tensor"};
  double confidence_threshold{0.25};
  double nms_threshold{0.45};
  int64_t num_classes{80};
};

vision_msgs::msg::Detection2DArray DecodeYoloV8TensorBundle(
  const gpu_ros_tensor_bundle_msgs::msg::TensorBundle & msg,
  const YoloV8DecoderConfig & config);

vision_msgs::msg::Detection2DArray DecodeYoloV8Values(
  const std_msgs::msg::Header & header,
  const float * values,
  size_t value_count,
  const std::vector<int64_t> & shape,
  const YoloV8DecoderConfig & config);

}  // namespace gpu_ros::yolov8

#endif  // GPU_ROS_YOLOV8__YOLOV8_DECODER_HPP_
