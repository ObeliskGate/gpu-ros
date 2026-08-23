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

#include "gpu_ros_rtdetr/rtdetr_decoder.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace gpu_ros::rtdetr
{

vision_msgs::msg::Detection2DArray DecodeRtDetrValues(
  const std_msgs::msg::Header & header,
  const int64_t * labels,
  size_t label_count,
  const float * boxes,
  size_t box_value_count,
  const float * scores,
  size_t score_count,
  const RtDetrDecoderConfig & config)
{
  if (labels == nullptr || boxes == nullptr || scores == nullptr ||
    label_count != score_count ||
    score_count > std::numeric_limits<size_t>::max() / 4U ||
    box_value_count != score_count * 4U)
  {
    throw std::invalid_argument("RT-DETR decoder tensor spans have inconsistent sizes");
  }
  if (config.confidence_threshold < 0.0 || config.confidence_threshold > 1.0) {
    throw std::invalid_argument("RT-DETR confidence_threshold must be in [0, 1]");
  }
  vision_msgs::msg::Detection2DArray detections;
  detections.header = header;
  for (size_t index = 0; index < score_count; ++index) {
    if (!std::isfinite(scores[index]) || scores[index] <= config.confidence_threshold) {
      continue;
    }
    const float x1 = boxes[4U * index];
    const float y1 = boxes[4U * index + 1U];
    const float x2 = boxes[4U * index + 2U];
    const float y2 = boxes[4U * index + 3U];
    if (!std::isfinite(x1) || !std::isfinite(y1) || !std::isfinite(x2) ||
      !std::isfinite(y2) || x2 < x1 || y2 < y1)
    {
      continue;
    }
    vision_msgs::msg::Detection2D detection;
    detection.header = header;
    vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
    hypothesis.hypothesis.class_id = std::to_string(labels[index]);
    hypothesis.hypothesis.score = scores[index];
    detection.results.push_back(hypothesis);
    detection.bbox.center.position.x = (x1 + x2) / 2.0F;
    detection.bbox.center.position.y = (y1 + y2) / 2.0F;
    detection.bbox.size_x = x2 - x1;
    detection.bbox.size_y = y2 - y1;
    detections.detections.push_back(std::move(detection));
  }
  return detections;
}

}  // namespace gpu_ros::rtdetr
