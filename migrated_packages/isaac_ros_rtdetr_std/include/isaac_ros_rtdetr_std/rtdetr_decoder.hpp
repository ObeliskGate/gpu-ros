// Copyright 2026 Maintainer
// Licensed under the Apache License, Version 2.0.
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
