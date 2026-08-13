// Copyright 2026 Maintainer
// Licensed under the Apache License, Version 2.0.
#ifndef ISAAC_ROS_DETECTION_COMMON__HIP_PREPROCESS_HPP_
#define ISAAC_ROS_DETECTION_COMMON__HIP_PREPROCESS_HPP_

#include <cstdint>

#include <hip/hip_runtime_api.h>

#include "isaac_ros_detection_common/image_preprocess.hpp"

namespace nvidia::isaac_ros::detection_common
{

void LaunchHipPreprocess(
  const uint8_t * raw_image,
  float * output,
  const ImagePreprocessPlan & plan,
  hipStream_t stream);

void LaunchHipWriteInt64Pair(
  int64_t * output, int64_t first, int64_t second, hipStream_t stream);

}  // namespace nvidia::isaac_ros::detection_common

#endif  // ISAAC_ROS_DETECTION_COMMON__HIP_PREPROCESS_HPP_
