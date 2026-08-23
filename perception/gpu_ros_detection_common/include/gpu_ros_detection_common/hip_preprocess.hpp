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

#ifndef GPU_ROS_DETECTION_COMMON__HIP_PREPROCESS_HPP_
#define GPU_ROS_DETECTION_COMMON__HIP_PREPROCESS_HPP_

#include <cstdint>

#include <hip/hip_runtime_api.h>

#include "gpu_ros_detection_common/image_preprocess.hpp"

namespace gpu_ros::detection_common
{

void LaunchHipPreprocess(
  const uint8_t * raw_image,
  float * output,
  const ImagePreprocessPlan & plan,
  hipStream_t stream);

void LaunchHipWriteInt64Pair(
  int64_t * output, int64_t first, int64_t second, hipStream_t stream);

}  // namespace gpu_ros::detection_common

#endif  // GPU_ROS_DETECTION_COMMON__HIP_PREPROCESS_HPP_
