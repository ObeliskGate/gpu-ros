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

#ifndef ISAAC_ROS_DETECTION_COMMON__IMAGE_PREPROCESS_HPP_
#define ISAAC_ROS_DETECTION_COMMON__IMAGE_PREPROCESS_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "sensor_msgs/msg/image.hpp"

namespace nvidia::isaac_ros::detection_common
{

enum class ImageEncoding
{
  kRgb8,
  kBgr8,
  kRgba8,
  kBgra8,
  kMono8
};

enum class PreprocessNormalization
{
  kNone,
  kUnitRange
};

struct ImagePreprocessPlan
{
  ImageEncoding encoding{ImageEncoding::kRgb8};
  uint32_t input_width{0};
  uint32_t input_height{0};
  uint32_t input_step{0};
  uint32_t input_channels{0};
  int output_width{0};
  int output_height{0};
  int resized_width{0};
  int resized_height{0};
  PreprocessNormalization normalization{PreprocessNormalization::kNone};
};

ImagePreprocessPlan MakeImagePreprocessPlan(
  const sensor_msgs::msg::Image & image,
  int64_t output_width,
  int64_t output_height,
  PreprocessNormalization normalization);

// CPU reference implementation. The result is contiguous NCHW float32 with
// channel order RGB and top-left placement with zero bottom/right padding.
std::vector<float> ExecuteCpuPreprocess(
  const sensor_msgs::msg::Image & image,
  const ImagePreprocessPlan & plan);

size_t SourceRowBytes(const ImagePreprocessPlan & plan);
const char * ImageEncodingName(ImageEncoding encoding) noexcept;

}  // namespace nvidia::isaac_ros::detection_common

#endif  // ISAAC_ROS_DETECTION_COMMON__IMAGE_PREPROCESS_HPP_
