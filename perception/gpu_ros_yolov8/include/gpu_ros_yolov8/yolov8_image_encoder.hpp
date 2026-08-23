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

#ifndef GPU_ROS_YOLOV8__YOLOV8_IMAGE_ENCODER_HPP_
#define GPU_ROS_YOLOV8__YOLOV8_IMAGE_ENCODER_HPP_

#include <cstdint>
#include <string>

#include "gpu_ros_tensor_bundle_msgs/msg/tensor_bundle.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace gpu_ros::yolov8
{

struct YoloV8ImageEncoderConfig
{
  std::string tensor_name{"images"};
  int64_t output_width{640};
  int64_t output_height{640};
};

// Convert a standard ROS2 image into the fixed YOLOv8 model input contract:
// RGB8, aspect-ratio-preserving resize, bottom/right zero padding, NCHW
// float32, and a fixed 1/255 normalization.
gpu_ros_tensor_bundle_msgs::msg::TensorBundle EncodeYoloV8Image(
  const sensor_msgs::msg::Image & image,
  const YoloV8ImageEncoderConfig & config);

}  // namespace gpu_ros::yolov8

#endif  // GPU_ROS_YOLOV8__YOLOV8_IMAGE_ENCODER_HPP_
