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

#include "gpu_ros_yolov8/yolov8_image_encoder.hpp"
#include "gpu_ros_tensor_bundle_msgs/msg/tensor.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "gpu_ros_detection_common/image_preprocess.hpp"

namespace gpu_ros::yolov8
{
namespace
{
constexpr int kChannels = 3;
}

gpu_ros_tensor_bundle_msgs::msg::TensorBundle EncodeYoloV8Image(
  const sensor_msgs::msg::Image & image,
  const YoloV8ImageEncoderConfig & config)
{
  if (config.tensor_name.empty()) {
    throw std::invalid_argument("YOLOv8 tensor_name must not be empty");
  }
  const auto plan = detection_common::MakeImagePreprocessPlan(
    image, config.output_width, config.output_height,
    detection_common::PreprocessNormalization::kUnitRange);
  const auto values = detection_common::ExecuteCpuPreprocess(image, plan);

  gpu_ros_tensor_bundle_msgs::msg::Tensor tensor;
  tensor.name = config.tensor_name;
  tensor.data_type = gpu_ros_tensor_bundle_msgs::msg::Tensor::FLOAT32;
  tensor.shape = {1, kChannels, config.output_height, config.output_width};
  tensor.data.resize(values.size() * sizeof(float));
  std::memcpy(tensor.data.data(), values.data(), tensor.data.size());

  gpu_ros_tensor_bundle_msgs::msg::TensorBundle output;
  output.header = image.header;
  output.tensors.push_back(std::move(tensor));
  return output;
}

}  // namespace gpu_ros::yolov8
