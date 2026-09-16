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

#include "gpu_ros_rtdetr/rtdetr_image_encoder_node.hpp"

#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "gpu_ros_detection_common/image_preprocess.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace gpu_ros::rtdetr
{
namespace
{
constexpr int kChannels = 3;
}

RtDetrImageEncoderNode::RtDetrImageEncoderNode(const rclcpp::NodeOptions options)
    : rclcpp::Node("rtdetr_image_encoder_node", options),
      tensor_name_{declare_parameter<std::string>("tensor_name", "input_tensor")},
      output_height_{declare_parameter<int64_t>("output_height", 640)},
      output_width_{declare_parameter<int64_t>("output_width", 640)}
{
  if (tensor_name_.empty() || output_height_ <= 0 || output_width_ <= 0 ||
      output_height_ > std::numeric_limits<int>::max() ||
      output_width_ > std::numeric_limits<int>::max())
  {
    throw std::invalid_argument(
      "tensor_name must not be empty and output dimensions must be positive and fit in int");
  }
  pub_ = create_publisher<TensorBundle>("encoded_tensor", 10);
  sub_ = create_subscription<Image>(
    "image", 10, std::bind(&RtDetrImageEncoderNode::InputCallback, this, std::placeholders::_1));
}

void RtDetrImageEncoderNode::InputCallback(const Image::ConstSharedPtr msg)
{
  try {
    const auto plan = detection_common::MakeImagePreprocessPlan(*msg, output_width_, output_height_,
      // RT-DETRv2's validation pipeline consumes float32 RGB values in [0, 1].
      detection_common::PreprocessNormalization::kUnitRange);
    const auto values = detection_common::ExecuteCpuPreprocess(*msg, plan);

    Tensor tensor;
    tensor.name = tensor_name_;
    tensor.data_type = Tensor::FLOAT32;
    tensor.shape = {1, kChannels, output_height_, output_width_};
    tensor.data.resize(values.size() * sizeof(float));
    std::memcpy(tensor.data.data(), values.data(), tensor.data.size());

    TensorBundle output;
    output.header = msg->header;
    output.tensors.push_back(std::move(tensor));
    pub_->publish(std::move(output));
  } catch (const std::exception & error) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000, "RT-DETR image encoder dropped frame: %s", error.what());
  }
}

} // namespace gpu_ros::rtdetr

RCLCPP_COMPONENTS_REGISTER_NODE(gpu_ros::rtdetr::RtDetrImageEncoderNode)
