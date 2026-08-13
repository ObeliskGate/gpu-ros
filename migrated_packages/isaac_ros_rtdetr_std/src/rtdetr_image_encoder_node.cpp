// Copyright 2026 Maintainer
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

#include "isaac_ros_rtdetr_std/rtdetr_image_encoder_node.hpp"

#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "isaac_ros_detection_common/image_preprocess.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace nvidia::isaac_ros::rtdetr_std
{
namespace
{
constexpr int kGxfFloat32 = 9;
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
  pub_ = create_publisher<TensorList>("encoded_tensor", 10);
  sub_ = create_subscription<Image>(
    "image", 10,
    std::bind(&RtDetrImageEncoderNode::InputCallback, this, std::placeholders::_1));
}

void RtDetrImageEncoderNode::InputCallback(const Image::ConstSharedPtr msg)
{
  try {
    const auto plan = detection_common::MakeImagePreprocessPlan(
      *msg, output_width_, output_height_,
      detection_common::PreprocessNormalization::kNone);
    const auto values = detection_common::ExecuteCpuPreprocess(*msg, plan);

    Tensor tensor;
    tensor.name = tensor_name_;
    tensor.data_type = kGxfFloat32;
    tensor.shape.rank = 4;
    tensor.shape.dims = {
      1U, static_cast<uint32_t>(kChannels),
      static_cast<uint32_t>(output_height_), static_cast<uint32_t>(output_width_)};
    tensor.data.resize(values.size() * sizeof(float));
    std::memcpy(tensor.data.data(), values.data(), tensor.data.size());

    TensorList output;
    output.header = msg->header;
    output.tensors.push_back(std::move(tensor));
    pub_->publish(std::move(output));
  } catch (const std::exception & error) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "RT-DETR image encoder dropped frame: %s", error.what());
  }
}

}  // namespace nvidia::isaac_ros::rtdetr_std

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::rtdetr_std::RtDetrImageEncoderNode)
