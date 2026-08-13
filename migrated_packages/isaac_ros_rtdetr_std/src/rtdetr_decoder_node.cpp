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

#include "isaac_ros_rtdetr_std/rtdetr_decoder_node.hpp"

#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "isaac_ros_rtdetr_std/rtdetr_decoder.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace nvidia::isaac_ros::rtdetr_std
{
namespace
{
template<typename T>
std::vector<T> TensorToVector(
  const isaac_ros_tensor_list_interfaces::msg::TensorList & message,
  const std::string & name,
  int expected_dtype,
  const std::vector<uint32_t> & expected_shape)
{
  for (const auto & tensor : message.tensors) {
    if (tensor.name == name) {
      size_t expected_elements = 1U;
      for (const auto dimension : expected_shape) {
        if (dimension == 0U ||
          expected_elements > std::numeric_limits<size_t>::max() / dimension)
        {
          throw std::invalid_argument("RT-DETR tensor contract size overflow");
        }
        expected_elements *= dimension;
      }
      if (expected_elements > std::numeric_limits<size_t>::max() / sizeof(T)) {
        throw std::invalid_argument("RT-DETR tensor byte size overflow");
      }
      const size_t expected_bytes = expected_elements * sizeof(T);
      if (tensor.data_type != expected_dtype ||
        static_cast<size_t>(tensor.shape.rank) != expected_shape.size() ||
        tensor.shape.dims != expected_shape || tensor.data.size() != expected_bytes)
      {
        throw std::invalid_argument("RT-DETR tensor '" + name + "' has the wrong contract");
      }
      std::vector<T> values(tensor.data.size() / sizeof(T));
      std::memcpy(values.data(), tensor.data.data(), tensor.data.size());
      return values;
    }
  }
  throw std::invalid_argument("RT-DETR tensor '" + name + "' not found");
}
}  // namespace

RtDetrDecoderNode::RtDetrDecoderNode(const rclcpp::NodeOptions options)
: rclcpp::Node("rtdetr_decoder_node", options),
  labels_tensor_name_{declare_parameter<std::string>("labels_tensor_name", "labels")},
  boxes_tensor_name_{declare_parameter<std::string>("boxes_tensor_name", "boxes")},
  scores_tensor_name_{declare_parameter<std::string>("scores_tensor_name", "scores")},
  confidence_threshold_{declare_parameter<double>("confidence_threshold", 0.9)}
{
  pub_ = create_publisher<vision_msgs::msg::Detection2DArray>("detections_output", 10);
  sub_ = create_subscription<TensorList>(
    "tensor_sub", 10,
    std::bind(&RtDetrDecoderNode::InputCallback, this, std::placeholders::_1));
}

void RtDetrDecoderNode::InputCallback(const TensorList::SharedPtr msg)
{
  try {
    const auto labels = TensorToVector<int64_t>(*msg, labels_tensor_name_, 7, {1, 300});
    const auto boxes = TensorToVector<float>(*msg, boxes_tensor_name_, 9, {1, 300, 4});
    const auto scores = TensorToVector<float>(*msg, scores_tensor_name_, 9, {1, 300});
    RtDetrDecoderConfig config;
    config.labels_tensor_name = labels_tensor_name_;
    config.boxes_tensor_name = boxes_tensor_name_;
    config.scores_tensor_name = scores_tensor_name_;
    config.confidence_threshold = confidence_threshold_;
    pub_->publish(DecodeRtDetrValues(
      msg->header, labels.data(), labels.size(), boxes.data(), boxes.size(),
      scores.data(), scores.size(), config));
  } catch (const std::exception & error) {
    RCLCPP_ERROR(get_logger(), "Failed to decode RT-DETR TensorList: %s", error.what());
  }
}

}  // namespace nvidia::isaac_ros::rtdetr_std

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::rtdetr_std::RtDetrDecoderNode)
