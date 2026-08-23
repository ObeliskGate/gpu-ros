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

#include "gpu_ros_yolov8/yolov8_image_encoder_node.hpp"

#include <exception>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

#include "rclcpp_components/register_node_macro.hpp"

namespace gpu_ros::yolov8
{

YoloV8ImageEncoderNode::YoloV8ImageEncoderNode(const rclcpp::NodeOptions options)
: rclcpp::Node("yolov8_image_encoder_node", options)
{
  config_.tensor_name = declare_parameter<std::string>("tensor_name", "images");
  config_.output_width = declare_parameter<int64_t>("output_width", 640);
  config_.output_height = declare_parameter<int64_t>("output_height", 640);
  debug_message_flow_ = declare_parameter<bool>("debug_message_flow", false);
  if (
    config_.tensor_name.empty() || config_.output_width <= 0 || config_.output_height <= 0 ||
    config_.output_width > std::numeric_limits<int>::max() ||
    config_.output_height > std::numeric_limits<int>::max())
  {
    throw std::invalid_argument(
            "YOLOv8 tensor_name must not be empty and output dimensions must be positive "
            "and fit in an int");
  }

  pub_ = create_publisher<TensorBundle>("encoded_tensor", 10);
  sub_ = create_subscription<Image>(
    "image", 10,
    std::bind(&YoloV8ImageEncoderNode::InputCallback, this, std::placeholders::_1));
}

void YoloV8ImageEncoderNode::InputCallback(const Image::ConstSharedPtr msg)
{
  TrackMessageId(msg->header.stamp.sec);
  try {
    pub_->publish(EncodeYoloV8Image(*msg, config_));
  } catch (const std::exception & error) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "YOLOv8 image encoder dropped frame: %s", error.what());
  }
}

void YoloV8ImageEncoderNode::TrackMessageId(int64_t message_id)
{
  if (!debug_message_flow_) {
    return;
  }
  if (has_last_message_id_ && message_id > last_message_id_ + 1) {
    RCLCPP_WARN(
      get_logger(),
      "MESSAGE_FLOW_GAP stage=image_encoder_input previous=%lld current=%lld missing=%lld",
      static_cast<long long>(last_message_id_), static_cast<long long>(message_id),
      static_cast<long long>(message_id - last_message_id_ - 1));
  } else if (has_last_message_id_ && message_id <= last_message_id_) {
    RCLCPP_INFO(
      get_logger(), "MESSAGE_FLOW_RESET stage=image_encoder_input previous=%lld current=%lld",
      static_cast<long long>(last_message_id_), static_cast<long long>(message_id));
  }
  last_message_id_ = message_id;
  has_last_message_id_ = true;
}

}  // namespace gpu_ros::yolov8

RCLCPP_COMPONENTS_REGISTER_NODE(gpu_ros::yolov8::YoloV8ImageEncoderNode)
