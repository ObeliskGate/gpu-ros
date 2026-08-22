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

#ifndef ISAAC_ROS_YOLOV8_STD__YOLOV8_IMAGE_ENCODER_NODE_HPP_
#define ISAAC_ROS_YOLOV8_STD__YOLOV8_IMAGE_ENCODER_NODE_HPP_

#include "isaac_ros_yolov8_std/yolov8_image_encoder.hpp"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace yolov8_std
{

class YoloV8ImageEncoderNode : public rclcpp::Node
{
public:
  explicit YoloV8ImageEncoderNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());

private:
  using Image = sensor_msgs::msg::Image;
  using TensorList = isaac_ros_tensor_list_interfaces::msg::TensorList;

  void InputCallback(const Image::ConstSharedPtr msg);

  rclcpp::Subscription<Image>::SharedPtr sub_;
  rclcpp::Publisher<TensorList>::SharedPtr pub_;
  YoloV8ImageEncoderConfig config_;
};

}  // namespace yolov8_std
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_YOLOV8_STD__YOLOV8_IMAGE_ENCODER_NODE_HPP_
