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

#ifndef GPU_ROS_YOLOV8__YOLOV8_IMAGE_ENCODER_NODE_HPP_
#define GPU_ROS_YOLOV8__YOLOV8_IMAGE_ENCODER_NODE_HPP_

#include "gpu_ros_yolov8/yolov8_image_encoder.hpp"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace gpu_ros::yolov8
{

class YoloV8ImageEncoderNode : public rclcpp::Node
{
public:
  explicit YoloV8ImageEncoderNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());

private:
  using Image = sensor_msgs::msg::Image;
  using TensorBundle = gpu_ros_tensor_bundle_msgs::msg::TensorBundle;

  void InputCallback(const Image::ConstSharedPtr msg);

  rclcpp::Subscription<Image>::SharedPtr sub_;
  rclcpp::Publisher<TensorBundle>::SharedPtr pub_;
  YoloV8ImageEncoderConfig config_;
};

}  // namespace gpu_ros::yolov8

#endif  // GPU_ROS_YOLOV8__YOLOV8_IMAGE_ENCODER_NODE_HPP_
