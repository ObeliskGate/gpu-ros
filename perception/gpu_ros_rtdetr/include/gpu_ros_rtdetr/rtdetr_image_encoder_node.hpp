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

#ifndef GPU_ROS_RTDETR__RTDETR_IMAGE_ENCODER_NODE_HPP_
#define GPU_ROS_RTDETR__RTDETR_IMAGE_ENCODER_NODE_HPP_

#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "gpu_ros_tensor_bundle_msgs/msg/tensor.hpp"
#include "gpu_ros_tensor_bundle_msgs/msg/tensor_bundle.hpp"

namespace gpu_ros::rtdetr
{

// Standard-ROS2 RT-DETR image encoder.
// Converts sensor_msgs/Image to a host-memory float32 NCHW TensorBundle tensor:
// RGB, aspect-ratio resize, bottom/right zero padding, shape [1, 3, H, W].
class RtDetrImageEncoderNode : public rclcpp::Node
{
public:
  explicit RtDetrImageEncoderNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());

private:
  using Image = sensor_msgs::msg::Image;
  using Tensor = gpu_ros_tensor_bundle_msgs::msg::Tensor;
  using TensorBundle = gpu_ros_tensor_bundle_msgs::msg::TensorBundle;

  void InputCallback(const Image::ConstSharedPtr msg);

  rclcpp::Subscription<Image>::SharedPtr sub_;
  rclcpp::Publisher<TensorBundle>::SharedPtr pub_;

  std::string tensor_name_;
  int64_t output_height_;
  int64_t output_width_;
};

}  // namespace gpu_ros::rtdetr

#endif  // GPU_ROS_RTDETR__RTDETR_IMAGE_ENCODER_NODE_HPP_
