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

#ifndef ISAAC_ROS_RTDETR_STD__RTDETR_DECODER_NODE_HPP_
#define ISAAC_ROS_RTDETR_STD__RTDETR_DECODER_NODE_HPP_

#include <string>

#include "rclcpp/rclcpp.hpp"
#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace rtdetr_std
{

// Standard-ROS2 (NITROS-free) RT-DETR decoder.
// Reads labels/boxes/scores host tensors and publishes a Detection2DArray.
class RtDetrDecoderNode : public rclcpp::Node
{
public:
  explicit RtDetrDecoderNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());

private:
  using TensorList = isaac_ros_tensor_list_interfaces::msg::TensorList;

  void InputCallback(const TensorList::SharedPtr msg);

  rclcpp::Subscription<TensorList>::SharedPtr sub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr pub_;

  std::string labels_tensor_name_;
  std::string boxes_tensor_name_;
  std::string scores_tensor_name_;
  double confidence_threshold_;
};

}  // namespace rtdetr_std
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_RTDETR_STD__RTDETR_DECODER_NODE_HPP_
