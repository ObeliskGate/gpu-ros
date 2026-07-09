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

#ifndef ISAAC_ROS_YOLOV8_STD__YOLOV8_DECODER_NODE_HPP_
#define ISAAC_ROS_YOLOV8_STD__YOLOV8_DECODER_NODE_HPP_

#include "isaac_ros_yolov8_std/yolov8_decoder.hpp"

#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"
#include "rclcpp/rclcpp.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace yolov8_std
{

class YoloV8DecoderNode : public rclcpp::Node
{
public:
  explicit YoloV8DecoderNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());

private:
  using TensorList = isaac_ros_tensor_list_interfaces::msg::TensorList;

  void InputCallback(const TensorList::SharedPtr msg);

  rclcpp::Subscription<TensorList>::SharedPtr sub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr pub_;
  YoloV8DecoderConfig config_;
};

}  // namespace yolov8_std
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_YOLOV8_STD__YOLOV8_DECODER_NODE_HPP_
