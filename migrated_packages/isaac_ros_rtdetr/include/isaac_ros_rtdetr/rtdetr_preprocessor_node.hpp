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

#ifndef ISAAC_ROS_RTDETR__RTDETR_PREPROCESSOR_NODE_HPP_
#define ISAAC_ROS_RTDETR__RTDETR_PREPROCESSOR_NODE_HPP_

#include <string>

#include "rclcpp/rclcpp.hpp"
#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace rtdetr
{

// Standard-ROS2 (NITROS-free) RT-DETR preprocessor.
// Subscribes to a single encoded image tensor and republishes it together with
// an int64 orig_target_sizes [1, 2] tensor, as required by the RT-DETR model.
class RtDetrPreprocessorNode : public rclcpp::Node
{
public:
  explicit RtDetrPreprocessorNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());

private:
  using TensorList = isaac_ros_tensor_list_interfaces::msg::TensorList;

  void InputCallback(const TensorList::SharedPtr msg);

  rclcpp::Subscription<TensorList>::SharedPtr sub_;
  rclcpp::Publisher<TensorList>::SharedPtr pub_;

  std::string input_image_tensor_name_;
  std::string output_image_tensor_name_;
  std::string output_size_tensor_name_;
  int64_t image_height_;
  int64_t image_width_;
  bool use_max_dim_for_orig_size_;
};

}  // namespace rtdetr
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_RTDETR__RTDETR_PREPROCESSOR_NODE_HPP_
