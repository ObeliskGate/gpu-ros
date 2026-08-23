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

#include "gpu_ros_nvidia_tensor_bundle_compat/tensor_bundle_conversion.hpp"

#include <exception>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace gpu_ros::nvidia_tensor_bundle_compat
{

class NvidiaTensorListToTensorBundleNode final : public rclcpp::Node
{
public:
  explicit NvidiaTensorListToTensorBundleNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : rclcpp::Node("nvidia_tensor_list_to_tensor_bundle", options)
  {
    publisher_ = create_publisher<TensorBundle>("tensor_output", 10);
    subscription_ = create_subscription<NvidiaTensorList>(
      "tensor_input", 10,
      [this](const NvidiaTensorList::SharedPtr message) {
        try {
          publisher_->publish(ToTensorBundle(*message));
        } catch (const std::exception & error) {
          RCLCPP_ERROR(get_logger(), "TensorList to TensorBundle conversion failed: %s", error.what());
        }
      });
  }

private:
  rclcpp::Publisher<TensorBundle>::SharedPtr publisher_;
  rclcpp::Subscription<NvidiaTensorList>::SharedPtr subscription_;
};

class TensorBundleToNvidiaTensorListNode final : public rclcpp::Node
{
public:
  explicit TensorBundleToNvidiaTensorListNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : rclcpp::Node("tensor_bundle_to_nvidia_tensor_list", options)
  {
    publisher_ = create_publisher<NvidiaTensorList>("tensor_output", 10);
    subscription_ = create_subscription<TensorBundle>(
      "tensor_input", 10,
      [this](const TensorBundle::SharedPtr message) {
        try {
          publisher_->publish(ToNvidiaTensorList(*message));
        } catch (const std::exception & error) {
          RCLCPP_ERROR(get_logger(), "TensorBundle to TensorList conversion failed: %s", error.what());
        }
      });
  }

private:
  rclcpp::Publisher<NvidiaTensorList>::SharedPtr publisher_;
  rclcpp::Subscription<TensorBundle>::SharedPtr subscription_;
};

}  // namespace gpu_ros::nvidia_tensor_bundle_compat

RCLCPP_COMPONENTS_REGISTER_NODE(
  gpu_ros::nvidia_tensor_bundle_compat::NvidiaTensorListToTensorBundleNode)
RCLCPP_COMPONENTS_REGISTER_NODE(
  gpu_ros::nvidia_tensor_bundle_compat::TensorBundleToNvidiaTensorListNode)
