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
//
// NOTE: std-ROS2 TensorList sub/pub is implemented in a later task.
// This file provides the node skeleton so the package compiles.
#include "isaac_ros_onnx_inference/onnx_inference_node.hpp"

#include <stdexcept>
#include <string>

#include "rclcpp_components/register_node_macro.hpp"

namespace nvidia::isaac_ros::onnx_inference
{

OnnxInferenceNode::OnnxInferenceNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("onnx_inference_node", options)
{
  // Parameters
  const std::string model_file_path =
    declare_parameter<std::string>("model_file_path", "");
  const std::string ep_str =
    declare_parameter<std::string>("execution_provider", "cuda");
  const int gpu_device_id =
    declare_parameter<int>("gpu_device_id", 0);

  if (model_file_path.empty()) {
    RCLCPP_WARN(get_logger(), "model_file_path is empty — inference core not initialized.");
    return;
  }

  OnnxInferenceCore::Config cfg;
  cfg.model_file_path = model_file_path;
  cfg.ep = ParseExecutionProvider(ep_str);
  cfg.gpu_device_id = gpu_device_id;
  core_ = std::make_unique<OnnxInferenceCore>(cfg);

  RCLCPP_INFO(
    get_logger(),
    "Loaded model '%s' with %zu inputs, %zu outputs, EP=%s",
    model_file_path.c_str(),
    core_->GetInputCount(),
    core_->GetOutputCount(),
    ep_str.c_str());

  // TODO(Task 3): wire up sub_ and pub_
}

void OnnxInferenceNode::OnTensorList(
  const isaac_ros_tensor_list_interfaces::msg::TensorList::SharedPtr /*msg*/)
{
  // TODO(Task 3): implement
  throw std::logic_error("OnnxInferenceNode::OnTensorList not yet implemented (Task 3)");
}

}  // namespace nvidia::isaac_ros::onnx_inference

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::onnx_inference::OnnxInferenceNode)
