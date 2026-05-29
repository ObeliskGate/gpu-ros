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

#include "isaac_ros_onnx_inference/onnx_inference_node.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include "rclcpp_components/register_node_macro.hpp"

namespace nvidia::isaac_ros::onnx_inference
{

namespace
{

// isaac_ros_tensor_list_interfaces Tensor.data_type uses the GXF PrimitiveType
// enum, which differs from ONNXTensorElementDataType. Map between the two.
constexpr int kGxfFloat32 = 9;
constexpr int kGxfInt64 = 7;

ONNXTensorElementDataType GxfToOnnxDtype(int gxf_dtype)
{
  switch (gxf_dtype) {
    case kGxfFloat32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case kGxfInt64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    default:
      throw std::runtime_error(
              "OnnxInferenceNode: unsupported input data_type " + std::to_string(gxf_dtype));
  }
}

int OnnxToGxfDtype(ONNXTensorElementDataType onnx_dtype)
{
  switch (onnx_dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return kGxfFloat32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return kGxfInt64;
    default:
      throw std::runtime_error(
              "OnnxInferenceNode: unsupported output dtype " +
              std::to_string(static_cast<int>(onnx_dtype)));
  }
}

}  // namespace

OnnxInferenceNode::OnnxInferenceNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("onnx_inference_node", options)
{
  const std::string model_file_path =
    declare_parameter<std::string>("model_file_path", "");
  const std::string ep_str =
    declare_parameter<std::string>("execution_provider", "cuda");
  const int gpu_device_id =
    declare_parameter<int>("gpu_device_id", 0);

  pub_ = create_publisher<isaac_ros_tensor_list_interfaces::msg::TensorList>(
    "tensor_output", 10);
  sub_ = create_subscription<isaac_ros_tensor_list_interfaces::msg::TensorList>(
    "tensor_input", 10,
    std::bind(&OnnxInferenceNode::OnTensorList, this, std::placeholders::_1));

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
}

void OnnxInferenceNode::OnTensorList(
  const isaac_ros_tensor_list_interfaces::msg::TensorList::SharedPtr msg)
{
  if (!core_) {
    RCLCPP_WARN_ONCE(get_logger(), "Received tensor but inference core is not initialized.");
    return;
  }

  std::vector<HostTensor> inputs;
  inputs.reserve(msg->tensors.size());
  for (const auto & t : msg->tensors) {
    HostTensor ht;
    ht.name = t.name;
    ht.dtype = GxfToOnnxDtype(t.data_type);
    ht.shape.assign(t.shape.dims.begin(), t.shape.dims.end());
    ht.data = t.data;
    inputs.push_back(std::move(ht));
  }

  std::vector<HostTensor> outputs = core_->RunInference(inputs);

  isaac_ros_tensor_list_interfaces::msg::TensorList out_msg;
  out_msg.header = msg->header;
  out_msg.tensors.reserve(outputs.size());
  for (const auto & ht : outputs) {
    isaac_ros_tensor_list_interfaces::msg::Tensor t;
    t.name = ht.name;
    t.data_type = OnnxToGxfDtype(ht.dtype);
    t.shape.rank = static_cast<int32_t>(ht.shape.size());
    t.shape.dims.assign(ht.shape.begin(), ht.shape.end());
    t.data = ht.data;
    out_msg.tensors.push_back(std::move(t));
  }
  pub_->publish(out_msg);
}

}  // namespace nvidia::isaac_ros::onnx_inference

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::onnx_inference::OnnxInferenceNode)
