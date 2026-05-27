// Copyright 2026 - Apache-2.0
#ifndef ISAAC_ROS_ONNX_INFERENCE__ONNX_INFERENCE_NODE_HPP_
#define ISAAC_ROS_ONNX_INFERENCE__ONNX_INFERENCE_NODE_HPP_

#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"
#include "isaac_ros_onnx_inference/onnx_inference_core.hpp"

namespace nvidia::isaac_ros::onnx_inference
{

class OnnxInferenceNode : public rclcpp::Node
{
public:
  explicit OnnxInferenceNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void OnTensorList(
    const isaac_ros_tensor_list_interfaces::msg::TensorList::SharedPtr msg);

  std::unique_ptr<OnnxInferenceCore> core_;

  rclcpp::Subscription<isaac_ros_tensor_list_interfaces::msg::TensorList>::SharedPtr sub_;
  rclcpp::Publisher<isaac_ros_tensor_list_interfaces::msg::TensorList>::SharedPtr pub_;

  std::vector<std::string> input_tensor_names_;
  std::vector<std::string> input_binding_names_;
  std::vector<std::string> output_tensor_names_;
  std::vector<std::string> output_binding_names_;
};

}  // namespace nvidia::isaac_ros::onnx_inference

#endif  // ISAAC_ROS_ONNX_INFERENCE__ONNX_INFERENCE_NODE_HPP_
