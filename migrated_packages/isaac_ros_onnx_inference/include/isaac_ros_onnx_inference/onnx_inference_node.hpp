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

#ifndef ISAAC_ROS_ONNX_INFERENCE__ONNX_INFERENCE_NODE_HPP_
#define ISAAC_ROS_ONNX_INFERENCE__ONNX_INFERENCE_NODE_HPP_

#include <cstddef>
#include <memory>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "isaac_ros_onnx_inference/onnx_inference_core.hpp"
#include "isaac_ros_onnx_inference/tensor_list_io.hpp"

namespace nvidia::isaac_ros::onnx_inference
{

class OnnxInferenceNode : public rclcpp::Node
{
public:
  explicit OnnxInferenceNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~OnnxInferenceNode() override;

private:
  void OnTensors(gpu_ros_managed::ManagedTensorListView inputs);
  void FinalizeOrtProfile(const char * reason) noexcept;

  std::unique_ptr<OnnxInferenceCore> core_;
  std::unique_ptr<ITensorListIO> io_;
  size_t ort_profile_frames_{0};
  size_t inference_count_{0};
};

}  // namespace nvidia::isaac_ros::onnx_inference

#endif  // ISAAC_ROS_ONNX_INFERENCE__ONNX_INFERENCE_NODE_HPP_
