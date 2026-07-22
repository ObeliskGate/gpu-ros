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

#ifndef ISAAC_ROS_ONNX_INFERENCE__TENSOR_LIST_IO_HPP_
#define ISAAC_ROS_ONNX_INFERENCE__TENSOR_LIST_IO_HPP_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/header.hpp"

#include "isaac_ros_onnx_inference/onnx_inference_core.hpp"

namespace nvidia::isaac_ros::onnx_inference
{

// Transport-agnostic tensor IO. Input views are valid only for the duration of
// the synchronous callback. Publish takes ownership of inference results.
class ITensorListIO
{
public:
  using Callback = std::function<void (const std::vector<TensorView> &,
      const std_msgs::msg::Header &)>;

  virtual ~ITensorListIO() = default;
  virtual void Subscribe(Callback callback) = 0;
  virtual TensorMemoryKind OutputMemoryKind() const = 0;
  virtual void Publish(
    std::vector<OwnedTensor> tensors,
    const std_msgs::msg::Header & header) = 0;
};

// Factory: transport is "std" or "nitros". NITROS requires BUILD_NITROS_TRANSPORT.
std::unique_ptr<ITensorListIO> CreateTensorListIO(
  rclcpp::Node * node, const std::string & transport);

}  // namespace nvidia::isaac_ros::onnx_inference

#endif  // ISAAC_ROS_ONNX_INFERENCE__TENSOR_LIST_IO_HPP_
