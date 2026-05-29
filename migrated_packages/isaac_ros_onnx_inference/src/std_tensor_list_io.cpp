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

#include <stdexcept>
#include <string>
#include <vector>

#include "isaac_ros_onnx_inference/tensor_list_io.hpp"
#include "isaac_ros_onnx_inference/onnx_dtype.hpp"
#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"

namespace nvidia::isaac_ros::onnx_inference
{

namespace
{

using TensorListMsg = isaac_ros_tensor_list_interfaces::msg::TensorList;

class StdTensorListIO : public ITensorListIO
{
public:
  explicit StdTensorListIO(rclcpp::Node * node)
  : node_{node}
  {
    pub_ = node_->create_publisher<TensorListMsg>("tensor_output", 10);
  }

  void Subscribe(Callback callback) override
  {
    callback_ = std::move(callback);
    sub_ = node_->create_subscription<TensorListMsg>(
      "tensor_input", 10,
      [this](const TensorListMsg::SharedPtr msg) {OnMsg(msg);});
  }

  void Publish(
    const std::vector<HostTensor> & tensors,
    const std_msgs::msg::Header & header) override
  {
    TensorListMsg msg;
    msg.header = header;
    msg.tensors.reserve(tensors.size());
    for (const auto & ht : tensors) {
      isaac_ros_tensor_list_interfaces::msg::Tensor t;
      t.name = ht.name;
      t.data_type = OnnxToGxfDtype(ht.dtype);
      t.shape.rank = static_cast<int32_t>(ht.shape.size());
      t.shape.dims.assign(ht.shape.begin(), ht.shape.end());
      t.data = ht.data;
      msg.tensors.push_back(std::move(t));
    }
    pub_->publish(msg);
  }

private:
  void OnMsg(const TensorListMsg::SharedPtr msg)
  {
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
    callback_(inputs, msg->header);
  }

  rclcpp::Node * node_;
  Callback callback_;
  rclcpp::Subscription<TensorListMsg>::SharedPtr sub_;
  rclcpp::Publisher<TensorListMsg>::SharedPtr pub_;
};

}  // namespace

std::unique_ptr<ITensorListIO> CreateStdTensorListIO(rclcpp::Node * node)
{
  return std::make_unique<StdTensorListIO>(node);
}

}  // namespace nvidia::isaac_ros::onnx_inference
