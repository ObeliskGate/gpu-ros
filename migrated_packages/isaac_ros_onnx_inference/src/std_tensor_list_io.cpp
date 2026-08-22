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

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
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

  OutputPlacement output_placement() const noexcept override
  {
    return OutputPlacement::kHost;
  }

  void Publish(TensorListOutput && output) override
  {
    auto loaned_message = pub_->borrow_loaned_message();
    auto & msg = loaned_message.get();
    msg.header = std::move(output.header);
    msg.tensors.reserve(output.tensors.size());
    for (auto & tensor : output.tensors) {
      auto * host_data = std::get_if<std::vector<uint8_t>>(&tensor.storage);
      if (host_data == nullptr) {
        throw std::runtime_error("StdTensorListIO received a device-memory output tensor");
      }
      isaac_ros_tensor_list_interfaces::msg::Tensor t;
      t.name = std::move(tensor.name);
      t.data_type = OnnxToGxfDtype(tensor.dtype);
      t.shape.rank = static_cast<int32_t>(tensor.shape.size());
      t.shape.dims.assign(tensor.shape.begin(), tensor.shape.end());
      t.strides = gpu_ros_managed::contiguous_strides(
        tensor.shape,
        static_cast<gpu_ros_managed::TensorDataType>(OnnxToGxfDtype(tensor.dtype)));
      t.data = std::move(*host_data);
      msg.tensors.push_back(std::move(t));
    }
    pub_->publish(std::move(loaned_message));
  }

private:
  void OnMsg(const TensorListMsg::SharedPtr msg)
  {
    std::vector<gpu_ros_managed::ManagedTensor> inputs;
    inputs.reserve(msg->tensors.size());
    for (const auto & t : msg->tensors) {
      std::vector<int64_t> shape(t.shape.dims.begin(), t.shape.dims.end());
      if (t.shape.rank != shape.size()) {
        throw std::invalid_argument("StdTensorListIO tensor rank does not match dims");
      }
      inputs.push_back(gpu_ros_managed::ManagedTensor::from_host_external(
          t.name, static_cast<gpu_ros_managed::TensorDataType>(t.data_type),
          std::move(shape), std::static_pointer_cast<const void>(msg),
          t.data.data(), t.data.size(), t.strides));
    }
    auto list = std::make_shared<gpu_ros_managed::ManagedTensorList>(
      msg->header, std::move(inputs));
    callback_(gpu_ros_managed::ManagedTensorListView(std::move(list)));
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
