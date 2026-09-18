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

#include "gpu_ros_onnx_inference/tensor_bundle_io.hpp"
#include "gpu_ros_onnx_inference/tensor_dtype.hpp"
#include "gpu_ros_tensor_bundle_msgs/msg/tensor_bundle.hpp"

namespace gpu_ros::onnx_inference
{

namespace
{

using TensorBundleMsg = gpu_ros_tensor_bundle_msgs::msg::TensorBundle;

class StdTensorBundleIO : public ITensorBundleIO
{
public:
  explicit StdTensorBundleIO(rclcpp::Node * node, bool publish_output) : node_{node}
  {
    if (publish_output) {
      pub_ = node_->create_publisher<TensorBundleMsg>("tensor_output", 10);
    }
  }

  void Subscribe(Callback callback) override
  {
    callback_ = std::move(callback);
    sub_ = node_->create_subscription<TensorBundleMsg>(
      "tensor_input", 10, [this](const TensorBundleMsg::SharedPtr msg) { OnMsg(msg); });
  }

  OutputPlacement output_placement() const noexcept override { return OutputPlacement::kHost; }

  void Publish(TensorBundleOutput && output) override
  {
    if (!pub_) {
      throw std::logic_error("standard TensorBundle output is disabled for this IO");
    }
    auto loaned_message = pub_->borrow_loaned_message();
    auto & msg = loaned_message.get();
    msg.header = std::move(output.header);
    msg.tensors.reserve(output.tensors.size());
    for (auto & tensor : output.tensors) {
      auto * host_data = std::get_if<std::vector<uint8_t>>(&tensor.storage);
      if (host_data == nullptr) {
        throw std::runtime_error("StdTensorBundleIO received a device-memory output tensor");
      }
      gpu_ros_tensor_bundle_msgs::msg::Tensor t;
      t.name = std::move(tensor.name);
      t.data_type = OnnxToBundleDtype(tensor.dtype);
      t.shape.assign(tensor.shape.begin(), tensor.shape.end());
      t.data = std::move(*host_data);
      msg.tensors.push_back(std::move(t));
    }
    pub_->publish(std::move(loaned_message));
  }

private:
  void OnMsg(const TensorBundleMsg::SharedPtr msg)
  {
    std::vector<gpu_ros_managed::ManagedTensor> inputs;
    inputs.reserve(msg->tensors.size());
    for (const auto & t : msg->tensors) {
      std::vector<int64_t> shape(t.shape.begin(), t.shape.end());
      inputs.push_back(gpu_ros_managed::ManagedTensor::from_host_external(t.name,
        static_cast<gpu_ros_managed::TensorDataType>(t.data_type), std::move(shape),
        std::static_pointer_cast<const void>(msg), t.data.data(), t.data.size()));
    }
    auto list =
      std::make_shared<gpu_ros_managed::ManagedTensorBundle>(msg->header, std::move(inputs));
    callback_(gpu_ros_managed::ManagedTensorBundleView(std::move(list)));
  }

  rclcpp::Node * node_;
  Callback callback_;
  rclcpp::Subscription<TensorBundleMsg>::SharedPtr sub_;
  rclcpp::Publisher<TensorBundleMsg>::SharedPtr pub_;
};

} // namespace

std::unique_ptr<ITensorBundleIO> CreateStdTensorBundleIO(rclcpp::Node * node, bool publish_output)
{
  return std::make_unique<StdTensorBundleIO>(node, publish_output);
}

} // namespace gpu_ros::onnx_inference
