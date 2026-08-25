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

#include "gpu_ros_managed_ros/managed_pub_sub.hpp"
#include "gpu_ros_managed_tensor_bundle/tensor_bundle.hpp"
#include "gpu_ros_managed_tensor_bundle/type_adapter.hpp"
#include "gpu_ros_onnx_inference/tensor_dtype.hpp"
#include "gpu_ros_onnx_inference/tensor_bundle_io.hpp"

namespace gpu_ros::onnx_inference
{
namespace
{
gpu_ros_managed::TensorDataType ToManagedType(ONNXTensorElementDataType dtype)
{
  return static_cast<gpu_ros_managed::TensorDataType>(OnnxToBundleDtype(dtype));
}

class ManagedTensorBundleIO final : public ITensorBundleIO
{
public:
  explicit ManagedTensorBundleIO(rclcpp::Node * node, bool publish_output)
  : placement_(
      (node->get_parameter("execution_provider").as_string() == "cuda" ||
      node->get_parameter("execution_provider").as_string() == "migraphx") ?
      OutputPlacement::kDevice : OutputPlacement::kHost),
    node_(node)
  {
    if (publish_output) {
      publisher_ = std::make_unique<
        gpu_ros_managed::ManagedPublisher<gpu_ros_managed::ManagedTensorBundle>>(
        node, "tensor_output", rclcpp::QoS(10));
    }
  }

  void Subscribe(Callback callback) override
  {
    subscriber_ = std::make_unique<
      gpu_ros_managed::ManagedSubscriber<gpu_ros_managed::ManagedTensorBundleView>>(
      node_, "tensor_input", std::move(callback), rclcpp::QoS(10));
  }

  OutputPlacement output_placement() const noexcept override {return placement_;}

  void Publish(TensorBundleOutput && output) override
  {
    if (!publisher_) {
      throw std::logic_error("managed TensorBundle output is disabled for this IO");
    }
    std::vector<gpu_ros_managed::ManagedTensor> tensors;
    tensors.reserve(output.tensors.size());
    for (auto & tensor : output.tensors) {
      if (auto * host = std::get_if<std::vector<uint8_t>>(&tensor.storage)) {
        auto owner = std::make_shared<std::vector<uint8_t>>(std::move(*host));
        tensors.push_back(gpu_ros_managed::ManagedTensor::from_host_external(
            std::move(tensor.name), ToManagedType(tensor.dtype), std::move(tensor.shape),
            owner, owner->data(), owner->size()));
      } else {
        tensors.emplace_back(
          std::move(tensor.name), ToManagedType(tensor.dtype), std::move(tensor.shape),
          std::get<std::shared_ptr<gpu_ros_managed::DeviceBuffer>>(std::move(tensor.storage)));
      }
    }
    publisher_->publish(gpu_ros_managed::ManagedTensorBundle(
        std::move(output.header), std::move(tensors)));
  }

private:
  OutputPlacement placement_;
  std::unique_ptr<gpu_ros_managed::ManagedPublisher<gpu_ros_managed::ManagedTensorBundle>>
  publisher_;
  rclcpp::Node * node_;
  std::unique_ptr<
    gpu_ros_managed::ManagedSubscriber<gpu_ros_managed::ManagedTensorBundleView>> subscriber_;
};
}  // namespace

std::unique_ptr<ITensorBundleIO> CreateManagedTensorBundleIO(
  rclcpp::Node * node, bool publish_output)
{
  return std::make_unique<ManagedTensorBundleIO>(node, publish_output);
}
}  // namespace gpu_ros::onnx_inference
