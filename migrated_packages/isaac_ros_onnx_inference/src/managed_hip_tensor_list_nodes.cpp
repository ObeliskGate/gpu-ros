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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"

#include "gpu_ros_managed_hip/hip_backend.hpp"
#include "gpu_ros_managed_ros/managed_pub_sub.hpp"
#include "gpu_ros_managed_tensor_list/tensor_list.hpp"
#include "gpu_ros_managed_tensor_list/type_adapter.hpp"
#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"

namespace nvidia::isaac_ros::onnx_inference
{
using TensorListMsg = isaac_ros_tensor_list_interfaces::msg::TensorList;

gpu_ros_managed::ManagedTensor MakeManagedHipTensor(
  const isaac_ros_tensor_list_interfaces::msg::Tensor & source, int gpu_device_id)
{
  std::vector<int64_t> shape(source.shape.dims.begin(), source.shape.dims.end());
  if (source.shape.rank != shape.size()) {
    throw std::invalid_argument(
            "Standard TensorList tensor '" + source.name + "' rank does not match dims");
  }

  const auto dtype = static_cast<gpu_ros_managed::TensorDataType>(source.data_type);
  const size_t bytes = gpu_ros_managed::tensor_byte_size(shape, dtype);
  if (source.data.size() < bytes) {
    throw std::invalid_argument(
            "Standard TensorList tensor '" + source.name + "' has insufficient data");
  }

  auto buffer = gpu_ros_managed::hip::allocate(bytes, gpu_device_id);
  buffer->copy_from_host_blocking(source.data.data(), bytes);
  return gpu_ros_managed::ManagedTensor(
    source.name, dtype, std::move(shape), std::move(buffer), source.strides);
}

void CheckHipDevice(
  const gpu_ros_managed::ManagedTensor & tensor, int gpu_device_id)
{
  const auto * buffer = std::get_if<std::shared_ptr<gpu_ros_managed::DeviceBuffer>>(
    &tensor.storage());
  if (buffer == nullptr || !*buffer) {
    throw std::invalid_argument(
            "Managed HIP TensorList tensor '" + tensor.name() + "' is not device-backed");
  }
  if ((*buffer)->device_id() !=
    gpu_ros_managed::DeviceId{gpu_ros_managed::BackendKind::kHip, gpu_device_id})
  {
    throw std::invalid_argument(
            "Managed HIP TensorList tensor '" + tensor.name() +
            "' is on the wrong backend or device");
  }
}

class StdToManagedHipTensorListNode final : public rclcpp::Node
{
public:
  explicit StdToManagedHipTensorListNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : rclcpp::Node("std_to_managed_hip_tensor_list_node", options),
    gpu_device_id_(declare_parameter<int>("gpu_device_id", 0)),
    publisher_(this, "tensor_output", rclcpp::QoS(10))
  {
    subscription_ = create_subscription<TensorListMsg>(
      "tensor_input", rclcpp::QoS(10),
      [this](const TensorListMsg::SharedPtr message) {OnTensorList(message);});
  }

private:
  void OnTensorList(const TensorListMsg::SharedPtr & message)
  {
    try {
      std::vector<gpu_ros_managed::ManagedTensor> tensors;
      tensors.reserve(message->tensors.size());
      for (const auto & tensor : message->tensors) {
        tensors.push_back(MakeManagedHipTensor(tensor, gpu_device_id_));
      }
      publisher_.publish(gpu_ros_managed::ManagedTensorList(
          message->header, std::move(tensors)));
    } catch (const std::exception & error) {
      RCLCPP_ERROR(get_logger(), "Failed to stage standard TensorList to HIP: %s", error.what());
    }
  }

  int gpu_device_id_;
  gpu_ros_managed::ManagedPublisher<gpu_ros_managed::ManagedTensorList> publisher_;
  rclcpp::Subscription<TensorListMsg>::SharedPtr subscription_;
};

class ManagedHipToStdTensorListNode final : public rclcpp::Node
{
public:
  explicit ManagedHipToStdTensorListNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : rclcpp::Node("managed_hip_to_std_tensor_list_node", options),
    gpu_device_id_(declare_parameter<int>("gpu_device_id", 0)),
    publisher_(create_publisher<TensorListMsg>("tensor_output", rclcpp::QoS(10)))
  {
    subscription_ = std::make_unique<
      gpu_ros_managed::ManagedSubscriber<gpu_ros_managed::ManagedTensorListView>>(
      this, "tensor_input",
      [this](gpu_ros_managed::ManagedTensorListView input) {
        OnTensorList(std::move(input));
      }, rclcpp::QoS(10));
  }

private:
  void OnTensorList(gpu_ros_managed::ManagedTensorListView input)
  {
    try {
      auto output = std::make_unique<TensorListMsg>();
      output->header = input.header();
      output->tensors.reserve(input.tensors().size());
      for (const auto & tensor : input.tensors()) {
        CheckHipDevice(tensor, gpu_device_id_);
        const auto & buffer =
          std::get<std::shared_ptr<gpu_ros_managed::DeviceBuffer>>(tensor.storage());
        auto & message_tensor = output->tensors.emplace_back();
        message_tensor.name = tensor.name();
        message_tensor.data_type = static_cast<int32_t>(tensor.data_type());
        message_tensor.shape.rank = static_cast<uint8_t>(tensor.shape().size());
        message_tensor.shape.dims.assign(tensor.shape().begin(), tensor.shape().end());
        message_tensor.strides = tensor.strides();
        message_tensor.data.resize(tensor.byte_size());
        buffer->copy_to_host_blocking(message_tensor.data.data(), message_tensor.data.size());
      }
      publisher_->publish(std::move(output));
    } catch (const std::exception & error) {
      RCLCPP_ERROR(get_logger(), "Failed to stage Managed HIP TensorList to host: %s",
          error.what());
    }
  }

  int gpu_device_id_;
  rclcpp::Publisher<TensorListMsg>::SharedPtr publisher_;
  std::unique_ptr<
    gpu_ros_managed::ManagedSubscriber<gpu_ros_managed::ManagedTensorListView>> subscription_;
};
}  // namespace nvidia::isaac_ros::onnx_inference

RCLCPP_COMPONENTS_REGISTER_NODE(
  nvidia::isaac_ros::onnx_inference::StdToManagedHipTensorListNode)
RCLCPP_COMPONENTS_REGISTER_NODE(
  nvidia::isaac_ros::onnx_inference::ManagedHipToStdTensorListNode)
