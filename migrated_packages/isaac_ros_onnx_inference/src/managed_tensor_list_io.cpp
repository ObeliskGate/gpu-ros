// Copyright 2026 Maintainer
// Licensed under the Apache License, Version 2.0.
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "gpu_ros_managed_ros/managed_pub_sub.hpp"
#include "gpu_ros_managed_tensor_list/tensor_list.hpp"
#include "gpu_ros_managed_tensor_list/type_adapter.hpp"
#include "isaac_ros_onnx_inference/onnx_dtype.hpp"
#include "isaac_ros_onnx_inference/tensor_list_io.hpp"

namespace nvidia::isaac_ros::onnx_inference
{
namespace
{
gpu_ros_managed::TensorDataType ToManagedType(ONNXTensorElementDataType dtype)
{
  return static_cast<gpu_ros_managed::TensorDataType>(OnnxToGxfDtype(dtype));
}

class ManagedTensorListIO final : public ITensorListIO
{
public:
  explicit ManagedTensorListIO(rclcpp::Node * node)
  : placement_(
      node->get_parameter("execution_provider").as_string() == "cuda" ?
      OutputPlacement::kDevice : OutputPlacement::kHost),
    publisher_(node, "tensor_output", rclcpp::QoS(10)),
    node_(node)
  {}

  void Subscribe(Callback callback) override
  {
    subscriber_ = std::make_unique<
      gpu_ros_managed::ManagedSubscriber<gpu_ros_managed::ManagedTensorListView>>(
      node_, "tensor_input", std::move(callback), rclcpp::QoS(10));
  }

  OutputPlacement output_placement() const noexcept override {return placement_;}

  void Publish(TensorListOutput && output) override
  {
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
    publisher_.publish(gpu_ros_managed::ManagedTensorList(
        std::move(output.header), std::move(tensors)));
  }

private:
  OutputPlacement placement_;
  gpu_ros_managed::ManagedPublisher<gpu_ros_managed::ManagedTensorList> publisher_;
  rclcpp::Node * node_;
  std::unique_ptr<
    gpu_ros_managed::ManagedSubscriber<gpu_ros_managed::ManagedTensorListView>> subscriber_;
};
}  // namespace

std::unique_ptr<ITensorListIO> CreateManagedTensorListIO(rclcpp::Node * node)
{
  return std::make_unique<ManagedTensorListIO>(node);
}
}  // namespace nvidia::isaac_ros::onnx_inference
