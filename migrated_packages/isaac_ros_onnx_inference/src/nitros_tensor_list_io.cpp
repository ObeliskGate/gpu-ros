// Copyright 2026 Maintainer
// Licensed under the Apache License, Version 2.0.
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"
#include "isaac_ros_onnx_inference/nitros_managed_tensor_list_adapter.hpp"
#include "isaac_ros_onnx_inference/tensor_list_io.hpp"

namespace nvidia::isaac_ros::onnx_inference
{
namespace
{
namespace nitros = nvidia::isaac_ros::nitros;

class NitrosTensorListIO final : public ITensorListIO
{
public:
  explicit NitrosTensorListIO(rclcpp::Node * node)
  : node_(node),
    gpu_device_id_(node_->has_parameter("gpu_device_id") ?
      static_cast<int>(node_->get_parameter("gpu_device_id").as_int()) :
      node_->declare_parameter<int>("gpu_device_id", 0)),
    input_adapter_(gpu_device_id_)
  {
    publisher_ = std::make_shared<nitros::ManagedNitrosPublisher<nitros::NitrosTensorList>>(
      node_, "tensor_output", NitrosTensorListFormat(), nitros::NitrosDiagnosticsConfig{},
      rclcpp::QoS(10));
  }

  void Subscribe(Callback callback) override
  {
    callback_ = std::move(callback);
    subscription_ = std::make_shared<
      nitros::ManagedNitrosSubscriber<nitros::NitrosTensorListView>>(
      node_, "tensor_input", NitrosTensorListFormat(),
      [this](const nitros::NitrosTensorListView & view) {OnView(view);},
      nitros::NitrosDiagnosticsConfig{}, rclcpp::QoS(10));
  }

  OutputPlacement output_placement() const noexcept override
  {
    return OutputPlacement::kDevice;
  }

  void Publish(TensorListOutput && output) override
  {
    publisher_->publish(BuildNitrosTensorList(std::move(output), gpu_device_id_));
  }

private:
  void OnView(const nitros::NitrosTensorListView & view)
  {
    auto list = std::make_shared<gpu_ros_managed::ManagedTensorList>(input_adapter_.Convert(view));
    callback_(gpu_ros_managed::ManagedTensorListView(std::move(list)));
  }

  rclcpp::Node * node_;
  Callback callback_;
  int gpu_device_id_;
  NitrosToManagedTensorListAdapter input_adapter_;
  std::shared_ptr<nitros::ManagedNitrosPublisher<nitros::NitrosTensorList>> publisher_;
  std::shared_ptr<nitros::ManagedNitrosSubscriber<nitros::NitrosTensorListView>> subscription_;
};
}  // namespace

std::unique_ptr<ITensorListIO> CreateNitrosTensorListIO(rclcpp::Node * node)
{
  return std::make_unique<NitrosTensorListIO>(node);
}

}  // namespace nvidia::isaac_ros::onnx_inference
