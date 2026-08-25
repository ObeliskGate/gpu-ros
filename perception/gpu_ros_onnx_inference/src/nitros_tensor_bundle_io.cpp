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

#include <functional>
#include <memory>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"
#include "gpu_ros_onnx_inference/nitros_managed_tensor_bundle_adapter.hpp"
#include "gpu_ros_onnx_inference/tensor_bundle_io.hpp"

namespace gpu_ros::onnx_inference
{
namespace
{
namespace nitros = nvidia::isaac_ros::nitros;

class NitrosTensorBundleIO final : public ITensorBundleIO
{
public:
  explicit NitrosTensorBundleIO(rclcpp::Node * node, bool publish_output)
  : node_(node),
    gpu_device_id_(node_->has_parameter("gpu_device_id") ?
      static_cast<int>(node_->get_parameter("gpu_device_id").as_int()) :
      node_->declare_parameter<int>("gpu_device_id", 0)),
    input_adapter_(gpu_device_id_)
  {
    if (publish_output) {
      publisher_ = std::make_shared<
        nitros::ManagedNitrosPublisher<nitros::NitrosTensorList>>(
        node_, "tensor_output", NitrosTensorBundleFormat(),
        nitros::NitrosDiagnosticsConfig{}, rclcpp::QoS(10));
    }
  }

  void Subscribe(Callback callback) override
  {
    callback_ = std::move(callback);
    subscription_ = std::make_shared<
      nitros::ManagedNitrosSubscriber<nitros::NitrosTensorListView>>(
      node_, "tensor_input", NitrosTensorBundleFormat(),
      [this](const nitros::NitrosTensorListView & view) {OnView(view);},
      nitros::NitrosDiagnosticsConfig{}, rclcpp::QoS(10));
  }

  OutputPlacement output_placement() const noexcept override
  {
    return OutputPlacement::kDevice;
  }

  void Publish(TensorBundleOutput && output) override
  {
    if (!publisher_) {
      throw std::logic_error("NITROS TensorBundle output is disabled for this IO");
    }
    try {
      publisher_->publish(BuildNitrosTensorBundle(std::move(output), gpu_device_id_));
    } catch (const std::exception & error) {
      RCLCPP_ERROR_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 5000,
        "Dropping NITROS output frame: %s", error.what());
    } catch (...) {
      RCLCPP_ERROR_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 5000,
        "Dropping NITROS output frame after unknown conversion failure");
    }
  }

private:
  void OnView(const nitros::NitrosTensorListView & view)
  {
    try {
      auto list =
        std::make_shared<gpu_ros_managed::ManagedTensorBundle>(input_adapter_.Convert(view));
      callback_(gpu_ros_managed::ManagedTensorBundleView(std::move(list)));
    } catch (const std::exception & error) {
      RCLCPP_ERROR_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 5000,
        "Dropping NITROS input frame: %s", error.what());
    } catch (...) {
      RCLCPP_ERROR_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 5000,
        "Dropping NITROS input frame after unknown conversion failure");
    }
  }

  rclcpp::Node * node_;
  Callback callback_;
  int gpu_device_id_;
  NitrosToManagedTensorBundleAdapter input_adapter_;
  std::shared_ptr<nitros::ManagedNitrosPublisher<nitros::NitrosTensorList>> publisher_;
  std::shared_ptr<nitros::ManagedNitrosSubscriber<nitros::NitrosTensorListView>> subscription_;
};
}  // namespace

std::unique_ptr<ITensorBundleIO> CreateNitrosTensorBundleIO(
  rclcpp::Node * node, bool publish_output)
{
  return std::make_unique<NitrosTensorBundleIO>(node, publish_output);
}

}  // namespace gpu_ros::onnx_inference
