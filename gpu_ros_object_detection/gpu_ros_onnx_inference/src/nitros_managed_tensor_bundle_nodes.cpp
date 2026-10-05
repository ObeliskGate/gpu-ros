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

#include <algorithm>
#include <chrono>
#include <exception>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"

#include "gpu_ros_managed_ros/managed_pub_sub.hpp"
#include "gpu_ros_managed_tensor_bundle/type_adapter.hpp"
#include "gpu_ros_nvidia_tensor_bundle_compat/tensor_list_buffer_adapter.hpp"
#include "gpu_ros_onnx_inference/nitros_managed_tensor_bundle_adapter.hpp"

namespace gpu_ros::onnx_inference
{
namespace
{

class TimingReporter
{
public:
  TimingReporter(rclcpp::Node * node, std::string boundary)
      : node_(node), boundary_(std::move(boundary)),
        enabled_(node_->declare_parameter<bool>(boundary_ + ".enable_timing", false)),
        log_every_(node_->declare_parameter<int>(boundary_ + ".timing_log_every", 500))
  {
  }

  void Record(std::chrono::steady_clock::time_point start)
  {
    if (!enabled_) {
      return;
    }
    const auto end = std::chrono::steady_clock::now();
    samples_ms_.push_back(
      static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) /
      1000.0);
    if (log_every_ <= 0 || static_cast<int>(samples_ms_.size()) < log_every_) {
      return;
    }

    std::vector<double> sorted = samples_ms_;
    std::sort(sorted.begin(), sorted.end());
    const double mean =
      std::accumulate(sorted.begin(), sorted.end(), 0.0) / static_cast<double>(sorted.size());
    const size_t p95_index = std::min(
      sorted.size() - 1, static_cast<size_t>(0.95 * static_cast<double>(sorted.size() - 1)));
    RCLCPP_INFO(node_->get_logger(), "%s boundary over %zu frames: mean=%.3f ms p95=%.3f ms",
      boundary_.c_str(), sorted.size(), mean, sorted[p95_index]);
    samples_ms_.clear();
  }

private:
  rclcpp::Node * node_;
  std::string boundary_;
  bool enabled_;
  int log_every_;
  std::vector<double> samples_ms_;
};
} // namespace

class NitrosToManagedTensorBundleNode : public rclcpp::Node
{
public:
  explicit NitrosToManagedTensorBundleNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
      : rclcpp::Node("nitros_to_managed_tensor_bundle_node", options),
        gpu_device_id_(declare_parameter<int>("gpu_device_id", 0)),
        adapter_(this, gpu_device_id_, "tensor_output", false), timing_(this, "nitros_to_managed")
  {
    publisher_ =
      std::make_unique<gpu_ros_managed::ManagedPublisher<gpu_ros_managed::ManagedTensorBundle>>(
        this, "tensor_output", rclcpp::QoS(10));
    adapter_.Subscribe(
      [this](gpu_ros_managed::ManagedTensorBundleView view) { OnTensorBundle(std::move(view)); });
  }
  ~NitrosToManagedTensorBundleNode() override { adapter_.Unsubscribe(); }

private:
  void OnTensorBundle(gpu_ros_managed::ManagedTensorBundleView view)
  {
    try {
      const auto start = std::chrono::steady_clock::now();
      publisher_->publish(
        std::make_unique<gpu_ros_managed::ManagedTensorBundle>(view.get()));
      timing_.Record(start);
    } catch (const std::exception & error) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
        "TensorList-to-Managed conversion dropped frame: %s", error.what());
    } catch (...) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
        "TensorList-to-Managed conversion dropped frame: unknown exception");
    }
  }

  int gpu_device_id_;
  gpu_ros::nvidia_tensor_bundle_compat::TensorListTransport adapter_;
  TimingReporter timing_;
  std::unique_ptr<gpu_ros_managed::ManagedPublisher<gpu_ros_managed::ManagedTensorBundle>>
    publisher_;
};

class ManagedToNitrosTensorBundleNode : public rclcpp::Node
{
public:
  explicit ManagedToNitrosTensorBundleNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
      : rclcpp::Node("managed_to_nitros_tensor_bundle_node", options),
        gpu_device_id_(declare_parameter<int>("gpu_device_id", 0)),
        timing_(this, "managed_to_nitros")
  {
    publisher_ = std::make_unique<gpu_ros::nvidia_tensor_bundle_compat::TensorListTransport>(
      this, gpu_device_id_);
    subscription_ = std::make_unique<
      gpu_ros_managed::ManagedSubscriber<gpu_ros_managed::ManagedTensorBundleView>>(
      this, "tensor_input",
      [this](gpu_ros_managed::ManagedTensorBundleView input) { OnTensorBundle(std::move(input)); },
      rclcpp::QoS(10));
  }

private:
  void OnTensorBundle(gpu_ros_managed::ManagedTensorBundleView input)
  {
    try {
      const auto start = std::chrono::steady_clock::now();
      publisher_->Publish(input.get());
      timing_.Record(start);
    } catch (const std::exception & error) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
        "Managed-to-TensorList conversion dropped frame: %s", error.what());
    } catch (...) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
        "Managed-to-TensorList conversion dropped frame: unknown exception");
    }
  }

  int gpu_device_id_;
  TimingReporter timing_;
  std::unique_ptr<gpu_ros::nvidia_tensor_bundle_compat::TensorListTransport> publisher_;
  std::unique_ptr<gpu_ros_managed::ManagedSubscriber<gpu_ros_managed::ManagedTensorBundleView>>
    subscription_;
};

} // namespace gpu_ros::onnx_inference

RCLCPP_COMPONENTS_REGISTER_NODE(gpu_ros::onnx_inference::NitrosToManagedTensorBundleNode)
RCLCPP_COMPONENTS_REGISTER_NODE(gpu_ros::onnx_inference::ManagedToNitrosTensorBundleNode)
