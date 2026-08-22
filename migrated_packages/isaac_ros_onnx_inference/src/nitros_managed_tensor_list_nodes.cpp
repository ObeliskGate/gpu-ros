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
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"

#include "gpu_ros_managed_ros/managed_pub_sub.hpp"
#include "gpu_ros_managed_tensor_list/type_adapter.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"
#include "isaac_ros_onnx_inference/nitros_managed_tensor_list_adapter.hpp"

namespace nvidia::isaac_ros::onnx_inference
{
namespace
{
namespace nitros = nvidia::isaac_ros::nitros;

class TimingReporter
{
public:
  TimingReporter(rclcpp::Node * node, std::string boundary)
  : node_(node), boundary_(std::move(boundary)),
    enabled_(node_->declare_parameter<bool>(boundary_ + ".enable_timing", false)),
    log_every_(node_->declare_parameter<int>(boundary_ + ".timing_log_every", 500))
  {}

  void Record(std::chrono::steady_clock::time_point start)
  {
    if (!enabled_) {return;}
    const auto end = std::chrono::steady_clock::now();
    samples_ms_.push_back(static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000.0);
    if (log_every_ <= 0 || static_cast<int>(samples_ms_.size()) < log_every_) {return;}

    std::vector<double> sorted = samples_ms_;
    std::sort(sorted.begin(), sorted.end());
    const double mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) /
      static_cast<double>(sorted.size());
    const size_t p95_index = std::min(
      sorted.size() - 1, static_cast<size_t>(0.95 * static_cast<double>(sorted.size() - 1)));
    RCLCPP_INFO(
      node_->get_logger(),
          "%s boundary over %zu frames: mean=%.3f ms p95=%.3f ms; payload copies=0",
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
}  // namespace

class NitrosToManagedTensorListNode : public rclcpp::Node
{
public:
  explicit NitrosToManagedTensorListNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : rclcpp::Node("nitros_to_managed_tensor_list_node", options),
    gpu_device_id_(declare_parameter<int>("gpu_device_id", 0)),
    adapter_(gpu_device_id_), timing_(this, "nitros_to_managed")
  {
    publisher_ = std::make_unique<
      gpu_ros_managed::ManagedPublisher<gpu_ros_managed::ManagedTensorList>>(
      this, "tensor_output", rclcpp::QoS(10));
    subscription_ = std::make_shared<
      nitros::ManagedNitrosSubscriber<nitros::NitrosTensorListView>>(
      this, "tensor_input", NitrosTensorListFormat(),
      [this](const nitros::NitrosTensorListView & view) {OnTensorList(view);},
      nitros::NitrosDiagnosticsConfig{}, rclcpp::QoS(10));
  }

private:
  void OnTensorList(const nitros::NitrosTensorListView & view)
  {
    const auto start = std::chrono::steady_clock::now();
    publisher_->publish(adapter_.Convert(view));
    timing_.Record(start);
  }

  int gpu_device_id_;
  NitrosToManagedTensorListAdapter adapter_;
  TimingReporter timing_;
  std::unique_ptr<gpu_ros_managed::ManagedPublisher<gpu_ros_managed::ManagedTensorList>> publisher_;
  std::shared_ptr<nitros::ManagedNitrosSubscriber<nitros::NitrosTensorListView>> subscription_;
};

class ManagedToNitrosTensorListNode : public rclcpp::Node
{
public:
  explicit ManagedToNitrosTensorListNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : rclcpp::Node("managed_to_nitros_tensor_list_node", options),
    gpu_device_id_(declare_parameter<int>("gpu_device_id", 0)),
    timing_(this, "managed_to_nitros")
  {
    publisher_ = std::make_shared<nitros::ManagedNitrosPublisher<nitros::NitrosTensorList>>(
      this, "tensor_output", NitrosTensorListFormat(), nitros::NitrosDiagnosticsConfig{},
      rclcpp::QoS(10));
    subscription_ = std::make_unique<
      gpu_ros_managed::ManagedSubscriber<gpu_ros_managed::ManagedTensorListView>>(
      this, "tensor_input",
      [this](gpu_ros_managed::ManagedTensorListView input) {OnTensorList(std::move(input));},
      rclcpp::QoS(10));
  }

private:
  void OnTensorList(gpu_ros_managed::ManagedTensorListView input)
  {
    const auto start = std::chrono::steady_clock::now();
    publisher_->publish(BuildNitrosTensorList(std::move(input), gpu_device_id_));
    timing_.Record(start);
  }

  int gpu_device_id_;
  TimingReporter timing_;
  std::shared_ptr<nitros::ManagedNitrosPublisher<nitros::NitrosTensorList>> publisher_;
  std::unique_ptr<
    gpu_ros_managed::ManagedSubscriber<gpu_ros_managed::ManagedTensorListView>> subscription_;
};

}  // namespace nvidia::isaac_ros::onnx_inference

RCLCPP_COMPONENTS_REGISTER_NODE(
  nvidia::isaac_ros::onnx_inference::NitrosToManagedTensorListNode)
RCLCPP_COMPONENTS_REGISTER_NODE(
  nvidia::isaac_ros::onnx_inference::ManagedToNitrosTensorListNode)
