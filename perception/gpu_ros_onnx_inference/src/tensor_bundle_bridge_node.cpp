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

// Transport bridge: forwards a TensorBundle from a std-ROS2 (or NITROS) input to
// a NITROS output, without inference. Needed to feed NITROS-only vendor nodes
// (e.g. TensorRTNode) from a std-ROS2 publisher, since std pub -> NITROS sub is
// not bridged automatically (only the reverse direction is).

#include <algorithm>
#include <chrono>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"

#include "gpu_ros_managed_cuda/cuda_backend.hpp"
#include "gpu_ros_managed_tensor_bundle/tensor_bundle.hpp"
#include "gpu_ros_onnx_inference/nitros_managed_tensor_bundle_adapter.hpp"
#include "gpu_ros_onnx_inference/tensor_bundle_io.hpp"

#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"

namespace gpu_ros::onnx_inference
{

namespace
{
namespace nitros = nvidia::isaac_ros::nitros;
}  // namespace

// Subscribes via the configured input transport and republishes over NITROS.
class TensorBundleBridgeNode : public rclcpp::Node
{
public:
  explicit TensorBundleBridgeNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : rclcpp::Node("tensor_bundle_bridge_node", options)
  {
    const std::string input_transport =
      declare_parameter<std::string>("input_transport", "std");
    enable_timing_ = declare_parameter<bool>("enable_timing", false);
    timing_log_every_ = declare_parameter<int>("timing_log_every", 500);
    gpu_device_id_ = declare_parameter<int>("gpu_device_id", 0);
    if (gpu_device_id_ < 0) {
      throw std::invalid_argument("gpu_device_id must be non-negative");
    }

    pub_ = std::make_shared<nitros::ManagedNitrosPublisher<nitros::NitrosTensorList>>(
      this, "tensor_output",
      nitros::nitros_tensor_list_nchw_rgb_f32_t::supported_type_name);

    // The bridge owns its NITROS output publisher. The transport IO is input
    // only; creating its normal tensor_output publisher would collide with
    // the remapped NITROS output topic.
    input_io_ = CreateTensorBundleIO(this, input_transport, false);
    input_io_->Subscribe(
      [this](gpu_ros_managed::ManagedTensorBundleView tensors) {
        Forward(std::move(tensors));
      });
  }

  ~TensorBundleBridgeNode() override = default;

private:
  void Forward(gpu_ros_managed::ManagedTensorBundleView input)
  {
    try {
      ForwardOrThrow(std::move(input));
    } catch (const std::exception & error) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Dropping TensorBundle bridge frame: %s", error.what());
    } catch (...) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Dropping TensorBundle bridge frame after unknown conversion failure");
    }
  }

  void ForwardOrThrow(gpu_ros_managed::ManagedTensorBundleView input)
  {
    std::chrono::steady_clock::time_point start;
    if (enable_timing_) {
      start = std::chrono::steady_clock::now();
    }

    std::vector<gpu_ros_managed::ManagedTensor> tensors;
    tensors.reserve(input.tensors().size());
    for (const auto & tensor : input.tensors()) {
      const auto * host = std::get_if<gpu_ros_managed::HostBuffer>(&tensor.storage());
      if (host == nullptr) {
        throw std::invalid_argument("TensorBundleBridge only supports standard host-memory input");
      }
      auto device = gpu_ros_managed::cuda::allocate(tensor.byte_size(), gpu_device_id_);
      device->copy_from_host_blocking(host->data(), tensor.byte_size());
      tensors.emplace_back(
        tensor.name(), tensor.data_type(), tensor.shape(), std::move(device), tensor.strides());
    }
    auto bundle = std::make_shared<gpu_ros_managed::ManagedTensorBundle>(
      input.header(), std::move(tensors));
    pub_->publish(BuildNitrosTensorBundle(
      gpu_ros_managed::ManagedTensorBundleView(std::move(bundle)), gpu_device_id_));

    if (enable_timing_) {
      const auto end = std::chrono::steady_clock::now();
      const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      RecordTiming(static_cast<double>(elapsed_us) / 1000.0);
    }
  }

  void RecordTiming(double elapsed_ms)
  {
    timings_ms_.push_back(elapsed_ms);
    if (timing_log_every_ <= 0 ||
      static_cast<int>(timings_ms_.size()) < timing_log_every_)
    {
      return;
    }

    std::vector<double> sorted = timings_ms_;
    std::sort(sorted.begin(), sorted.end());
    const double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
    const double mean = sum / static_cast<double>(sorted.size());
    const double p95 = sorted.at(
      std::min(
        sorted.size() - 1,
        static_cast<size_t>(0.95 * static_cast<double>(sorted.size() - 1))));
    const double max = sorted.back();

    RCLCPP_INFO(
      get_logger(),
      "TensorBundleBridge timing over %zu frames: mean=%.3f ms p95=%.3f ms max=%.3f ms",
      sorted.size(), mean, p95, max);
    timings_ms_.clear();
  }

  int gpu_device_id_{0};
  bool enable_timing_{false};
  int timing_log_every_{500};
  std::vector<double> timings_ms_;
  std::unique_ptr<ITensorBundleIO> input_io_;
  std::shared_ptr<nitros::ManagedNitrosPublisher<nitros::NitrosTensorList>> pub_;
};

}  // namespace gpu_ros::onnx_inference

RCLCPP_COMPONENTS_REGISTER_NODE(gpu_ros::onnx_inference::TensorBundleBridgeNode)
