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

// Explicit standard-host to native TensorList bridge. The upload is an expected H2D boundary.

#include <algorithm>
#include <chrono>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"

#include "gpu_ros_nvidia_tensor_bundle_compat/native_tensor_list.hpp"
#include "gpu_ros_onnx_inference/tensor_contract.hpp"
#include "gpu_ros_onnx_inference/tensor_bundle_io.hpp"

namespace gpu_ros::onnx_inference
{

// Subscribes via the configured input transport and republishes native TensorList.
class TensorBundleBridgeNode : public rclcpp::Node
{
public:
  explicit TensorBundleBridgeNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
      : rclcpp::Node("tensor_bundle_bridge_node", options)
  {
    const std::string input_transport = declare_parameter<std::string>("input_transport", "std");
    enable_timing_ = declare_parameter<bool>("enable_timing", false);
    timing_log_every_ = declare_parameter<int>("timing_log_every", 500);
    gpu_device_id_ = declare_parameter<int>("gpu_device_id", 0);
    if (gpu_device_id_ < 0) {
      throw std::invalid_argument("gpu_device_id must be non-negative");
    }
    pub_ = std::make_unique<gpu_ros::nvidia_tensor_bundle_compat::native::TensorListTransport>(
      this, gpu_device_id_);

    // The input IO must not create a second output publisher.
    input_io_ = CreateTensorBundleIO(this, input_transport, false);
    input_io_->Subscribe(
      [this](TensorBindingBatch tensors) { Forward(std::move(tensors)); });
  }

  ~TensorBundleBridgeNode() override
  {
    input_io_.reset();
    pub_.reset();
  }

private:
  void Forward(TensorBindingBatch input)
  {
    try {
      ForwardOrThrow(std::move(input));
    } catch (const std::exception & error) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 5000, "Dropping TensorBundle bridge frame: %s", error.what());
    } catch (...) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
        "Dropping TensorBundle bridge frame after unknown conversion failure");
    }
  }

  void ForwardOrThrow(TensorBindingBatch input)
  {
    std::chrono::steady_clock::time_point start;
    if (enable_timing_) {
      start = std::chrono::steady_clock::now();
    }

    namespace wire = gpu_ros::nvidia_tensor_bundle_compat::native;
    auto owner = std::make_shared<TensorBindingBatch>(std::move(input));
    std::vector<wire::TensorSpec> specs;
    std::vector<wire::CopySource> sources;
    for (const auto & tensor : owner->bindings) {
      const auto type = tensor.value.GetTensorTypeAndShapeInfo();
      const auto dtype = type.GetElementType();
      if (dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        throw std::invalid_argument("bridge requires float32 or int64");
      }
      specs.push_back({tensor.name, static_cast<uint8_t>(dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? 2 : 0),
        static_cast<uint8_t>(dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? 32 : 64), 1, type.GetShape()});
      sources.push_back({tensor.data, tensor.byte_count,
        tensor.value.GetTensorMemoryInfo().GetDeviceType() == OrtMemoryInfoDeviceType_GPU});
    }
    auto batch = pub_->Allocate(owner->header, specs);
    batch.CopyFrom(sources, owner);
    pub_->Publish(batch.message());

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
    if (timing_log_every_ <= 0 || static_cast<int>(timings_ms_.size()) < timing_log_every_) {
      return;
    }

    std::vector<double> sorted = timings_ms_;
    std::sort(sorted.begin(), sorted.end());
    const double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
    const double mean = sum / static_cast<double>(sorted.size());
    const double p95 = sorted.at(std::min(
      sorted.size() - 1, static_cast<size_t>(0.95 * static_cast<double>(sorted.size() - 1))));
    const double max = sorted.back();

    RCLCPP_INFO(get_logger(),
      "TensorBundleBridge timing over %zu frames: mean=%.3f ms p95=%.3f ms max=%.3f ms",
      sorted.size(), mean, p95, max);
    timings_ms_.clear();
  }

  int gpu_device_id_{0};
  bool enable_timing_{false};
  int timing_log_every_{500};
  std::vector<double> timings_ms_;
  std::unique_ptr<ITensorBundleIO> input_io_;
  std::unique_ptr<gpu_ros::nvidia_tensor_bundle_compat::native::TensorListTransport> pub_;
};

} // namespace gpu_ros::onnx_inference

RCLCPP_COMPONENTS_REGISTER_NODE(gpu_ros::onnx_inference::TensorBundleBridgeNode)
