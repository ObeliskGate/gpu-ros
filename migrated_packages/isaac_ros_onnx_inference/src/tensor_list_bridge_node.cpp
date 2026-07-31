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

// Transport bridge: forwards a TensorList from a std-ROS2 (or NITROS) input to
// a NITROS output, without inference. Needed to feed NITROS-only vendor nodes
// (e.g. TensorRTNode) from a std-ROS2 publisher, since std pub -> NITROS sub is
// not bridged automatically (only the reverse direction is).

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"

#include "isaac_ros_onnx_inference/tensor_list_io.hpp"

#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_builder.hpp"

namespace nvidia::isaac_ros::onnx_inference
{

namespace
{
namespace nitros = nvidia::isaac_ros::nitros;

nitros::NitrosDataType OnnxToNitrosDtype(ONNXTensorElementDataType dtype)
{
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return nitros::NitrosDataType::kFloat32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return nitros::NitrosDataType::kInt64;
    default:
      throw std::runtime_error("TensorListBridge: unsupported dtype");
  }
}
}  // namespace

// Subscribes via the configured input transport and republishes over NITROS.
class TensorListBridgeNode : public rclcpp::Node
{
public:
  explicit TensorListBridgeNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : rclcpp::Node("tensor_list_bridge_node", options)
  {
    const std::string input_transport =
      declare_parameter<std::string>("input_transport", "std");
    enable_timing_ = declare_parameter<bool>("enable_timing", false);
    timing_log_every_ = declare_parameter<int>("timing_log_every", 500);
    declare_parameter<int>("gpu_device_id", 0);

    cudaStreamCreate(&stream_);
    pub_ = std::make_shared<nitros::ManagedNitrosPublisher<nitros::NitrosTensorList>>(
      this, "tensor_output",
      nitros::nitros_tensor_list_nchw_rgb_f32_t::supported_type_name);

    input_io_ = CreateTensorListIO(this, input_transport);
    input_io_->Subscribe(
      [this](gpu_ros_managed::ManagedTensorListView tensors) {
        Forward(std::move(tensors));
      });
  }

  ~TensorListBridgeNode() override
  {
    cudaStreamDestroy(stream_);
  }

private:
  void Forward(gpu_ros_managed::ManagedTensorListView input)
  {
    std::chrono::steady_clock::time_point start;
    if (enable_timing_) {
      start = std::chrono::steady_clock::now();
    }

    nitros::NitrosTensorListBuilder builder;
    builder.WithHeader(input.header());
    for (const auto & tensor : input.tensors()) {
      const auto * host = std::get_if<gpu_ros_managed::HostBuffer>(&tensor.storage());
      if (host == nullptr) {
        throw std::invalid_argument("TensorListBridge only supports standard host-memory input");
      }
      void * gpu_buffer = nullptr;
      cudaMallocAsync(&gpu_buffer, tensor.byte_size(), stream_);
      cudaMemcpyAsync(
        gpu_buffer, host->data(), tensor.byte_size(), cudaMemcpyHostToDevice, stream_);
      std::vector<int32_t> dims(tensor.shape().begin(), tensor.shape().end());
      builder.AddTensor(
        tensor.name(),
        nitros::NitrosTensorBuilder()
        .WithShape(nitros::NitrosTensorShape(dims))
        .WithDataType(static_cast<nitros::NitrosDataType>(tensor.data_type()))
        .WithData(gpu_buffer)
        .Build());
    }
    cudaStreamSynchronize(stream_);
    pub_->publish(builder.Build());

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
      "TensorListBridge timing over %zu frames: mean=%.3f ms p95=%.3f ms max=%.3f ms",
      sorted.size(), mean, p95, max);
    timings_ms_.clear();
  }

  cudaStream_t stream_;
  bool enable_timing_{false};
  int timing_log_every_{500};
  std::vector<double> timings_ms_;
  std::unique_ptr<ITensorListIO> input_io_;
  std::shared_ptr<nitros::ManagedNitrosPublisher<nitros::NitrosTensorList>> pub_;
};

}  // namespace nvidia::isaac_ros::onnx_inference

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::onnx_inference::TensorListBridgeNode)
