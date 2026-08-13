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

#include "isaac_ros_onnx_inference/onnx_inference_node.hpp"

#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "rclcpp_components/register_node_macro.hpp"

namespace nvidia::isaac_ros::onnx_inference
{

OnnxInferenceNode::OnnxInferenceNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("onnx_inference_node", options)
{
  const std::string model_file_path =
    declare_parameter<std::string>("model_file_path", "");
  const std::string ep_str =
    declare_parameter<std::string>("execution_provider", "cuda");
  const int gpu_device_id =
    declare_parameter<int>("gpu_device_id", 0);
  const std::string ort_profile_prefix =
    declare_parameter<std::string>("ort_profile_prefix", "");
  const int64_t ort_profile_frames =
    declare_parameter<int64_t>("ort_profile_frames", 0);
  const std::string binding_report_path =
    declare_parameter<std::string>("binding_report_path", "");
  const std::string transport =
    declare_parameter<std::string>("transport", "std");
  const std::string managed_io_contract =
    declare_parameter<std::string>("managed_io_contract", "");
  const auto managed_input_contracts =
    declare_parameter<std::vector<std::string>>("managed_input_contracts", {});
  const auto managed_output_contracts =
    declare_parameter<std::vector<std::string>>("managed_output_contracts", {});
  const int64_t managed_pool_capacity =
    declare_parameter<int64_t>("managed_pool_capacity", 16);
  const int64_t managed_pool_wait_timeout_ms =
    declare_parameter<int64_t>("managed_pool_wait_timeout_ms", 100);
  const ExecutionProvider execution_provider = ParseExecutionProvider(ep_str);

  if (ort_profile_frames < 0) {
    throw std::invalid_argument("ort_profile_frames must be non-negative");
  }
  if (ort_profile_frames > 0 && ort_profile_prefix.empty()) {
    throw std::invalid_argument(
            "ort_profile_frames requires a non-empty ort_profile_prefix");
  }
  if (gpu_device_id < 0 || managed_pool_capacity <= 0 ||
    static_cast<uint64_t>(managed_pool_capacity) > std::numeric_limits<size_t>::max() ||
    managed_pool_wait_timeout_ms < 0)
  {
    throw std::invalid_argument(
            "gpu_device_id must be non-negative, managed_pool_capacity must be positive, "
            "and managed_pool_wait_timeout_ms must be non-negative");
  }
  ort_profile_frames_ = static_cast<size_t>(ort_profile_frames);

  if (transport == "nitros" && execution_provider != ExecutionProvider::kCuda) {
    throw std::invalid_argument("transport=nitros requires execution_provider=cuda");
  }
  if (!managed_io_contract.empty() && managed_io_contract != "hip_managed_strict") {
    throw std::invalid_argument(
            "managed_io_contract must be empty or hip_managed_strict");
  }
  if (managed_io_contract == "hip_managed_strict" && model_file_path.empty()) {
    throw std::invalid_argument(
            "hip_managed_strict requires a non-empty model_file_path");
  }

  io_ = CreateTensorListIO(this, transport);
  io_->Subscribe(
    [this](gpu_ros_managed::ManagedTensorListView inputs) {
      OnTensors(std::move(inputs));
    });

  if (model_file_path.empty()) {
    RCLCPP_WARN(get_logger(), "model_file_path is empty — inference core not initialized.");
    return;
  }

  OnnxInferenceCore::Config cfg;
  cfg.model_file_path = model_file_path;
  cfg.ep = execution_provider;
  cfg.gpu_device_id = gpu_device_id;
  cfg.ort_profile_prefix = ort_profile_prefix;
  cfg.binding_report_path = binding_report_path;
  cfg.transport = transport;
  cfg.managed_io_contract = managed_io_contract;
  cfg.managed_input_contracts = managed_input_contracts;
  cfg.managed_output_contracts = managed_output_contracts;
  cfg.managed_pool_capacity = static_cast<size_t>(managed_pool_capacity);
  cfg.managed_pool_wait_timeout = std::chrono::milliseconds(managed_pool_wait_timeout_ms);
  core_ = std::make_unique<OnnxInferenceCore>(cfg);

  RCLCPP_INFO(
    get_logger(),
    "Loaded model '%s' with %zu inputs, %zu outputs, EP=%s, transport=%s, "
    "managed_io_contract=%s",
    model_file_path.c_str(),
    core_->GetInputCount(),
    core_->GetOutputCount(),
    ep_str.c_str(),
    transport.c_str(), managed_io_contract.empty() ? "compat" : managed_io_contract.c_str());

  const std::string output_probe = core_->OutputBindingProbeReport();
  if (!output_probe.empty()) {
    RCLCPP_INFO(get_logger(), "Output binding plan: %s", output_probe.c_str());
  }

  if (core_->IsProfilingEnabled()) {
    RCLCPP_INFO(
      get_logger(),
      "ONNX Runtime profiling enabled with prefix '%s'. The profile is finalized on shutdown.",
      ort_profile_prefix.c_str());
  }
}

OnnxInferenceNode::~OnnxInferenceNode()
{
  // Destroy the subscriber/publisher before draining fixed pools so no ROS
  // callback can retain a TensorList buffer during pool shutdown.
  io_.reset();
  if (core_ && !core_->shutdown(std::chrono::seconds(5))) {
    RCLCPP_ERROR(
      get_logger(),
      "Managed output pool shutdown did not drain within the configured timeout; "
      "buffers remain orphan-safe and were not force-released.");
  }
  FinalizeOrtProfile("shutdown");
}

void OnnxInferenceNode::FinalizeOrtProfile(const char * reason) noexcept
{
  if (!core_ || !core_->IsProfilingEnabled()) {
    return;
  }

  try {
    const std::string profile_path = core_->EndProfiling();
    RCLCPP_INFO(
      get_logger(), "ONNX Runtime profile written to '%s' after %zu frames (%s).",
      profile_path.c_str(), inference_count_, reason);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Failed to finalize ONNX Runtime profile: %s", e.what());
  }
}

void OnnxInferenceNode::OnTensors(gpu_ros_managed::ManagedTensorListView inputs)
{
  if (!core_) {
    RCLCPP_WARN_ONCE(get_logger(), "Received tensor but inference core is not initialized.");
    return;
  }

  std::lock_guard<std::mutex> lock(inference_mutex_);
  try {
    TensorListOutput output;
    output.header = inputs.header();
    output.tensors = core_->RunInference(std::move(inputs), io_->output_placement());
    ++inference_count_;
    if (!output_probe_runtime_logged_) {
      const std::string output_probe = core_->OutputBindingProbeReport();
      if (!output_probe.empty()) {
        RCLCPP_INFO(get_logger(), "Output binding result: %s", output_probe.c_str());
      }
      output_probe_runtime_logged_ = true;
    }
    if (ort_profile_frames_ > 0 && inference_count_ >= ort_profile_frames_) {
      FinalizeOrtProfile("configured frame limit");
    }
    io_->Publish(std::move(output));
  } catch (const Ort::Exception & e) {
    const char * message = e.what();
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "ONNX Runtime dropped an input frame: error_code=%d message=%s",
      static_cast<int>(e.GetOrtErrorCode()),
      (message != nullptr && message[0] != '\0') ? message : "<empty>");
  } catch (const std::exception & e) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "Inference dropped an input frame: %s", e.what());
  } catch (...) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "Inference dropped an input frame after an unknown exception");
  }
}

}  // namespace nvidia::isaac_ros::onnx_inference

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::onnx_inference::OnnxInferenceNode)
