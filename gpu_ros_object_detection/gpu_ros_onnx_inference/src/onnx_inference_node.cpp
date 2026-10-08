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

#include "gpu_ros_onnx_inference/onnx_inference_node.hpp"
#include "model_path.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rclcpp_components/register_node_macro.hpp"
#ifdef GPU_ROS_ORT_BINDING_TEST
#include "onnx_binding_test_peer.hpp"
#endif

namespace gpu_ros::onnx_inference
{

OnnxInferenceNode::OnnxInferenceNode(const rclcpp::NodeOptions & options)
    : rclcpp::Node("onnx_inference_node", options)
{
  std::string model_file_path = declare_parameter<std::string>("model_file_path", "");
  const std::string ep_str = declare_parameter<std::string>("execution_provider", "cuda");
  const std::string model_profile = declare_parameter<std::string>("model_profile", "");
  const char * assets_root_environment = std::getenv("OVG_ASSETS_ROOT");
  const std::string model_assets_root = declare_parameter<std::string>("model_assets_root",
    assets_root_environment == nullptr ? "/workspaces/ovg-assets" : assets_root_environment);
  const int gpu_device_id = declare_parameter<int>("gpu_device_id", 0);
  const std::string ort_profile_prefix = declare_parameter<std::string>("ort_profile_prefix", "");
  const int64_t ort_profile_frames = declare_parameter<int64_t>("ort_profile_frames", 0);
  const std::string binding_report_path = declare_parameter<std::string>("binding_report_path", "");
  const std::string transport = declare_parameter<std::string>("transport", "std");
  declare_parameter<std::string>("message_format", "tensor_bundle");
  const std::string io_contract = declare_parameter<std::string>("io_contract", "");
  const auto input_contracts = declare_parameter<std::vector<std::string>>(
    "input_contracts", std::vector<std::string>{});
  const auto output_contracts = declare_parameter<std::vector<std::string>>(
    "output_contracts", std::vector<std::string>{});
  const int64_t output_pool_capacity = declare_parameter<int64_t>("output_pool_capacity", 16);
  const int64_t output_pool_wait_timeout_ms =
    declare_parameter<int64_t>("output_pool_wait_timeout_ms", 100);
  const ExecutionProvider execution_provider = ParseExecutionProvider(ep_str);
  io_ = CreateTensorBundleIO(this, transport);

  if (!model_profile.empty()) {
    if (!model_file_path.empty()) {
      RCLCPP_WARN(get_logger(), "model_file_path explicitly overrides model_profile=%s",
        model_profile.c_str());
    } else {
      model_file_path = ResolveModelProfile(model_profile, ep_str, model_assets_root);
    }
  }

  if (ort_profile_frames < 0) {
    throw std::invalid_argument("ort_profile_frames must be non-negative");
  }
  if (ort_profile_frames > 0 && ort_profile_prefix.empty()) {
    throw std::invalid_argument("ort_profile_frames requires a non-empty ort_profile_prefix");
  }
  if (gpu_device_id < 0 || output_pool_capacity <= 0 ||
      static_cast<uint64_t>(output_pool_capacity) > std::numeric_limits<size_t>::max() ||
      output_pool_wait_timeout_ms < 0)
  {
    throw std::invalid_argument(
      "gpu_device_id must be non-negative, output_pool_capacity must be positive, "
      "and output_pool_wait_timeout_ms must be non-negative");
  }
  ort_profile_frames_ = static_cast<size_t>(ort_profile_frames);

  if (!io_contract.empty() && io_contract != "device_strict") {
    throw std::invalid_argument("io_contract must be empty or device_strict");
  }
  if (io_contract == "device_strict" && model_file_path.empty()) {
    throw std::invalid_argument("device_strict requires a non-empty model_file_path");
  }

  callback_state_ = std::make_shared<CallbackState>();
  callback_state_->node = this;

  if (model_file_path.empty()) {
    RCLCPP_WARN(get_logger(), "model_file_path is empty — inference core not initialized.");
    io_->Subscribe(MakeCallback());
    return;
  }

  OnnxInferenceCore::Config cfg;
  cfg.model_file_path = model_file_path;
  cfg.ep = execution_provider;
  cfg.gpu_device_id = gpu_device_id;
  cfg.ort_profile_prefix = ort_profile_prefix;
  cfg.binding_report_path = binding_report_path;
  cfg.transport = transport;
  cfg.io_contract = io_contract;
  cfg.input_contracts = input_contracts;
  cfg.output_contracts = output_contracts;
  cfg.output_pool_capacity = static_cast<size_t>(output_pool_capacity);
  cfg.output_pool_wait_timeout = std::chrono::milliseconds(output_pool_wait_timeout_ms);
  core_ = std::make_unique<OnnxInferenceCore>(cfg);

  RCLCPP_INFO(get_logger(),
    "Loaded model '%s' with %zu inputs, %zu outputs, EP=%s, transport=%s, "
    "io_contract=%s",
    model_file_path.c_str(), core_->GetInputCount(), core_->GetOutputCount(), ep_str.c_str(),
    transport.c_str(), io_contract.empty() ? "compat" : io_contract.c_str());

  const std::string output_probe = core_->OutputBindingProbeReport();
  if (!output_probe.empty()) {
    RCLCPP_INFO(get_logger(), "Output binding plan: %s", output_probe.c_str());
  }

  if (core_->IsProfilingEnabled()) {
    RCLCPP_INFO(get_logger(),
      "ONNX Runtime profiling enabled with prefix '%s'. The profile is finalized on shutdown.",
      ort_profile_prefix.c_str());
  }
  io_->Subscribe(MakeCallback());
}

ITensorBundleIO::Callback OnnxInferenceNode::MakeCallback() const
{
  return [state = callback_state_](TensorBindingBatch inputs) {
    OnnxInferenceNode * node = nullptr;
    {
      std::lock_guard<std::mutex> lock(state->mutex);
      if (state->shutting_down || state->node == nullptr) { return; }
      node = state->node;
      ++state->active_callbacks;
    }
    try {
#ifdef GPU_ROS_ORT_BINDING_TEST
      OnnxBindingTestPeer::CallbackEntered();
#endif
      node->OnTensors(std::move(inputs));
    } catch (...) {}
    {
      std::lock_guard<std::mutex> lock(state->mutex);
      --state->active_callbacks;
    }
    state->cv.notify_all();
  };
}

OnnxInferenceNode::~OnnxInferenceNode()
{
  // Stop new callbacks first. The callback state is captured by value, so a
  // queued executor callback remains safe after io_ is reset and observes the
  // shutdown flag instead of dereferencing a destroyed node.
  if (callback_state_) {
    std::lock_guard<std::mutex> lock(callback_state_->mutex);
    callback_state_->shutting_down = true;
    callback_state_->node = nullptr;
  }
  if (callback_state_) { callback_state_->cv.notify_all(); }

  // Drain any callback that was already active before destroying the IO.  An
  // active callback may still be inside OnTensors and can publish through
  // io_; resetting io_ first would turn a normal SIGINT into a use-after-free.
  if (callback_state_) {
    std::unique_lock<std::mutex> lock(callback_state_->mutex);
    callback_state_->cv.wait(lock, [this] { return callback_state_->active_callbacks == 0; });
  }
  if (io_ && !io_->shutdown(std::chrono::seconds(5))) {
    RCLCPP_ERROR(get_logger(),
      "Managed output pool shutdown did not drain within the configured timeout; "
      "buffers remain orphan-safe and were not force-released.");
  }
  io_.reset();
  FinalizeOrtProfile("shutdown");
  callback_state_.reset();
}

void OnnxInferenceNode::FinalizeOrtProfile(const char * reason) noexcept
{
  if (!core_ || !core_->IsProfilingEnabled()) {
    return;
  }

  try {
    const std::string profile_path = core_->EndProfiling();
    RCLCPP_INFO(get_logger(), "ONNX Runtime profile written to '%s' after %zu frames (%s).",
      profile_path.c_str(), inference_count_, reason);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Failed to finalize ONNX Runtime profile: %s", e.what());
  }
}

void OnnxInferenceNode::OnTensors(TensorBindingBatch inputs)
{
  if (!core_) {
    RCLCPP_WARN_ONCE(get_logger(), "Received tensor but inference core is not initialized.");
    return;
  }

  std::lock_guard<std::mutex> lock(inference_mutex_);
  try {
    TensorBundleOutput output;
    output.header = inputs.header;
    output.tensors = core_->RunInference(
      std::move(inputs), io_->output_placement(), nullptr, io_->device_output_allocator());
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
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000,
      "ONNX Runtime dropped an input frame: error_code=%d message=%s",
      static_cast<int>(e.GetOrtErrorCode()),
      (message != nullptr && message[0] != '\0') ? message : "<empty>");
  } catch (const std::exception & e) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 1000, "Inference dropped an input frame: %s", e.what());
  } catch (...) {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000,
      "Inference dropped an input frame after an unknown exception");
  }
}

} // namespace gpu_ros::onnx_inference

RCLCPP_COMPONENTS_REGISTER_NODE(gpu_ros::onnx_inference::OnnxInferenceNode)
