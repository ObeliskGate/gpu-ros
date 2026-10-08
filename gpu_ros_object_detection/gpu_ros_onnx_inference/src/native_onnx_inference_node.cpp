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

#include "gpu_ros_onnx_inference/native_onnx_inference_node.hpp"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "model_path.hpp"
#include "native_onnx_executor.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#ifdef GPU_ROS_NATIVE_EXECUTOR_TEST
#include "native_onnx_executor_test_peer.hpp"
#endif

namespace gpu_ros::onnx_inference
{

NativeOnnxInferenceNode::NativeOnnxInferenceNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("onnx_inference_node", options)
{
  std::string model_file_path = declare_parameter<std::string>("model_file_path", "");
  const auto provider = declare_parameter<std::string>("execution_provider", "cuda");
  const auto model_profile = declare_parameter<std::string>("model_profile", "");
  const char * environment = std::getenv("OVG_ASSETS_ROOT");
  const auto assets_root = declare_parameter<std::string>("model_assets_root",
    environment == nullptr ? "/workspaces/ovg-assets" : environment);
  const int device_id = declare_parameter<int>("gpu_device_id", 0);
  const auto profile_prefix = declare_parameter<std::string>("ort_profile_prefix", "");
  const auto profile_frames = declare_parameter<int64_t>("ort_profile_frames", 0);
  const auto report_path = declare_parameter<std::string>("binding_report_path", "");
  const auto transport = declare_parameter<std::string>("transport", "tensor_list");
  const auto contracts = declare_parameter<std::vector<std::string>>(
    "output_contracts", std::vector<std::string>{});

  if (provider != "cuda" || transport != "tensor_list") {
    throw std::invalid_argument(
      "NativeOnnxInferenceNode requires execution_provider=cuda and transport=tensor_list");
  }
  if (device_id < 0) {
    throw std::invalid_argument("gpu_device_id must be non-negative");
  }
  if (profile_frames < 0 ||
    static_cast<uint64_t>(profile_frames) > std::numeric_limits<size_t>::max())
  {
    throw std::invalid_argument("ort_profile_frames must be non-negative and fit size_t");
  }
  if (profile_frames > 0 && profile_prefix.empty()) {
    throw std::invalid_argument("ort_profile_frames requires a non-empty ort_profile_prefix");
  }
  ort_profile_frames_ = static_cast<size_t>(profile_frames);
  if (!model_profile.empty()) {
    if (!model_file_path.empty()) {
      RCLCPP_WARN(get_logger(), "model_file_path explicitly overrides model_profile=%s",
        model_profile.c_str());
    } else {
      model_file_path = ResolveModelProfile(model_profile, provider, assets_root);
    }
  }

  // Finish all throwing initialization before exposing a subscriber callback.
  if (!model_file_path.empty()) {
    NativeOnnxExecutor::Config config;
    config.session = {model_file_path, ExecutionProvider::kCuda, device_id, profile_prefix};
    config.output_contracts = contracts;
    config.binding_report_path = report_path;
    executor_ = std::make_unique<NativeOnnxExecutor>(config);
    RCLCPP_INFO(get_logger(), "Loaded native CUDA model '%s', transport=tensor_list",
      model_file_path.c_str());
  } else {
    RCLCPP_WARN(get_logger(), "model_file_path is empty; native inference is not initialized.");
  }
  transport_ = std::make_unique<Transport>(this, device_id);
  callback_state_ = std::make_shared<CallbackState>();
  callback_state_->node = this;
  transport_->Subscribe(MakeCallback());
}

NativeOnnxInferenceNode::Transport::Callback NativeOnnxInferenceNode::MakeCallback() const
{
  return [state = callback_state_](TensorList input) {
    NativeOnnxInferenceNode * node = nullptr;
    {
      std::lock_guard<std::mutex> lock(state->mutex);
      if (state->shutting_down || state->node == nullptr) {
        return;
      }
      node = state->node;
      ++state->active_callbacks;
    }
    try {
#ifdef GPU_ROS_NATIVE_EXECUTOR_TEST
      NativeOnnxExecutorTestPeer::CallbackEntered();
#endif
      node->OnTensors(std::move(input));
    } catch (...) {
      // Admission must always be released, even if a future callback adds a throw.
    }
    {
      std::lock_guard<std::mutex> lock(state->mutex);
      --state->active_callbacks;
    }
    state->cv.notify_all();
  };
}

NativeOnnxInferenceNode::~NativeOnnxInferenceNode()
{
  if (callback_state_) {
    {
      std::lock_guard<std::mutex> lock(callback_state_->mutex);
      callback_state_->shutting_down = true;
      callback_state_->node = nullptr;
    }
    callback_state_->cv.notify_all();
    std::unique_lock<std::mutex> lock(callback_state_->mutex);
    callback_state_->cv.wait(lock, [this] { return callback_state_->active_callbacks == 0; });
  }
  // Active callbacks can publish until the drain completes. Queued callbacks
  // retain only CallbackState and now reject admission without touching this.
  if (transport_) {
    transport_->Unsubscribe();
  }
  transport_.reset();
  FinalizeOrtProfile("shutdown");
  executor_.reset();
  callback_state_.reset();
}

void NativeOnnxInferenceNode::FinalizeOrtProfile(const char * reason) noexcept
{
  if (!executor_ || !executor_->IsProfilingEnabled()) {
    return;
  }
  try {
    const auto path = executor_->EndProfiling();
    RCLCPP_INFO(get_logger(), "ONNX Runtime profile written to '%s' after %zu frames (%s).",
      path.c_str(), executor_->successful_runs(), reason);
  } catch (const std::exception & error) {
    RCLCPP_ERROR(get_logger(), "Failed to finalize ONNX Runtime profile: %s", error.what());
  }
}

void NativeOnnxInferenceNode::OnTensors(TensorList inputs)
{
  std::lock_guard<std::mutex> lock(inference_mutex_);
  if (!executor_) {
    RCLCPP_WARN_ONCE(get_logger(), "Received tensor but native inference is not initialized.");
    return;
  }
  try {
    auto output = executor_->Run(std::move(inputs), *transport_);
    if (ort_profile_frames_ > 0 && executor_->successful_runs() >= ort_profile_frames_) {
      FinalizeOrtProfile("configured frame limit");
    }
    transport_->Publish(output);
  } catch (const Ort::Exception & error) {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000,
      "Native ONNX Runtime dropped an input frame: error_code=%d message=%s",
      static_cast<int>(error.GetOrtErrorCode()), error.what());
  } catch (const std::exception & error) {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000,
      "Native inference dropped an input frame: %s", error.what());
  } catch (...) {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000,
      "Native inference dropped an input frame after an unknown exception");
  }
  // A successful Run still counts if reporting or publishing failed afterward.
  if (ort_profile_frames_ > 0 && executor_->successful_runs() >= ort_profile_frames_) {
    FinalizeOrtProfile("configured frame limit");
  }
}

} // namespace gpu_ros::onnx_inference

RCLCPP_COMPONENTS_REGISTER_NODE(gpu_ros::onnx_inference::NativeOnnxInferenceNode)
