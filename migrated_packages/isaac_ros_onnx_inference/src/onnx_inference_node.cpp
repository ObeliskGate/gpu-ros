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

#include <cstdint>
#include <exception>
#include <memory>
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
  const std::string transport =
    declare_parameter<std::string>("transport", "std");
  const ExecutionProvider execution_provider = ParseExecutionProvider(ep_str);

  if (ort_profile_frames < 0) {
    throw std::invalid_argument("ort_profile_frames must be non-negative");
  }
  if (ort_profile_frames > 0 && ort_profile_prefix.empty()) {
    throw std::invalid_argument(
            "ort_profile_frames requires a non-empty ort_profile_prefix");
  }
  ort_profile_frames_ = static_cast<size_t>(ort_profile_frames);

  if (transport == "nitros" && execution_provider != ExecutionProvider::kCuda) {
    throw std::invalid_argument("transport=nitros requires execution_provider=cuda");
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
  core_ = std::make_unique<OnnxInferenceCore>(cfg);

  RCLCPP_INFO(
    get_logger(),
    "Loaded model '%s' with %zu inputs, %zu outputs, EP=%s, transport=%s, "
    "CPU fallback=allowed",
    model_file_path.c_str(),
    core_->GetInputCount(),
    core_->GetOutputCount(),
    ep_str.c_str(),
    transport.c_str());

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
}

}  // namespace nvidia::isaac_ros::onnx_inference

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::onnx_inference::OnnxInferenceNode)
