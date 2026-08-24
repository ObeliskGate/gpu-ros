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

#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <exception>
#include <fstream>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "rclcpp_components/register_node_macro.hpp"

namespace gpu_ros::onnx_inference
{

namespace
{

using SteadyClock = std::chrono::steady_clock;

int64_t ToNanoseconds(SteadyClock::time_point timestamp)
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    timestamp.time_since_epoch()).count();
}

int64_t DurationNanoseconds(
  SteadyClock::time_point start, SteadyClock::time_point end)
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

}  // namespace

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
  debug_message_flow_ = declare_parameter<bool>("debug_message_flow", false);
  timing_report_path_ = declare_parameter<std::string>("timing_report_path", "");
  if (!timing_report_path_.empty()) {
    timing_records_.reserve(16384);
  }
  const auto managed_input_contracts =
    declare_parameter<std::vector<std::string>>(
      "managed_input_contracts", std::vector<std::string>{});
  const auto managed_output_contracts =
    declare_parameter<std::vector<std::string>>(
      "managed_output_contracts", std::vector<std::string>{});
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

  io_ = CreateTensorBundleIO(this, transport);
  io_->Subscribe(
    [this](gpu_ros_managed::ManagedTensorBundleView inputs) {
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
  // callback can retain a TensorBundle buffer during pool shutdown.
  io_.reset();
  if (core_ && !core_->shutdown(std::chrono::seconds(5))) {
    RCLCPP_ERROR(
      get_logger(),
      "Managed output pool shutdown did not drain within the configured timeout; "
      "buffers remain orphan-safe and were not force-released.");
  }
  FinalizeOrtProfile("shutdown");
  WriteTimingReport();
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

void OnnxInferenceNode::OnTensors(gpu_ros_managed::ManagedTensorBundleView inputs)
{
  if (!core_) {
    RCLCPP_WARN_ONCE(get_logger(), "Received tensor but inference core is not initialized.");
    return;
  }

  const int64_t message_id = inputs.header().stamp.sec;
  const auto callback_start = SteadyClock::now();
  TrackMessageId(message_id);
  const auto lock_start = SteadyClock::now();
  std::lock_guard<std::mutex> lock(inference_mutex_);
  const auto lock_acquired = SteadyClock::now();
  auto run_finished = lock_acquired;
  auto callback_finished = lock_acquired;
  OnnxInferenceCore::InferenceStageTiming stage_timing;
  uint8_t timing_status = 1;
  try {
    TensorBundleOutput output;
    output.header = inputs.header();
    output.tensors = core_->RunInference(
      std::move(inputs), io_->output_placement(),
      timing_report_path_.empty() ? nullptr : &stage_timing);
    run_finished = SteadyClock::now();
    timing_status = 2;
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
    callback_finished = SteadyClock::now();
    timing_status = 0;
  } catch (const Ort::Exception & e) {
    callback_finished = SteadyClock::now();
    const char * message = e.what();
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "ONNX Runtime dropped an input frame: error_code=%d message=%s",
      static_cast<int>(e.GetOrtErrorCode()),
      (message != nullptr && message[0] != '\0') ? message : "<empty>");
  } catch (const std::exception & e) {
    callback_finished = SteadyClock::now();
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "Inference dropped an input frame: %s", e.what());
  } catch (...) {
    callback_finished = SteadyClock::now();
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "Inference dropped an input frame after an unknown exception");
  }

  if (!timing_report_path_.empty()) {
    const auto inference_end = timing_status == 1 ? callback_finished : run_finished;
    const auto publish_start = run_finished;
    const auto publish_end = timing_status == 1 ? run_finished : callback_finished;
    timing_records_.push_back(TimingRecord{
        message_id,
        ToNanoseconds(callback_start),
        DurationNanoseconds(lock_start, lock_acquired),
        DurationNanoseconds(lock_acquired, inference_end),
        stage_timing.input_setup_ns,
        stage_timing.ort_session_run_ns,
        stage_timing.output_materialize_ns,
        DurationNanoseconds(publish_start, publish_end),
        DurationNanoseconds(callback_start, callback_finished),
        timing_status});
  }
}

void OnnxInferenceNode::WriteTimingReport() noexcept
{
  if (timing_report_path_.empty()) {
    return;
  }

  std::ofstream output(timing_report_path_, std::ios::out | std::ios::trunc);
  if (!output) {
    RCLCPP_ERROR(
      get_logger(), "Failed to open inference timing report '%s'.",
      timing_report_path_.c_str());
    return;
  }

  output <<
    "message_id,callback_start_ns,lock_wait_ns,run_inference_ns,input_setup_ns,"
    "ort_session_run_ns,output_materialize_ns,publish_ns,total_ns,status\n";
  for (const auto & record : timing_records_) {
    const char * status = record.status == 0 ? "ok" :
      (record.status == 1 ? "run_inference_error" : "publish_error");
    output << record.message_id << ',' << record.callback_start_ns << ',' <<
      record.lock_wait_ns << ',' << record.run_inference_ns << ',' <<
      record.input_setup_ns << ',' << record.ort_session_run_ns << ',' <<
      record.output_materialize_ns << ',' << record.publish_ns << ',' <<
      record.total_ns << ',' << status << '\n';
  }
  if (!output) {
    RCLCPP_ERROR(
      get_logger(), "Failed while writing inference timing report '%s'.",
      timing_report_path_.c_str());
    return;
  }
  RCLCPP_INFO(
    get_logger(), "Inference timing report written to '%s' (%zu callbacks).",
    timing_report_path_.c_str(), timing_records_.size());
}

void OnnxInferenceNode::TrackMessageId(int64_t message_id)
{
  if (!debug_message_flow_) {
    return;
  }
  if (has_last_message_id_ && message_id > last_message_id_ + 1) {
    RCLCPP_WARN(
      get_logger(),
      "MESSAGE_FLOW_GAP stage=inference_input previous=%" PRId64 " current=%" PRId64
      " missing=%" PRId64,
      last_message_id_, message_id, message_id - last_message_id_ - 1);
  } else if (has_last_message_id_ && message_id <= last_message_id_) {
    RCLCPP_INFO(
      get_logger(), "MESSAGE_FLOW_RESET stage=inference_input previous=%" PRId64
      " current=%" PRId64,
      last_message_id_, message_id);
  }
  last_message_id_ = message_id;
  has_last_message_id_ = true;
}

}  // namespace gpu_ros::onnx_inference

RCLCPP_COMPONENTS_REGISTER_NODE(gpu_ros::onnx_inference::OnnxInferenceNode)
