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

#include "gpu_ros_onnx_inference/onnx_session.hpp"
#include <cstdlib>
#include <stdexcept>
#include <unordered_map>

namespace gpu_ros::onnx_inference
{
namespace
{
OrtLoggingLevel GetOrtLoggingLevel()
{
  const char * value = std::getenv("GPU_ROS_ORT_LOG_LEVEL");
  if (value == nullptr || value[0] == '\0') {
    return ORT_LOGGING_LEVEL_WARNING;
  }

  const std::string level{value};
  if (level == "verbose") {
    return ORT_LOGGING_LEVEL_VERBOSE;
  }
  if (level == "info") {
    return ORT_LOGGING_LEVEL_INFO;
  }
  if (level == "warning") {
    return ORT_LOGGING_LEVEL_WARNING;
  }
  if (level == "error") {
    return ORT_LOGGING_LEVEL_ERROR;
  }
  if (level == "fatal") {
    return ORT_LOGGING_LEVEL_FATAL;
  }

  throw std::invalid_argument(
    "GPU_ROS_ORT_LOG_LEVEL must be verbose, info, warning, error, or fatal");
}

void AppendExecutionProvider(
  Ort::SessionOptions & opts, ExecutionProvider ep, [[maybe_unused]] int device_id)
{
  switch (ep) {
    case ExecutionProvider::kCuda:
#ifdef ORT_CUDA_AVAILABLE
    {
      OrtCUDAProviderOptions provider_options{};
      provider_options.device_id = device_id;
      opts.AppendExecutionProvider_CUDA(provider_options);
      break;
    }
#else
      throw std::runtime_error(
        "CUDA EP requested but not built. Recompile with -DORT_ENABLE_CUDA=ON.");
#endif
    case ExecutionProvider::kRocm:
#ifdef ORT_ROCM_AVAILABLE
    {
      const std::unordered_map<std::string, std::string> provider_options{
        {"device_id", std::to_string(device_id)}};
      opts.AppendExecutionProvider("ROCMExecutionProvider", provider_options);
      break;
    }
#else
      throw std::runtime_error(
        "ROCm EP requested but not built. Recompile with -DORT_ENABLE_ROCM=ON.");
#endif
    case ExecutionProvider::kMigraphx:
#ifdef ORT_MIGRAPHX_AVAILABLE
    {
      const std::unordered_map<std::string, std::string> provider_options{
        {"device_id", std::to_string(device_id)}};
      opts.AppendExecutionProvider("MIGraphXExecutionProvider", provider_options);
      break;
    }
#else
      throw std::runtime_error(
        "MIGraphX EP requested but not built. Recompile with -DORT_ENABLE_MIGRAPHX=ON.");
#endif
    case ExecutionProvider::kCpu:
      // CPU is the default fallback; no explicit provider needed.
      break;
  }
}
} // namespace

ExecutionProvider ParseExecutionProvider(const std::string & ep_str)
{
  if (ep_str == "cuda") {
    return ExecutionProvider::kCuda;
  }
  if (ep_str == "rocm") {
    return ExecutionProvider::kRocm;
  }
  if (ep_str == "migraphx") {
    return ExecutionProvider::kMigraphx;
  }
  if (ep_str == "cpu") {
    return ExecutionProvider::kCpu;
  }
  throw std::invalid_argument("Unknown execution_provider: " + ep_str);
}

OnnxSession::OnnxSession(const Config & cfg)
: env_(GetOrtLoggingLevel(), "gpu_ros_onnx_inference")
{
  if (cfg.gpu_device_id < 0) {
    throw std::invalid_argument("gpu_device_id must be non-negative");
  }
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (!cfg.ort_profile_prefix.empty()) {
    session_options_.EnableProfiling(cfg.ort_profile_prefix.c_str());
    profiling_enabled_ = true;
  }
  try {
    AppendExecutionProvider(session_options_, cfg.ep, cfg.gpu_device_id);
  } catch (const Ort::Exception & e) {
    throw std::runtime_error(
      "Failed to append requested ONNX Runtime execution provider: " + std::string(e.what()));
  }

  session_ = std::make_unique<Ort::Session>(env_, cfg.model_file_path.c_str(), session_options_);

  for (size_t i = 0; i < session_->GetInputCount(); ++i) {
    auto name = session_->GetInputNameAllocated(i, allocator_);
    input_names_.emplace_back(name.get());
  }
  for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
    auto name = session_->GetOutputNameAllocated(i, allocator_);
    output_names_.emplace_back(name.get());

  }
}

std::string OnnxSession::EndProfiling()
{
  if (!profiling_enabled_) {
    return {};
  }

  auto profile_path = session_->EndProfilingAllocated(allocator_);
  profiling_enabled_ = false;
  return profile_path ? profile_path.get() : std::string{};
}

} // namespace gpu_ros::onnx_inference
