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

#include "isaac_ros_onnx_inference/onnx_inference_core.hpp"

#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nvidia::isaac_ros::onnx_inference
{

namespace
{

OrtLoggingLevel GetOrtLoggingLevel()
{
  const char * value = std::getenv("ISAAC_ROS_ORT_LOG_LEVEL");
  if (value == nullptr || value[0] == '\0') {
    return ORT_LOGGING_LEVEL_WARNING;
  }

  const std::string level{value};
  if (level == "verbose") {return ORT_LOGGING_LEVEL_VERBOSE;}
  if (level == "info") {return ORT_LOGGING_LEVEL_INFO;}
  if (level == "warning") {return ORT_LOGGING_LEVEL_WARNING;}
  if (level == "error") {return ORT_LOGGING_LEVEL_ERROR;}
  if (level == "fatal") {return ORT_LOGGING_LEVEL_FATAL;}

  throw std::invalid_argument(
          "ISAAC_ROS_ORT_LOG_LEVEL must be verbose, info, warning, error, or fatal");
}

size_t ElementCount(const std::vector<int64_t> & shape)
{
  size_t n = 1;
  for (auto d : shape) {
    if (d < 0) {
      throw std::invalid_argument("Tensor shape contains a negative dimension");
    }
    const size_t dimension = static_cast<size_t>(d);
    if (dimension != 0 && n > std::numeric_limits<size_t>::max() / dimension) {
      throw std::overflow_error("Tensor element count overflows size_t");
    }
    n *= dimension;
  }
  return n;
}

size_t DtypeSize(ONNXTensorElementDataType dtype)
{
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return sizeof(float);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return sizeof(int64_t);
    default:
      throw std::invalid_argument(
              "Unsupported tensor dtype " + std::to_string(static_cast<int>(dtype)));
  }
}

size_t RequiredBytes(const TensorView & tensor)
{
  const size_t element_count = ElementCount(tensor.shape);
  const size_t element_size = DtypeSize(tensor.dtype);
  if (element_count > std::numeric_limits<size_t>::max() / element_size) {
    throw std::overflow_error("Tensor byte size overflows size_t: " + tensor.name);
  }
  return element_count * element_size;
}

void ValidateTensorView(
  const TensorView & tensor,
  ExecutionProvider execution_provider,
  int gpu_device_id)
{
  const size_t required_bytes = RequiredBytes(tensor);
  if (tensor.byte_size < required_bytes) {
    throw std::invalid_argument(
            "Tensor '" + tensor.name + "' has " + std::to_string(tensor.byte_size) +
            " bytes, but its shape and dtype require " + std::to_string(required_bytes));
  }
  if (required_bytes != 0 && tensor.data == nullptr) {
    throw std::invalid_argument("Tensor '" + tensor.name + "' has a null data pointer");
  }
  if (tensor.memory_kind == TensorMemoryKind::kCudaDevice) {
    if (execution_provider != ExecutionProvider::kCuda) {
      throw std::invalid_argument("CUDA device input requires the CUDA execution provider");
    }
    if (tensor.device_id != gpu_device_id) {
      throw std::invalid_argument(
              "Tensor '" + tensor.name + "' is on CUDA device " +
              std::to_string(tensor.device_id) + ", but the session uses device " +
              std::to_string(gpu_device_id));
    }
  }
}

void AppendExecutionProvider(
  Ort::SessionOptions & opts,
  ExecutionProvider ep,
  [[maybe_unused]] int device_id)
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
        const std::unordered_map<std::string, std::string> provider_options{};
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

}  // namespace

ExecutionProvider ParseExecutionProvider(const std::string & ep_str)
{
  if (ep_str == "cuda") {return ExecutionProvider::kCuda;}
  if (ep_str == "rocm") {return ExecutionProvider::kRocm;}
  if (ep_str == "migraphx") {return ExecutionProvider::kMigraphx;}
  if (ep_str == "cpu") {return ExecutionProvider::kCpu;}
  throw std::invalid_argument("Unknown execution_provider: " + ep_str);
}

OnnxInferenceCore::OnnxInferenceCore(const Config & cfg)
: env_(GetOrtLoggingLevel(), "isaac_ros_onnx_inference"),
  execution_provider_(cfg.ep),
  gpu_device_id_(cfg.gpu_device_id)
{
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (!cfg.ort_profile_prefix.empty()) {
    session_options_.EnableProfiling(cfg.ort_profile_prefix.c_str());
    profiling_enabled_ = true;
  }
  try {
    AppendExecutionProvider(session_options_, cfg.ep, cfg.gpu_device_id);
  } catch (const Ort::Exception & e) {
    throw std::runtime_error(
            "Failed to append requested ONNX Runtime execution provider: " +
            std::string(e.what()));
  }

  session_ = std::make_unique<Ort::Session>(
    env_, cfg.model_file_path.c_str(), session_options_);

  for (size_t i = 0; i < session_->GetInputCount(); ++i) {
    auto name = session_->GetInputNameAllocated(i, allocator_);
    input_names_.emplace_back(name.get());
  }
  for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
    auto name = session_->GetOutputNameAllocated(i, allocator_);
    output_names_.emplace_back(name.get());
  }
}

size_t OnnxInferenceCore::GetInputCount() const {return session_->GetInputCount();}
size_t OnnxInferenceCore::GetOutputCount() const {return session_->GetOutputCount();}
bool OnnxInferenceCore::IsProfilingEnabled() const {return profiling_enabled_;}

std::string OnnxInferenceCore::EndProfiling()
{
  if (!profiling_enabled_) {
    return {};
  }

  auto profile_path = session_->EndProfilingAllocated(allocator_);
  profiling_enabled_ = false;
  return profile_path ? profile_path.get() : std::string{};
}

std::vector<OwnedTensor> OnnxInferenceCore::RunInference(
  const std::vector<TensorView> & inputs,
  TensorMemoryKind output_memory_kind)
{
  if (output_memory_kind == TensorMemoryKind::kCudaDevice &&
    execution_provider_ != ExecutionProvider::kCuda)
  {
    throw std::invalid_argument("CUDA device output requires the CUDA execution provider");
  }

  std::vector<Ort::Value> ort_inputs;
  std::vector<const char *> input_name_ptrs;
  ort_inputs.reserve(inputs.size());
  input_name_ptrs.reserve(inputs.size());

  for (const auto & t : inputs) {
    ValidateTensorView(t, execution_provider_, gpu_device_id_);
    input_name_ptrs.push_back(t.name.c_str());
    auto memory_info = t.memory_kind == TensorMemoryKind::kCudaDevice ?
      Ort::MemoryInfo("Cuda", OrtArenaAllocator, t.device_id, OrtMemTypeDefault) :
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    ort_inputs.push_back(
      Ort::Value::CreateTensor(
        memory_info, const_cast<void *>(t.data), RequiredBytes(t),
        t.shape.data(), t.shape.size(), t.dtype));
  }

  std::vector<const char *> output_name_ptrs;
  output_name_ptrs.reserve(output_names_.size());
  for (const auto & n : output_names_) {
    output_name_ptrs.push_back(n.c_str());
  }

  std::vector<Ort::Value> ort_outputs;
  if (output_memory_kind == TensorMemoryKind::kCudaDevice) {
    Ort::MemoryInfo cuda_memory_info(
      "Cuda", OrtArenaAllocator, gpu_device_id_, OrtMemTypeDefault);
    Ort::IoBinding binding(*session_);
    for (size_t i = 0; i < ort_inputs.size(); ++i) {
      binding.BindInput(input_name_ptrs[i], ort_inputs[i]);
    }
    for (const auto * output_name : output_name_ptrs) {
      binding.BindOutput(output_name, cuda_memory_info);
    }

    binding.SynchronizeInputs();
    session_->Run(Ort::RunOptions{nullptr}, binding);
    binding.SynchronizeOutputs();

    const auto bound_output_names = binding.GetOutputNames();
    if (bound_output_names != output_names_) {
      throw std::runtime_error("ONNX Runtime returned unexpected bound output names");
    }
    ort_outputs = binding.GetOutputValues();
  } else {
    ort_outputs = session_->Run(
      Ort::RunOptions{nullptr},
      input_name_ptrs.data(), ort_inputs.data(), ort_inputs.size(),
      output_name_ptrs.data(), output_name_ptrs.size());
  }

  if (ort_outputs.size() != output_names_.size()) {
    throw std::runtime_error("ONNX Runtime returned an unexpected number of outputs");
  }

  std::vector<OwnedTensor> results;
  results.reserve(ort_outputs.size());
  for (size_t i = 0; i < ort_outputs.size(); ++i) {
    auto type_info = ort_outputs[i].GetTensorTypeAndShapeInfo();
    OwnedTensor tensor;
    tensor.name = output_names_[i];
    tensor.dtype = type_info.GetElementType();
    tensor.shape = type_info.GetShape();
    const size_t element_count = type_info.GetElementCount();
    const size_t element_size = DtypeSize(tensor.dtype);
    if (element_count > std::numeric_limits<size_t>::max() / element_size) {
      throw std::overflow_error("Output tensor byte size overflows size_t: " + tensor.name);
    }
    const size_t byte_count = element_count * element_size;

    if (output_memory_kind == TensorMemoryKind::kCudaDevice) {
      const auto memory_info = ort_outputs[i].GetTensorMemoryInfo();
      if (memory_info.GetAllocatorName() != "Cuda" ||
        memory_info.GetDeviceType() != OrtMemoryInfoDeviceType_GPU ||
        memory_info.GetDeviceId() != gpu_device_id_)
      {
        throw std::runtime_error(
                "Output tensor '" + tensor.name +
                "' was not allocated on the requested CUDA device");
      }
      auto owner = std::make_shared<Ort::Value>(std::move(ort_outputs[i]));
      tensor.storage = DeviceTensorBuffer{
        owner->GetTensorMutableRawData(), byte_count, gpu_device_id_, std::move(owner)};
    } else {
      const auto memory_info = ort_outputs[i].GetTensorMemoryInfo();
      if (memory_info.GetDeviceType() != OrtMemoryInfoDeviceType_CPU) {
        throw std::runtime_error(
                "Output tensor '" + tensor.name + "' is not in host-accessible memory");
      }
      const auto * raw =
        reinterpret_cast<const uint8_t *>(ort_outputs[i].GetTensorRawData());
      std::vector<uint8_t> host_data;
      if (byte_count != 0) {
        host_data.assign(raw, raw + byte_count);
      }
      tensor.storage = std::move(host_data);
    }
    results.push_back(std::move(tensor));
  }
  return results;
}

}  // namespace nvidia::isaac_ros::onnx_inference
