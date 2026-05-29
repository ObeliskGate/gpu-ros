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

#include <stdexcept>
#include <string>
#include <utility>

namespace nvidia::isaac_ros::onnx_inference
{

namespace
{

size_t ElementCount(const std::vector<int64_t> & shape)
{
  size_t n = 1;
  for (auto d : shape) {
    n *= static_cast<size_t>(d);
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
      return 1;
  }
}

void AppendExecutionProvider(
  Ort::SessionOptions & opts,
  ExecutionProvider ep,
  int device_id)
{
  switch (ep) {
    case ExecutionProvider::kCuda: {
        OrtCUDAProviderOptions cuda_opts{};
        cuda_opts.device_id = device_id;
        opts.AppendExecutionProvider_CUDA(cuda_opts);
        break;
      }
    case ExecutionProvider::kRocm:
#ifdef ORT_ROCM_AVAILABLE
      {
        OrtROCMProviderOptions rocm_opts{};
        rocm_opts.device_id = device_id;
        opts.AppendExecutionProvider_ROCM(rocm_opts);
        break;
      }
#else
      throw std::runtime_error(
              "ROCm EP requested but not built. Recompile with ORT_ROCM_AVAILABLE.");
#endif
    case ExecutionProvider::kMigraphx:
#ifdef ORT_MIGRAPHX_AVAILABLE
      {
        OrtMIGraphXProviderOptions migx_opts{};
        opts.AppendExecutionProvider_MIGraphX(migx_opts);
        break;
      }
#else
      throw std::runtime_error(
              "MIGraphX EP requested but not built. Recompile with ORT_MIGRAPHX_AVAILABLE.");
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
: env_(ORT_LOGGING_LEVEL_WARNING, "isaac_ros_onnx_inference")
{
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  AppendExecutionProvider(session_options_, cfg.ep, cfg.gpu_device_id);

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

std::vector<HostTensor> OnnxInferenceCore::RunInference(
  const std::vector<HostTensor> & inputs)
{
  Ort::MemoryInfo mem_info =
    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<Ort::Value> ort_inputs;
  std::vector<const char *> input_name_ptrs;
  ort_inputs.reserve(inputs.size());
  input_name_ptrs.reserve(inputs.size());

  for (const auto & t : inputs) {
    input_name_ptrs.push_back(t.name.c_str());
    const size_t elem_count = ElementCount(t.shape);
    auto * mutable_data = const_cast<uint8_t *>(t.data.data());

    if (t.dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      ort_inputs.push_back(
        Ort::Value::CreateTensor<float>(
          mem_info, reinterpret_cast<float *>(mutable_data),
          elem_count, t.shape.data(), t.shape.size()));
    } else if (t.dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      ort_inputs.push_back(
        Ort::Value::CreateTensor<int64_t>(
          mem_info, reinterpret_cast<int64_t *>(mutable_data),
          elem_count, t.shape.data(), t.shape.size()));
    } else {
      throw std::runtime_error(
              "OnnxInferenceCore: unsupported input dtype " +
              std::to_string(static_cast<int>(t.dtype)));
    }
  }

  std::vector<const char *> output_name_ptrs;
  output_name_ptrs.reserve(output_names_.size());
  for (const auto & n : output_names_) {
    output_name_ptrs.push_back(n.c_str());
  }

  auto ort_outputs = session_->Run(
    Ort::RunOptions{nullptr},
    input_name_ptrs.data(), ort_inputs.data(), ort_inputs.size(),
    output_name_ptrs.data(), output_name_ptrs.size());

  std::vector<HostTensor> results;
  results.reserve(ort_outputs.size());
  for (size_t i = 0; i < ort_outputs.size(); ++i) {
    auto type_info = ort_outputs[i].GetTensorTypeAndShapeInfo();
    HostTensor ht;
    ht.name = output_names_[i];
    ht.dtype = type_info.GetElementType();
    ht.shape = type_info.GetShape();
    const size_t byte_count = type_info.GetElementCount() * DtypeSize(ht.dtype);
    const auto * raw = reinterpret_cast<const uint8_t *>(ort_outputs[i].GetTensorRawData());
    ht.data.assign(raw, raw + byte_count);
    results.push_back(std::move(ht));
  }
  return results;
}

}  // namespace nvidia::isaac_ros::onnx_inference
