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

#ifndef ISAAC_ROS_ONNX_INFERENCE__ONNX_INFERENCE_CORE_HPP_
#define ISAAC_ROS_ONNX_INFERENCE__ONNX_INFERENCE_CORE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace nvidia::isaac_ros::onnx_inference
{

enum class TensorMemoryKind { kHost, kCudaDevice };

/// Non-owning tensor view. The backing storage must remain valid until RunInference returns.
struct TensorView
{
  std::string name;
  ONNXTensorElementDataType dtype{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
  std::vector<int64_t> shape;
  const void * data{nullptr};
  size_t byte_size{0};
  TensorMemoryKind memory_kind{TensorMemoryKind::kHost};
  int device_id{-1};
};

/// ORT-owned CUDA output kept alive until the transport releases it.
struct DeviceTensorBuffer
{
  void * data{nullptr};
  size_t byte_size{0};
  int device_id{-1};
  std::shared_ptr<Ort::Value> owner;
};

/// Tensor result owned either by host storage or by an ORT CUDA allocation.
struct OwnedTensor
{
  std::string name;
  ONNXTensorElementDataType dtype{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
  std::vector<int64_t> shape;
  std::variant<std::vector<uint8_t>, DeviceTensorBuffer> storage;
};

/// Supported execution providers.
enum class ExecutionProvider { kCuda, kRocm, kMigraphx, kCpu };

ExecutionProvider ParseExecutionProvider(const std::string & ep_str);

/// Core inference wrapper around Ort::Session.
/// Thread-compatible (not thread-safe): create one per node.
class OnnxInferenceCore
{
public:
  struct Config
  {
    std::string model_file_path;
    ExecutionProvider ep{ExecutionProvider::kCuda};
    int gpu_device_id{0};
    std::string ort_profile_prefix;
  };

  explicit OnnxInferenceCore(const Config & cfg);
  ~OnnxInferenceCore() = default;

  // Non-copyable
  OnnxInferenceCore(const OnnxInferenceCore &) = delete;
  OnnxInferenceCore & operator=(const OnnxInferenceCore &) = delete;

  std::vector<OwnedTensor> RunInference(
    const std::vector<TensorView> & inputs,
    TensorMemoryKind output_memory_kind = TensorMemoryKind::kHost);

  size_t GetInputCount() const;
  size_t GetOutputCount() const;
  bool IsProfilingEnabled() const;
  std::string EndProfiling();

private:
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  Ort::AllocatorWithDefaultOptions allocator_;
  bool profiling_enabled_{false};
  ExecutionProvider execution_provider_;
  int gpu_device_id_;

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
};

}  // namespace nvidia::isaac_ros::onnx_inference

#endif  // ISAAC_ROS_ONNX_INFERENCE__ONNX_INFERENCE_CORE_HPP_
