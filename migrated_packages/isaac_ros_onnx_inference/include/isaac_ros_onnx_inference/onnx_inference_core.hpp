// Copyright 2026 - Apache-2.0
#ifndef ISAAC_ROS_ONNX_INFERENCE__ONNX_INFERENCE_CORE_HPP_
#define ISAAC_ROS_ONNX_INFERENCE__ONNX_INFERENCE_CORE_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace nvidia::isaac_ros::onnx_inference
{

/// Flat host-memory tensor passed to/from RunInference.
struct HostTensor
{
  std::string name;
  ONNXTensorElementDataType dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> data;  // raw bytes, row-major
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
  };

  explicit OnnxInferenceCore(const Config & cfg);
  ~OnnxInferenceCore() = default;

  // Non-copyable
  OnnxInferenceCore(const OnnxInferenceCore &) = delete;
  OnnxInferenceCore & operator=(const OnnxInferenceCore &) = delete;

  std::vector<HostTensor> RunInference(const std::vector<HostTensor> & inputs);

  size_t GetInputCount() const;
  size_t GetOutputCount() const;

private:
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
};

}  // namespace nvidia::isaac_ros::onnx_inference

#endif  // ISAAC_ROS_ONNX_INFERENCE__ONNX_INFERENCE_CORE_HPP_
