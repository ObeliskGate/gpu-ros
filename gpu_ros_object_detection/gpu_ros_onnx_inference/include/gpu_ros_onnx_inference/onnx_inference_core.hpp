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
#ifndef GPU_ROS_ONNX_INFERENCE__ONNX_INFERENCE_CORE_HPP_
#define GPU_ROS_ONNX_INFERENCE__ONNX_INFERENCE_CORE_HPP_
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include "gpu_ros_onnx_inference/onnx_session.hpp"
#include "gpu_ros_onnx_inference/binding_report.hpp"
#include "gpu_ros_onnx_inference/tensor_types.hpp"
namespace gpu_ros::onnx_inference
{
/// Thread-compatible: serialize inference and shutdown for each instance.
class OnnxInferenceCore
{
public:
  struct InferenceStageTiming
  {
    int64_t input_setup_ns{0};
    int64_t ort_session_run_ns{0};
    int64_t output_materialize_ns{0};
  };
  struct Config
  {
    std::string model_file_path;
    ExecutionProvider ep{ExecutionProvider::kCuda};
    int gpu_device_id{0};
    std::string ort_profile_prefix;
    std::string binding_report_path;
    std::string transport;
    std::string io_contract;
    std::vector<std::string> input_contracts;
    std::vector<std::string> output_contracts;
    size_t output_pool_capacity{16};
    std::chrono::milliseconds output_pool_wait_timeout{100};
  };
  explicit OnnxInferenceCore(const Config &);
  ~OnnxInferenceCore();
  OnnxInferenceCore(const OnnxInferenceCore &) = delete;
  OnnxInferenceCore & operator=(const OnnxInferenceCore &) = delete;
  std::vector<OutputTensor> RunInference(TensorBindingBatch inputs,
    OutputPlacement placement = OutputPlacement::kHost,
    InferenceStageTiming * timing = nullptr, DeviceOutputAllocator * allocator = nullptr);
  size_t GetInputCount() const;
  size_t GetOutputCount() const;
  bool IsProfilingEnabled() const;
  std::string EndProfiling();
  std::string OutputBindingProbeReport() const;
  bool healthy() const noexcept { return healthy_; }
  size_t successful_runs() const noexcept { return successful_runs_; }
  bool shutdown(std::chrono::milliseconds) noexcept { return healthy_; }
private:
  std::shared_ptr<OnnxSession> session_;
  ExecutionProvider execution_provider_;
  int gpu_device_id_;
  bool strict_{false};
  bool healthy_{true};
  size_t successful_runs_{0};
  std::vector<TensorContract> model_inputs_;
  std::vector<TensorContract> input_contracts_;
  std::vector<TensorContract> output_plan_;
  std::vector<DeviceOutputSpec> output_specs_;
  BindingReportContext report_context_;
  BindingReportWriter report_writer_;
};
} // namespace gpu_ros::onnx_inference
#endif // GPU_ROS_ONNX_INFERENCE__ONNX_INFERENCE_CORE_HPP_
