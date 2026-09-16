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

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h" // NOLINT
#include "gpu_ros_managed_core/fixed_device_memory_pool.hpp"
#include "gpu_ros_managed_tensor_bundle/tensor_bundle.hpp"
#include "gpu_ros_onnx_inference/managed_io_contract.hpp"
#include "gpu_ros_onnx_inference/tensor_types.hpp"

namespace gpu_ros::onnx_inference
{

/// Supported execution providers.
enum class ExecutionProvider
{
  kCuda,
  kRocm,
  kMigraphx,
  kCpu
};

ExecutionProvider ParseExecutionProvider(const std::string & ep_str);

/// Core inference wrapper around Ort::Session.
/// Thread-compatible (not thread-safe): create one per node.
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
    std::string managed_io_contract;
    std::vector<std::string> managed_input_contracts;
    std::vector<std::string> managed_output_contracts;
    size_t managed_pool_capacity{16};
    std::chrono::milliseconds managed_pool_wait_timeout{100};
  };

  explicit OnnxInferenceCore(const Config & cfg);
  ~OnnxInferenceCore();

  // Non-copyable
  OnnxInferenceCore(const OnnxInferenceCore &) = delete;
  OnnxInferenceCore & operator=(const OnnxInferenceCore &) = delete;

  std::vector<OutputTensor> RunInference(gpu_ros_managed::ManagedTensorBundleView inputs,
    OutputPlacement output_placement = OutputPlacement::kHost,
    InferenceStageTiming * stage_timing = nullptr);

  size_t GetInputCount() const;
  size_t GetOutputCount() const;
  bool IsProfilingEnabled() const;
  std::string EndProfiling();
  std::string OutputBindingProbeReport() const;
  bool healthy() const noexcept { return strict_healthy_; }
  size_t pool_exhaustion_drops() const noexcept { return pool_exhaustion_drops_; }
  bool shutdown(std::chrono::milliseconds timeout) noexcept;

private:
  struct BindingTensorReport
  {
    std::string name;
    size_t bytes{0};
    std::string storage;
    std::string pointer;
    std::string ort_pointer;
    bool pointer_identity{false};
    std::string lifetime_path;
  };

  struct OutputBindingProbe
  {
    std::string name;
    bool metadata_shape_is_static{false};
    std::string decision;
  };

  Ort::Env env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  Ort::AllocatorWithDefaultOptions allocator_;
  bool profiling_enabled_{false};
  ExecutionProvider execution_provider_;
  int gpu_device_id_;

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<OutputBindingProbe> output_binding_probes_;
  gpu_ros_managed::DeviceStream hip_output_stream_;
  std::string binding_report_path_;
  std::string transport_;
  bool binding_report_written_{false};
  bool strict_managed_{false};
  bool strict_healthy_{true};
  size_t pool_exhaustion_drops_{0};
  size_t managed_pool_capacity_{16};
  std::chrono::milliseconds managed_pool_wait_timeout_{100};
  std::vector<ManagedTensorContract> managed_input_contracts_;
  std::vector<ManagedTensorContract> managed_output_contracts_;
  std::vector<std::unique_ptr<gpu_ros_managed::FixedDeviceMemoryPool>> managed_output_pools_;

  void WriteBindingReport(const std::vector<BindingTensorReport> & inputs,
    const std::vector<BindingTensorReport> & outputs, OutputPlacement output_placement);
  std::vector<OutputTensor> RunStrictManagedInference(
    gpu_ros_managed::ManagedTensorBundleView inputs);
};

} // namespace gpu_ros::onnx_inference

#endif // GPU_ROS_ONNX_INFERENCE__ONNX_INFERENCE_CORE_HPP_
