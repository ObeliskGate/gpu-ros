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

#ifndef GPU_ROS_ONNX_INFERENCE__NATIVE_ONNX_EXECUTOR_HPP_
#define GPU_ROS_ONNX_INFERENCE__NATIVE_ONNX_EXECUTOR_HPP_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "gpu_ros_nvidia_tensor_bundle_compat/native_tensor_list.hpp"
#include "gpu_ros_onnx_inference/binding_report.hpp"
#include "gpu_ros_onnx_inference/onnx_session.hpp"
#include "gpu_ros_onnx_inference/tensor_contract.hpp"

namespace gpu_ros::onnx_inference
{

namespace native = gpu_ros::nvidia_tensor_bundle_compat::native;

// Internal, serialized executor. The node owns callback admission and serialization.
class NativeOnnxExecutor
{
public:
  struct Config
  {
    OnnxSession::Config session;
    std::vector<std::string> output_contracts;
    std::string binding_report_path;
  };

  explicit NativeOnnxExecutor(const Config & config);
  native::TensorList Run(native::TensorList inputs, native::TensorListTransport & transport);
  size_t successful_runs() const noexcept { return successful_runs_; }
  bool poisoned() const noexcept { return poisoned_; }
  bool IsProfilingEnabled() const noexcept;
  std::string EndProfiling();

private:
  friend class NativeOnnxExecutorTestPeer;
  std::shared_ptr<OnnxSession> session_;
  int device_id_;
  std::vector<TensorContract> model_inputs_;
  std::vector<TensorContract> output_plan_;
  std::vector<native::TensorSpec> output_specs_;
  BindingReportContext report_context_;
  BindingReportWriter report_writer_;
  size_t successful_runs_{0};
  bool poisoned_{false};
};

} // namespace gpu_ros::onnx_inference

#endif // GPU_ROS_ONNX_INFERENCE__NATIVE_ONNX_EXECUTOR_HPP_
