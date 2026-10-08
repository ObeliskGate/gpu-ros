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

#ifndef GPU_ROS_ONNX_INFERENCE__ONNX_BINDING_HPP_
#define GPU_ROS_ONNX_INFERENCE__ONNX_BINDING_HPP_
#include "gpu_ros_onnx_inference/tensor_contract.hpp"
namespace gpu_ros::onnx_inference
{
struct BoundRunState
{
  bool may_have_submitted{false};
  bool outputs_synchronized{false};
};
using SubmissionHook = void (*)(void *);
void RunBound(Ort::Session & session, Ort::IoBinding & binding, BoundRunState & state,
  SubmissionHook after_submission_boundary = nullptr, void * hook_context = nullptr);
std::vector<TensorContract> PlanOutputBindings(Ort::Session & session,
  const std::vector<std::string> & output_names, const std::vector<TensorContract> & contracts,
  const char * contract_label);
void ValidateBoundOutputNames(const std::vector<std::string> & expected,
  const std::vector<std::string> & actual);
void ValidateOutputCount(size_t expected, size_t actual);
TensorContract ReadOutputMetadata(const std::string & name, const Ort::Value & output);
void ValidateOutputMetadata(const TensorContract & expected, const TensorContract & actual);
void ValidateOutputMetadata(const TensorContract & expected, const Ort::Value & output);
void ValidateOutputMetadata(const std::string & name, ONNXTensorElementDataType dtype,
  const std::vector<int64_t> & shape, const Ort::Value & output);
} // namespace gpu_ros::onnx_inference
#endif // GPU_ROS_ONNX_INFERENCE__ONNX_BINDING_HPP_
