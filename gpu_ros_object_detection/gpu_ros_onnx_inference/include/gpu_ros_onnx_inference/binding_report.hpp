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

#ifndef GPU_ROS_ONNX_INFERENCE__BINDING_REPORT_HPP_
#define GPU_ROS_ONNX_INFERENCE__BINDING_REPORT_HPP_
#include "gpu_ros_onnx_inference/onnx_session.hpp"
#include "gpu_ros_onnx_inference/tensor_contract.hpp"
namespace gpu_ros::onnx_inference
{
struct TensorBindingRecord
{
  std::string name;
  size_t bytes{0};
  std::string storage;
  std::string pointer;
  std::string ort_pointer;
  bool pointer_identity{false};
  std::string lifetime_path;
};
struct BindingReportContext
{
  std::string path;
  ExecutionProvider provider{ExecutionProvider::kCuda};
  std::string transport;
  std::string output_placement;
  std::string mode;
  size_t pool_capacity{0};
  int64_t pool_wait_timeout_ms{0};
  std::vector<TensorContract> input_contracts;
  std::vector<TensorContract> output_contracts;
};
std::string PointerString(const void * pointer);
class BindingReportWriter
{
public:
  bool NeedsReport(const BindingReportContext & context) const noexcept
  {
    return !context.path.empty() && !written_;
  }
  void WriteFirstSuccessfulFrame(const BindingReportContext & context,
    const std::vector<TensorBindingRecord> & inputs,
    const std::vector<TensorBindingRecord> & outputs);
private:
  bool written_{false};
};
} // namespace gpu_ros::onnx_inference
#endif // GPU_ROS_ONNX_INFERENCE__BINDING_REPORT_HPP_
