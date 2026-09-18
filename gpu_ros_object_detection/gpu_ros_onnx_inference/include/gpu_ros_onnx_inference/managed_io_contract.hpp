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

#ifndef GPU_ROS_ONNX_INFERENCE__MANAGED_IO_CONTRACT_HPP_
#define GPU_ROS_ONNX_INFERENCE__MANAGED_IO_CONTRACT_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h" // NOLINT

namespace gpu_ros::onnx_inference
{

struct ManagedTensorContract
{
  std::string name;
  ONNXTensorElementDataType dtype{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
  std::vector<int64_t> shape;
};

std::vector<ManagedTensorContract> ParseManagedTensorContracts(
  const std::vector<std::string> & specifications, const char * parameter_name);

size_t ManagedTensorElementSize(ONNXTensorElementDataType dtype);
size_t ManagedTensorByteSize(const ManagedTensorContract & contract);

void ValidateManagedTensorContracts(Ort::Session & session,
  const std::vector<ManagedTensorContract> & inputs,
  const std::vector<ManagedTensorContract> & outputs);

} // namespace gpu_ros::onnx_inference

#endif // GPU_ROS_ONNX_INFERENCE__MANAGED_IO_CONTRACT_HPP_
