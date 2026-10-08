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

#include "onnx_binding.hpp"
#include <algorithm>
#include <stdexcept>
#include <utility>
namespace gpu_ros::onnx_inference
{
void RunBound(Ort::Session & session, Ort::IoBinding & binding, BoundRunState & state,
  SubmissionHook after_submission_boundary, void * hook_context)
{
  binding.SynchronizeInputs();
  state.may_have_submitted = true;
  if (after_submission_boundary != nullptr) {
    after_submission_boundary(hook_context);
  }
  session.Run(Ort::RunOptions{nullptr}, binding);
  binding.SynchronizeOutputs();
  state.outputs_synchronized = true;
}

std::vector<TensorContract> PlanOutputBindings(Ort::Session & session,
  const std::vector<std::string> & output_names, const std::vector<TensorContract> & contracts,
  const char * contract_label)
{
  if (!contracts.empty()) {
    ValidateOutputContracts(session, contracts, contract_label);
  }
  std::vector<TensorContract> plan;
  plan.reserve(output_names.size());
  for (size_t index = 0; index < output_names.size(); ++index) {
    if (!contracts.empty()) {
      const auto found = std::find_if(contracts.begin(), contracts.end(),
        [&](const TensorContract & contract) { return contract.name == output_names[index]; });
      if (found == contracts.end()) {
        throw std::invalid_argument(std::string(contract_label) + " is missing model tensor '" +
          output_names[index] + "'");
      }
      plan.push_back(*found);
    } else {
      const auto type = session.GetOutputTypeInfo(index);
      const auto metadata = type.GetTensorTypeAndShapeInfo();
      auto shape = metadata.GetShape();
      if (!std::all_of(shape.begin(), shape.end(), [](int64_t dim) { return dim > 0; })) {
        return {};
      }
      TensorContract contract{output_names[index], metadata.GetElementType(), std::move(shape)};
      static_cast<void>(TensorByteSize(contract));
      plan.push_back(std::move(contract));
    }
  }
  return plan;
}

void ValidateBoundOutputNames(const std::vector<std::string> & expected,
  const std::vector<std::string> & actual)
{
  if (expected != actual) {
    throw std::runtime_error("ONNX Runtime returned unexpected bound output names");
  }
}
void ValidateOutputCount(size_t expected, size_t actual)
{
  if (expected != actual) {
    throw std::runtime_error("ONNX Runtime returned an unexpected number of outputs");
  }
}
TensorContract ReadOutputMetadata(const std::string & name, const Ort::Value & output)
{
  const auto metadata = output.GetTensorTypeAndShapeInfo();
  TensorContract result{name, metadata.GetElementType(), metadata.GetShape()};
  const auto bytes = TensorByteSize(result);
  if (metadata.GetElementCount() != bytes / TensorElementSize(result.dtype)) {
    throw std::runtime_error("ONNX Runtime output element count mismatch for '" + name + "'");
  }
  return result;
}
void ValidateOutputMetadata(const TensorContract & expected, const TensorContract & actual)
{
  if (actual.dtype != expected.dtype || actual.shape != expected.shape ||
      TensorByteSize(actual) != TensorByteSize(expected))
  {
    throw std::runtime_error(
      "ONNX Runtime output contract changed after Run for '" + expected.name + "'");
  }
}
void ValidateOutputMetadata(const TensorContract & expected, const Ort::Value & output)
{
  ValidateOutputMetadata(expected.name, expected.dtype, expected.shape, output);
}
void ValidateOutputMetadata(const std::string & name, ONNXTensorElementDataType dtype,
  const std::vector<int64_t> & shape, const Ort::Value & output)
{
  const auto metadata = output.GetTensorTypeAndShapeInfo();
  const auto actual_dtype = metadata.GetElementType();
  const auto actual_shape = metadata.GetShape();
  const auto actual_bytes = TensorByteSize(actual_dtype, actual_shape, name);
  if (metadata.GetElementCount() != actual_bytes / TensorElementSize(actual_dtype)) {
    throw std::runtime_error("ONNX Runtime output element count mismatch for '" + name + "'");
  }
  if (actual_dtype != dtype || actual_shape != shape ||
      actual_bytes != TensorByteSize(dtype, shape, name))
  {
    throw std::runtime_error("ONNX Runtime output contract changed after Run for '" + name + "'");
  }
}
} // namespace gpu_ros::onnx_inference
