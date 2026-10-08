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

#include "gpu_ros_onnx_inference/binding_report.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
namespace gpu_ros::onnx_inference
{
std::string PointerString(const void * pointer)
{
  std::ostringstream stream;
  stream << "0x" << std::hex << reinterpret_cast<std::uintptr_t>(pointer);
  return stream.str();
}
namespace
{
std::string JsonEscape(const std::string & value)
{
  std::ostringstream escaped;
  for (const char character : value) {
    switch (character) {
      case '\\':
        escaped << "\\\\";
        break;
      case '"':
        escaped << "\\\"";
        break;
      case '\n':
        escaped << "\\n";
        break;
      case '\r':
        escaped << "\\r";
        break;
      case '\t':
        escaped << "\\t";
        break;
      default:
        escaped << character;
        break;
    }
  }
  return escaped.str();
}

const char * ExecutionProviderName(ExecutionProvider provider)
{
  switch (provider) {
    case ExecutionProvider::kCuda:
      return "CUDAExecutionProvider";
    case ExecutionProvider::kRocm:
      return "ROCMExecutionProvider";
    case ExecutionProvider::kMigraphx:
      return "MIGraphXExecutionProvider";
    case ExecutionProvider::kCpu:
      return "CPUExecutionProvider";
  }
  return "unknown";
}
} // namespace
void BindingReportWriter::WriteFirstSuccessfulFrame(const BindingReportContext & context,
  const std::vector<TensorBindingRecord> & inputs,
  const std::vector<TensorBindingRecord> & outputs)
{
  if (context.path.empty() || written_) {
    return;
  }

  const std::filesystem::path report_path(context.path);
  if (report_path.has_parent_path()) {
    std::filesystem::create_directories(report_path.parent_path());
  }
  std::ofstream report(report_path, std::ios::out | std::ios::trunc);
  if (!report) {
    throw std::runtime_error(
      "Unable to write ONNX Runtime binding report: " + context.path);
  }

  const auto write_tensor = [&report](
                              const TensorBindingRecord & tensor, const char * pointer_key) {
    report << "    {\n"
           << "      \"name\": \"" << JsonEscape(tensor.name) << "\",\n"
           << "      \"bytes\": " << tensor.bytes << ",\n"
           << "      \"storage\": \"" << JsonEscape(tensor.storage) << "\",\n"
           << "      \"" << pointer_key << "\": \"" << tensor.pointer << "\",\n"
           << "      \"ort_pointer\": \"" << tensor.ort_pointer << "\",\n"
           << "      \"pointer_identity\": " << (tensor.pointer_identity ? "true" : "false")
           << ",\n"
           << "      \"lifetime_path\": \"" << JsonEscape(tensor.lifetime_path) << "\"\n"
           << "    }";
  };
  const auto dtype_name = [](ONNXTensorElementDataType dtype) {
    switch (dtype) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return "float32";
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return "int64";
      default:
        return "unknown";
    }
  };
  const auto write_contract = [&report, &dtype_name](const TensorContract & contract) {
    report << "    {\n"
           << "      \"name\": \"" << JsonEscape(contract.name) << "\",\n"
           << "      \"dtype\": \"" << dtype_name(contract.dtype) << "\",\n"
           << "      \"shape\": [";
    for (size_t index = 0; index < contract.shape.size(); ++index) {
      if (index != 0) {
        report << ", ";
      }
      report << contract.shape[index];
    }
    report << "]\n    }";
  };

  report << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"provider\": \"" << ExecutionProviderName(context.provider) << "\",\n"
         << "  \"transport\": \"" << JsonEscape(context.transport) << "\",\n"
         << "  \"output_placement\": \""
         << JsonEscape(context.output_placement) << "\",\n"
         << "  \"managed_io_contract\": \"" << JsonEscape(context.mode)
         << "\",\n"
         << "  \"managed_pool_capacity\": " << context.pool_capacity << ",\n"
         << "  \"managed_pool_wait_timeout_ms\": " << context.pool_wait_timeout_ms << ",\n"
         << "  \"managed_input_contracts\": [\n";
  for (size_t index = 0; index < context.input_contracts.size(); ++index) {
    if (index != 0) {
      report << ",\n";
    }
    write_contract(context.input_contracts[index]);
  }
  report << "\n  ],\n  \"managed_output_contracts\": [\n";
  for (size_t index = 0; index < context.output_contracts.size(); ++index) {
    if (index != 0) {
      report << ",\n";
    }
    write_contract(context.output_contracts[index]);
  }
  report << "\n  ],\n"
         << "  \"first_frame\": true,\n"
         << "  \"inputs\": [\n";
  for (size_t index = 0; index < inputs.size(); ++index) {
    if (index != 0) {
      report << ",\n";
    }
    write_tensor(inputs[index], "input_pointer");
  }
  report << "\n  ],\n  \"outputs\": [\n";
  for (size_t index = 0; index < outputs.size(); ++index) {
    if (index != 0) {
      report << ",\n";
    }
    write_tensor(outputs[index], "output_pointer");
  }
  report << "\n  ]\n}\n";
  report.flush();
  if (!report) {
    throw std::runtime_error(
      "Failed while writing ONNX Runtime binding report: " + context.path);
  }
  written_ = true;
}
} // namespace gpu_ros::onnx_inference
