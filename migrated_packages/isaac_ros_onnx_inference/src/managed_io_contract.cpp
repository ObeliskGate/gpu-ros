// Copyright 2026 Maintainer
// Licensed under the Apache License, Version 2.0.

#include "isaac_ros_onnx_inference/managed_io_contract.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace nvidia::isaac_ros::onnx_inference
{
namespace
{

std::string Trim(std::string value)
{
  const auto first = std::find_if_not(value.begin(), value.end(),
    [](unsigned char c) {return std::isspace(c) != 0;});
  const auto last = std::find_if_not(value.rbegin(), value.rend(),
    [](unsigned char c) {return std::isspace(c) != 0;}).base();
  if (first >= last) {return {};}
  return std::string(first, last);
}

ONNXTensorElementDataType ParseDtype(const std::string & text, const char * parameter_name)
{
  if (text == "float32") {return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
  if (text == "int64") {return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;}
  throw std::invalid_argument(
          std::string(parameter_name) + " uses unsupported dtype '" + text +
          "'; expected float32 or int64");
}

std::vector<int64_t> ParseShape(
  const std::string & text, const char * parameter_name)
{
  if (text.size() < 2U || text.front() != '[' || text.back() != ']') {
    throw std::invalid_argument(
            std::string(parameter_name) + " shape must use [dim0,dim1,...]");
  }
  const std::string body = text.substr(1, text.size() - 2U);
  if (body.empty()) {
    throw std::invalid_argument(std::string(parameter_name) + " rank must be positive");
  }
  std::vector<int64_t> shape;
  size_t start = 0;
  while (start <= body.size()) {
    const size_t comma = body.find(',', start);
    const std::string token = Trim(body.substr(
      start, comma == std::string::npos ? std::string::npos : comma - start));
    if (token.empty()) {
      throw std::invalid_argument(std::string(parameter_name) + " contains an empty dimension");
    }
    size_t parsed = 0;
    int64_t dimension = 0;
    try {
      dimension = std::stoll(token, &parsed, 10);
    } catch (const std::exception &) {
      throw std::invalid_argument(
              std::string(parameter_name) + " contains a non-integer dimension '" + token + "'");
    }
    if (parsed != token.size() || dimension <= 0) {
      throw std::invalid_argument(
              std::string(parameter_name) + " dimensions must be positive concrete integers");
    }
    shape.push_back(dimension);
    if (comma == std::string::npos) {break;}
    start = comma + 1U;
  }
  return shape;
}

void CheckUniqueNames(
  const std::vector<ManagedTensorContract> & contracts, const char * parameter_name)
{
  std::unordered_map<std::string, bool> seen;
  for (const auto & contract : contracts) {
    if (!seen.emplace(contract.name, true).second) {
      throw std::invalid_argument(
              std::string(parameter_name) + " contains duplicate tensor name '" +
              contract.name + "'");
    }
  }
}

std::vector<std::string> SessionNames(
  Ort::Session & session, bool inputs, Ort::AllocatorWithDefaultOptions & allocator)
{
  const size_t count = inputs ? session.GetInputCount() : session.GetOutputCount();
  std::vector<std::string> names;
  names.reserve(count);
  for (size_t index = 0; index < count; ++index) {
    auto name = inputs ? session.GetInputNameAllocated(index, allocator) :
      session.GetOutputNameAllocated(index, allocator);
    names.emplace_back(name.get());
  }
  return names;
}

void ValidateOneSide(
  Ort::Session & session,
  const std::vector<ManagedTensorContract> & contracts,
  bool inputs,
  const char * parameter_name)
{
  Ort::AllocatorWithDefaultOptions allocator;
  const auto names = SessionNames(session, inputs, allocator);
  if (names.size() != contracts.size()) {
    throw std::invalid_argument(
            std::string(parameter_name) + " count " + std::to_string(contracts.size()) +
            " does not match model " + (inputs ? "input" : "output") + " count " +
            std::to_string(names.size()));
  }
  std::unordered_map<std::string, const ManagedTensorContract *> by_name;
  for (const auto & contract : contracts) {by_name.emplace(contract.name, &contract);}
  for (size_t index = 0; index < names.size(); ++index) {
    const auto found = by_name.find(names[index]);
    if (found == by_name.end()) {
      throw std::invalid_argument(
              std::string(parameter_name) + " is missing model tensor '" + names[index] + "'");
    }
    const auto & contract = *found->second;
    const auto info = (inputs ? session.GetInputTypeInfo(index) :
      session.GetOutputTypeInfo(index)).GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != contract.dtype) {
      throw std::invalid_argument(
              std::string(parameter_name) + " dtype mismatch for '" + contract.name + "'");
    }
    const auto model_shape = info.GetShape();
    if (model_shape.size() != contract.shape.size()) {
      throw std::invalid_argument(
              std::string(parameter_name) + " rank mismatch for '" + contract.name + "'");
    }
    for (size_t dim = 0; dim < model_shape.size(); ++dim) {
      if (model_shape[dim] > 0 && model_shape[dim] != contract.shape[dim]) {
        throw std::invalid_argument(
                std::string(parameter_name) + " dimension mismatch for '" + contract.name +
                "' at dimension " + std::to_string(dim));
      }
    }
    static_cast<void>(ManagedTensorByteSize(contract));
  }
}

}  // namespace

std::vector<ManagedTensorContract> ParseManagedTensorContracts(
  const std::vector<std::string> & specifications,
  const char * parameter_name)
{
  if (specifications.empty()) {
    throw std::invalid_argument(std::string(parameter_name) + " must not be empty");
  }
  std::vector<ManagedTensorContract> contracts;
  contracts.reserve(specifications.size());
  for (const auto & specification : specifications) {
    const size_t equals = specification.find('=');
    if (equals == std::string::npos || equals == 0U ||
      specification.find('=', equals + 1U) != std::string::npos)
    {
      throw std::invalid_argument(
              std::string(parameter_name) + " entry must be name=dtype[dim0,dim1,...]: " +
              specification);
    }
    const std::string name = Trim(specification.substr(0, equals));
    if (name.empty()) {
      throw std::invalid_argument(
              std::string(parameter_name) + " entry has an empty tensor name: " + specification);
    }
    const std::string type_and_shape = Trim(specification.substr(equals + 1U));
    const size_t bracket = type_and_shape.find('[');
    if (bracket == std::string::npos) {
      throw std::invalid_argument(
              std::string(parameter_name) + " entry is missing shape: " + specification);
    }
    const auto dtype = ParseDtype(
      Trim(type_and_shape.substr(0, bracket)), parameter_name);
    const auto shape = ParseShape(type_and_shape.substr(bracket), parameter_name);
    contracts.push_back(ManagedTensorContract{name, dtype, shape});
  }
  CheckUniqueNames(contracts, parameter_name);
  return contracts;
}

size_t ManagedTensorElementSize(ONNXTensorElementDataType dtype)
{
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return sizeof(float);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return sizeof(int64_t);
    default:
      throw std::invalid_argument(
              "strict Managed I/O contract supports only float32 and int64 tensors");
  }
}

size_t ManagedTensorByteSize(const ManagedTensorContract & contract)
{
  size_t elements = 1;
  for (const int64_t dimension : contract.shape) {
    if (dimension <= 0 || static_cast<uint64_t>(dimension) >
      std::numeric_limits<size_t>::max())
    {
      throw std::invalid_argument("strict Managed I/O contract has an invalid dimension");
    }
    const auto value = static_cast<size_t>(dimension);
    if (elements > std::numeric_limits<size_t>::max() / value) {
      throw std::overflow_error(
              "strict Managed I/O contract element count overflows size_t for '" +
              contract.name + "'");
    }
    elements *= value;
  }
  const size_t element_size = ManagedTensorElementSize(contract.dtype);
  if (elements > std::numeric_limits<size_t>::max() / element_size) {
    throw std::overflow_error(
            "strict Managed I/O contract byte size overflows size_t for '" + contract.name + "'");
  }
  return elements * element_size;
}

void ValidateManagedTensorContracts(
  Ort::Session & session,
  const std::vector<ManagedTensorContract> & inputs,
  const std::vector<ManagedTensorContract> & outputs)
{
  CheckUniqueNames(inputs, "managed_input_contracts");
  CheckUniqueNames(outputs, "managed_output_contracts");
  ValidateOneSide(session, inputs, true, "managed_input_contracts");
  ValidateOneSide(session, outputs, false, "managed_output_contracts");
}

}  // namespace nvidia::isaac_ros::onnx_inference
