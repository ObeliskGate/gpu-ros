// Copyright 2026 Maintainer
// Licensed under the Apache License, Version 2.0.
#ifndef ISAAC_ROS_ONNX_INFERENCE__MANAGED_IO_CONTRACT_HPP_
#define ISAAC_ROS_ONNX_INFERENCE__MANAGED_IO_CONTRACT_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace nvidia::isaac_ros::onnx_inference
{

struct ManagedTensorContract
{
  std::string name;
  ONNXTensorElementDataType dtype{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
  std::vector<int64_t> shape;
};

std::vector<ManagedTensorContract> ParseManagedTensorContracts(
  const std::vector<std::string> & specifications,
  const char * parameter_name);

size_t ManagedTensorElementSize(ONNXTensorElementDataType dtype);
size_t ManagedTensorByteSize(const ManagedTensorContract & contract);

void ValidateManagedTensorContracts(
  Ort::Session & session,
  const std::vector<ManagedTensorContract> & inputs,
  const std::vector<ManagedTensorContract> & outputs);

}  // namespace nvidia::isaac_ros::onnx_inference

#endif  // ISAAC_ROS_ONNX_INFERENCE__MANAGED_IO_CONTRACT_HPP_
