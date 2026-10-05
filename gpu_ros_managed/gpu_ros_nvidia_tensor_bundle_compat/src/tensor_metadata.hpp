// Copyright 2026 gpu_ros_managed contributors
// Licensed under the Apache License, Version 2.0.
#ifndef GPU_ROS_NVIDIA_TENSOR_BUNDLE_COMPAT__TENSOR_METADATA_HPP_
#define GPU_ROS_NVIDIA_TENSOR_BUNDLE_COMPAT__TENSOR_METADATA_HPP_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include "gpu_ros_managed_tensor_bundle/tensor_bundle.hpp"
#include "isaac_ros_tensor_msgs/msg/tensor_list.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"

namespace gpu_ros::nvidia_tensor_bundle_compat::detail
{
struct ScalarType
{
  gpu_ros_managed::TensorDataType project;
  uint8_t code;
  uint8_t bits;
};
// DLPack codes are independent of the project's stable wire enum.
inline constexpr ScalarType kScalarTypes[]{
  {gpu_ros_managed::TensorDataType::kInt8, 0, 8},
  {gpu_ros_managed::TensorDataType::kUInt8, 1, 8},
  {gpu_ros_managed::TensorDataType::kInt16, 0, 16},
  {gpu_ros_managed::TensorDataType::kUInt16, 1, 16},
  {gpu_ros_managed::TensorDataType::kInt32, 0, 32},
  {gpu_ros_managed::TensorDataType::kUInt32, 1, 32},
  {gpu_ros_managed::TensorDataType::kInt64, 0, 64},
  {gpu_ros_managed::TensorDataType::kUInt64, 1, 64},
  {gpu_ros_managed::TensorDataType::kFloat32, 2, 32},
  {gpu_ros_managed::TensorDataType::kFloat64, 2, 64}};

inline gpu_ros_managed::TensorDataType ReadScalarType(
  const tensor_msgs::msg::ExperimentalTensor & tensor)
{
  if (tensor.dtype_lanes != 1) {
    throw std::invalid_argument("TensorList requires scalar dtype_lanes == 1");
  }
  for (const auto & type : kScalarTypes) {
    if (type.code == tensor.dtype_code && type.bits == tensor.dtype_bits) {
      return type.project;
    }
  }
  throw std::invalid_argument("TensorList has unsupported dtype_code/dtype_bits");
}

inline void WriteScalarType(
  tensor_msgs::msg::ExperimentalTensor & tensor, gpu_ros_managed::TensorDataType dtype)
{
  for (const auto & type : kScalarTypes) {
    if (type.project == dtype) {
      tensor.dtype_code = type.code;
      tensor.dtype_bits = type.bits;
      tensor.dtype_lanes = 1;
      return;
    }
  }
  throw std::invalid_argument("TensorList has unsupported project dtype");
}

inline void ValidateNames(const isaac_ros_tensor_msgs::msg::TensorList & message)
{
  if (message.names.size() != message.tensors.size()) {
    throw std::invalid_argument("TensorList names/tensors size mismatch");
  }
}

// Call only after tensor_byte_size has validated positive shape and byte overflow.
// DLPack strides are signed element counts, not Managed's unsigned byte counts.
inline int64_t NextElementStride(int64_t stride, int64_t dimension)
{
  if (stride > std::numeric_limits<int64_t>::max() / dimension) {
    throw std::overflow_error("TensorList element stride overflows int64_t");
  }
  return stride * dimension;
}

inline gpu_ros_managed::TensorDataType ValidateTensor(
  const tensor_msgs::msg::ExperimentalTensor & tensor)
{
  const auto dtype = ReadScalarType(tensor);
  const auto bytes = gpu_ros_managed::tensor_byte_size(tensor.shape, dtype);
  if (tensor.byte_offset != 0) {
    throw std::invalid_argument("TensorList byte_offset views are unsupported");
  }
  if (tensor.data.size() != bytes) {
    throw std::invalid_argument("TensorList payload size does not match shape and dtype");
  }
  if (!tensor.strides.empty() && tensor.strides.size() != tensor.shape.size()) {
    throw std::invalid_argument("TensorList strides rank does not match shape");
  }
  int64_t stride = 1;
  for (size_t i = tensor.shape.size(); i > 0; --i) {
    if (!tensor.strides.empty() && tensor.strides[i - 1] != stride) {
      throw std::invalid_argument("TensorList requires contiguous element strides");
    }
    if (i > 1) {
      stride = NextElementStride(stride, tensor.shape[i - 1]);
    }
  }
  return dtype;
}

inline size_t SetTensorMetadata(tensor_msgs::msg::ExperimentalTensor & tensor,
  gpu_ros_managed::TensorDataType dtype, const std::vector<int64_t> & shape)
{
  WriteScalarType(tensor, dtype);
  const auto bytes = gpu_ros_managed::tensor_byte_size(shape, dtype);
  tensor.shape = shape;
  tensor.strides.resize(shape.size());
  int64_t stride = 1;
  for (size_t i = shape.size(); i > 0; --i) {
    tensor.strides[i - 1] = stride;
    if (i > 1) {
      stride = NextElementStride(stride, shape[i - 1]);
    }
  }
  tensor.byte_offset = 0;
  return bytes;
}
} // namespace gpu_ros::nvidia_tensor_bundle_compat::detail
#endif
