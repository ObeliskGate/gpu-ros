// Copyright 2026 gpu_ros_managed contributors
// Licensed under the Apache License, Version 2.0.
#ifndef GPU_ROS_NVIDIA_TENSOR_BUNDLE_COMPAT__TENSOR_METADATA_HPP_
#define GPU_ROS_NVIDIA_TENSOR_BUNDLE_COMPAT__TENSOR_METADATA_HPP_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include "isaac_ros_tensor_msgs/msg/tensor_list.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"

namespace gpu_ros::nvidia_tensor_bundle_compat::detail
{
struct ScalarType
{
  uint8_t code;
  uint8_t bits;
};
inline constexpr ScalarType kScalarTypes[]{
  {0, 8}, {1, 8}, {0, 16}, {1, 16}, {0, 32}, {1, 32},
  {0, 64}, {1, 64}, {2, 32}, {2, 64}};
inline size_t ScalarBytes(uint8_t code, uint8_t bits, uint16_t lanes)
{
  if (lanes != 1) {
    throw std::invalid_argument("TensorList requires scalar dtype_lanes == 1");
  }
  for (const auto & type : kScalarTypes) {
    if (type.code == code && type.bits == bits) {
      return bits / 8;
    }
  }
  throw std::invalid_argument("TensorList has unsupported dtype_code/dtype_bits");
}
inline size_t TensorBytes(uint8_t code, uint8_t bits, uint16_t lanes,
  const std::vector<int64_t> & shape)
{
  size_t bytes = ScalarBytes(code, bits, lanes);
  if (shape.empty()) {
    throw std::invalid_argument("TensorList requires positive rank");
  }
  for (const int64_t dimension : shape) {
    if (dimension <= 0) {
      throw std::invalid_argument("TensorList requires positive dimensions");
    }
    if (static_cast<uint64_t>(dimension) > std::numeric_limits<size_t>::max() / bytes) {
      throw std::overflow_error("TensorList byte size overflows size_t");
    }
    bytes *= static_cast<size_t>(dimension);
  }
  return bytes;
}
inline void ValidateNames(const isaac_ros_tensor_msgs::msg::TensorList & message)
{
  if (message.names.size() != message.tensors.size()) {
    throw std::invalid_argument("TensorList names/tensors size mismatch");
  }
}
// Shape positivity is checked by TensorBytes before computing element strides.
inline int64_t NextElementStride(int64_t stride, int64_t dimension)
{
  if (stride > std::numeric_limits<int64_t>::max() / dimension) {
    throw std::overflow_error("TensorList element stride overflows int64_t");
  }
  return stride * dimension;
}
inline size_t ValidateTensor(const tensor_msgs::msg::ExperimentalTensor & tensor)
{
  const auto bytes = TensorBytes(
    tensor.dtype_code, tensor.dtype_bits, tensor.dtype_lanes, tensor.shape);
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
  return bytes;
}
inline size_t SetTensorMetadata(tensor_msgs::msg::ExperimentalTensor & tensor,
  uint8_t code, uint8_t bits, uint16_t lanes, const std::vector<int64_t> & shape)
{
  const auto bytes = TensorBytes(code, bits, lanes, shape);
  tensor.dtype_code = code;
  tensor.dtype_bits = bits;
  tensor.dtype_lanes = lanes;
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
