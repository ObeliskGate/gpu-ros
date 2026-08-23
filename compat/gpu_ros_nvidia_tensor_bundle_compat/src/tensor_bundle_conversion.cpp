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

#include "gpu_ros_nvidia_tensor_bundle_compat/tensor_bundle_conversion.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_ros::nvidia_tensor_bundle_compat
{
namespace
{

uint8_t ToBundleDataType(int32_t nvidia_data_type)
{
  // isaac_ros_tensor_list_interfaces exposes a GXF-derived integer field,
  // while TensorBundle owns its wire enum.  Keep this mapping explicit at the
  // NVIDIA boundary even though the supported numeric values currently align.
  switch (nvidia_data_type) {
    case 1: return gpu_ros_tensor_bundle_msgs::msg::Tensor::INT8;
    case 2: return gpu_ros_tensor_bundle_msgs::msg::Tensor::UINT8;
    case 3: return gpu_ros_tensor_bundle_msgs::msg::Tensor::INT16;
    case 4: return gpu_ros_tensor_bundle_msgs::msg::Tensor::UINT16;
    case 5: return gpu_ros_tensor_bundle_msgs::msg::Tensor::INT32;
    case 6: return gpu_ros_tensor_bundle_msgs::msg::Tensor::UINT32;
    case 7: return gpu_ros_tensor_bundle_msgs::msg::Tensor::INT64;
    case 8: return gpu_ros_tensor_bundle_msgs::msg::Tensor::UINT64;
    case 9: return gpu_ros_tensor_bundle_msgs::msg::Tensor::FLOAT32;
    case 10: return gpu_ros_tensor_bundle_msgs::msg::Tensor::FLOAT64;
    default:
      throw std::invalid_argument("TensorBundle compatibility: unsupported NVIDIA data_type " +
              std::to_string(nvidia_data_type));
  }
}

int32_t ToNvidiaDataType(uint8_t bundle_data_type)
{
  switch (bundle_data_type) {
    case gpu_ros_tensor_bundle_msgs::msg::Tensor::INT8: return 1;
    case gpu_ros_tensor_bundle_msgs::msg::Tensor::UINT8: return 2;
    case gpu_ros_tensor_bundle_msgs::msg::Tensor::INT16: return 3;
    case gpu_ros_tensor_bundle_msgs::msg::Tensor::UINT16: return 4;
    case gpu_ros_tensor_bundle_msgs::msg::Tensor::INT32: return 5;
    case gpu_ros_tensor_bundle_msgs::msg::Tensor::UINT32: return 6;
    case gpu_ros_tensor_bundle_msgs::msg::Tensor::INT64: return 7;
    case gpu_ros_tensor_bundle_msgs::msg::Tensor::UINT64: return 8;
    case gpu_ros_tensor_bundle_msgs::msg::Tensor::FLOAT32: return 9;
    case gpu_ros_tensor_bundle_msgs::msg::Tensor::FLOAT64: return 10;
    default:
      throw std::invalid_argument("TensorBundle compatibility: unsupported data_type " +
              std::to_string(bundle_data_type));
  }
}

size_t BytesPerElement(uint8_t data_type)
{
  switch (data_type) {
    case 1:
    case 2:
      return 1;
    case 3:
    case 4:
      return 2;
    case 5:
    case 6:
    case 9:
      return 4;
    case 7:
    case 8:
    case 10:
      return 8;
    default:
      throw std::invalid_argument("TensorBundle compatibility: unsupported data_type " +
              std::to_string(data_type));
  }
}

size_t ElementCount(const std::vector<int64_t> & shape)
{
  if (shape.empty()) {
    throw std::invalid_argument("TensorBundle compatibility: tensor rank must be positive");
  }
  size_t count = 1;
  for (const int64_t dimension : shape) {
    if (dimension <= 0 || static_cast<uint64_t>(dimension) >
      std::numeric_limits<size_t>::max())
    {
      throw std::invalid_argument("TensorBundle compatibility: tensor dimensions must be positive");
    }
    const auto value = static_cast<size_t>(dimension);
    if (count > std::numeric_limits<size_t>::max() / value) {
      throw std::overflow_error("TensorBundle compatibility: element count overflow");
    }
    count *= value;
  }
  return count;
}

std::vector<uint64_t> ContiguousStrides(
  const std::vector<int64_t> & shape, uint8_t data_type)
{
  const size_t element_size = BytesPerElement(data_type);
  static_cast<void>(ElementCount(shape));
  std::vector<uint64_t> strides(shape.size());
  strides.back() = element_size;
  for (size_t index = shape.size() - 1; index > 0; --index) {
    const auto dimension = static_cast<uint64_t>(shape[index]);
    if (strides[index] > std::numeric_limits<uint64_t>::max() / dimension) {
      throw std::overflow_error("TensorBundle compatibility: stride overflow");
    }
    strides[index - 1] = strides[index] * dimension;
  }
  return strides;
}

size_t ExpectedBytes(const std::vector<int64_t> & shape, uint8_t data_type)
{
  const size_t count = ElementCount(shape);
  const size_t element_size = BytesPerElement(data_type);
  if (count > std::numeric_limits<size_t>::max() / element_size) {
    throw std::overflow_error("TensorBundle compatibility: byte size overflow");
  }
  return count * element_size;
}

std::vector<int64_t> ReadShape(const NvidiaTensor & tensor)
{
  if (tensor.shape.rank != tensor.shape.dims.size()) {
    throw std::invalid_argument(
            "TensorBundle compatibility: TensorList rank does not match dimensions");
  }
  std::vector<int64_t> shape;
  shape.reserve(tensor.shape.dims.size());
  for (const auto dimension : tensor.shape.dims) {
    if (dimension == 0) {
      throw std::invalid_argument(
              "TensorBundle compatibility: tensor dimensions must be positive");
    }
    shape.push_back(static_cast<int64_t>(dimension));
  }
  return shape;
}

void ValidateNvidiaTensor(const NvidiaTensor & tensor)
{
  const auto shape = ReadShape(tensor);
  const auto data_type = ToBundleDataType(tensor.data_type);
  const auto expected_bytes = ExpectedBytes(shape, data_type);
  if (tensor.data.size() != expected_bytes) {
    throw std::invalid_argument(
            "TensorBundle compatibility: tensor '" + tensor.name +
            "' payload size does not match shape and dtype");
  }
  const auto expected_strides = ContiguousStrides(
    shape, data_type);
  if (!tensor.strides.empty() && tensor.strides != expected_strides) {
    throw std::invalid_argument(
            "TensorBundle compatibility: tensor '" + tensor.name +
            "' is non-contiguous");
  }
}

}  // namespace

TensorBundle ToTensorBundle(const NvidiaTensorList & source)
{
  TensorBundle output;
  output.header = source.header;
  output.tensors.reserve(source.tensors.size());
  for (const auto & source_tensor : source.tensors) {
    ValidateNvidiaTensor(source_tensor);
    const auto shape = ReadShape(source_tensor);
    auto & tensor = output.tensors.emplace_back();
    tensor.name = source_tensor.name;
    tensor.data_type = ToBundleDataType(source_tensor.data_type);
    tensor.shape = shape;
    tensor.data = source_tensor.data;
  }
  return output;
}

NvidiaTensorList ToNvidiaTensorList(const TensorBundle & source)
{
  NvidiaTensorList output;
  output.header = source.header;
  output.tensors.reserve(source.tensors.size());
  for (const auto & source_tensor : source.tensors) {
    const auto expected_bytes = ExpectedBytes(source_tensor.shape, source_tensor.data_type);
    if (source_tensor.data.size() != expected_bytes) {
      throw std::invalid_argument(
              "TensorBundle compatibility: tensor '" + source_tensor.name +
              "' payload size does not match shape and dtype");
    }
    if (source_tensor.shape.size() > std::numeric_limits<uint8_t>::max()) {
      throw std::overflow_error("TensorBundle compatibility: tensor rank overflows TensorList");
    }
    std::vector<uint32_t> dimensions;
    dimensions.reserve(source_tensor.shape.size());
    for (const int64_t dimension : source_tensor.shape) {
      if (dimension <= 0 || static_cast<uint64_t>(dimension) >
        std::numeric_limits<uint32_t>::max())
      {
        throw std::invalid_argument(
                "TensorBundle compatibility: tensor dimensions do not fit TensorList");
      }
      dimensions.push_back(static_cast<uint32_t>(dimension));
    }

    auto & tensor = output.tensors.emplace_back();
    tensor.name = source_tensor.name;
    tensor.data_type = ToNvidiaDataType(source_tensor.data_type);
    tensor.shape.rank = static_cast<decltype(tensor.shape.rank)>(dimensions.size());
    tensor.shape.dims = std::move(dimensions);
    tensor.strides = ContiguousStrides(source_tensor.shape, source_tensor.data_type);
    tensor.data = source_tensor.data;
  }
  return output;
}

}  // namespace gpu_ros::nvidia_tensor_bundle_compat
