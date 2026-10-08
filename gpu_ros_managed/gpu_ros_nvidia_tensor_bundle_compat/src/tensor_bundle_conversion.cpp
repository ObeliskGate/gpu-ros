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

#include <stdexcept>
#include "tensor_metadata.hpp"

namespace gpu_ros::nvidia_tensor_bundle_compat
{
namespace
{
using WireTensor = gpu_ros_tensor_bundle_msgs::msg::Tensor;
struct WireScalar
{
  uint8_t wire;
  uint8_t code;
  uint8_t bits;
};
constexpr WireScalar kWireScalars[]{
  {WireTensor::INT8, 0, 8}, {WireTensor::UINT8, 1, 8},
  {WireTensor::INT16, 0, 16}, {WireTensor::UINT16, 1, 16},
  {WireTensor::INT32, 0, 32}, {WireTensor::UINT32, 1, 32},
  {WireTensor::INT64, 0, 64}, {WireTensor::UINT64, 1, 64},
  {WireTensor::FLOAT32, 2, 32}, {WireTensor::FLOAT64, 2, 64}};
uint8_t WireType(const NvidiaTensor & tensor)
{
  for (const auto & scalar : kWireScalars) {
    if (scalar.code == tensor.dtype_code && scalar.bits == tensor.dtype_bits) {
      return scalar.wire;
    }
  }
  throw std::invalid_argument("TensorList has unsupported scalar dtype");
}
size_t SetWireMetadata(NvidiaTensor & output, const WireTensor & input)
{
  for (const auto & scalar : kWireScalars) {
    if (scalar.wire == input.data_type) {
      return detail::SetTensorMetadata(output, scalar.code, scalar.bits, 1, input.shape);
    }
  }
  throw std::invalid_argument("TensorBundle has unsupported scalar dtype");
}

// Standard TensorBundle conversion is an explicit host boundary. Never clone
// a non-CPU Buffer: materialize its payload and move the resulting CPU vector.
void CopyHostPayload(const rosidl::Buffer<uint8_t> & source, rosidl::Buffer<uint8_t> & output)
{
  if (source.get_backend_type() == "cpu") {
    output = source;
  } else {
    output = source.to_vector();
  }
}

} // namespace

TensorBundle ToTensorBundle(const NvidiaTensorList & source)
{
  detail::ValidateNames(source);
  TensorBundle output;
  output.header = source.header;
  output.tensors.reserve(source.tensors.size());
  for (size_t i = 0; i < source.tensors.size(); ++i) {
    const auto & source_tensor = source.tensors[i];
    detail::ValidateTensor(source_tensor);
    auto & tensor = output.tensors.emplace_back();
    tensor.name = source.names[i];
    tensor.data_type = WireType(source_tensor);
    tensor.shape = source_tensor.shape;
    CopyHostPayload(source_tensor.data, tensor.data);
  }
  return output;
}

NvidiaTensorList ToNvidiaTensorList(const TensorBundle & source)
{
  NvidiaTensorList output;
  output.header = source.header;
  output.names.reserve(source.tensors.size());
  output.tensors.reserve(source.tensors.size());
  for (const auto & source_tensor : source.tensors) {
    auto & tensor = output.tensors.emplace_back();
    const auto expected_bytes = SetWireMetadata(tensor, source_tensor);
    if (source_tensor.data.size() != expected_bytes) {
      throw std::invalid_argument("TensorBundle compatibility: tensor '" + source_tensor.name +
                                  "' payload size does not match shape and dtype");
    }
    output.names.push_back(source_tensor.name);
    CopyHostPayload(source_tensor.data, tensor.data);
  }
  return output;
}

} // namespace gpu_ros::nvidia_tensor_bundle_compat
