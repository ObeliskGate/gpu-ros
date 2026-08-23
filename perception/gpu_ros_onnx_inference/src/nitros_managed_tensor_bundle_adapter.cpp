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

#include "gpu_ros_onnx_inference/nitros_managed_tensor_bundle_adapter.hpp"

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "gpu_ros_managed_cuda/cuda_backend.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_builder.hpp"
#include "gpu_ros_onnx_inference/tensor_dtype.hpp"

namespace gpu_ros::onnx_inference
{
namespace
{
namespace nitros = nvidia::isaac_ros::nitros;

void CheckCuda(cudaError_t result, const char * operation)
{
  if (result == cudaSuccess) {return;}
  throw std::runtime_error(
          std::string(operation) + ": " + cudaGetErrorName(result) + " (" +
          cudaGetErrorString(result) + ")");
}

gpu_ros_managed::TensorDataType ToManagedType(nitros::NitrosDataType dtype)
{
  switch (dtype) {
    case nitros::NitrosDataType::kFloat32:
      return gpu_ros_managed::TensorDataType::kFloat32;
    case nitros::NitrosDataType::kInt64:
      return gpu_ros_managed::TensorDataType::kInt64;
    default:
      throw std::invalid_argument(
              "NITROS-to-Managed does not support dtype " +
              std::to_string(static_cast<int>(dtype)));
  }
}

nitros::NitrosDataType ToNitrosType(gpu_ros_managed::TensorDataType dtype)
{
  switch (dtype) {
    case gpu_ros_managed::TensorDataType::kFloat32:
      return nitros::NitrosDataType::kFloat32;
    case gpu_ros_managed::TensorDataType::kInt64:
      return nitros::NitrosDataType::kInt64;
    default:
      throw std::invalid_argument(
              "Managed-to-NITROS does not support dtype " +
              std::to_string(static_cast<int>(dtype)));
  }
}

gpu_ros_managed::TensorDataType ToManagedType(ONNXTensorElementDataType dtype)
{
  return static_cast<gpu_ros_managed::TensorDataType>(OnnxToBundleDtype(dtype));
}

std::vector<int32_t> ToNitrosShape(const std::vector<int64_t> & shape)
{
  std::vector<int32_t> result;
  result.reserve(shape.size());
  for (const int64_t dimension : shape) {
    if (dimension <= 0 || dimension > std::numeric_limits<int32_t>::max()) {
      throw std::invalid_argument("Tensor shape is outside the NITROS int32 range");
    }
    result.push_back(static_cast<int32_t>(dimension));
  }
  return result;
}

void AddManagedTensor(
  nitros::NitrosTensorListBuilder & builder,
  const std::string & name,
  gpu_ros_managed::TensorDataType dtype,
  const std::vector<int64_t> & shape,
  std::shared_ptr<gpu_ros_managed::DeviceBuffer> owner,
  int gpu_device_id)
{
  if (!owner) {throw std::invalid_argument("Managed-to-NITROS received a null DeviceBuffer");}
  if (owner->device_id() !=
    gpu_ros_managed::DeviceId{gpu_ros_managed::BackendKind::kCuda, gpu_device_id})
  {
    throw std::invalid_argument("Managed-to-NITROS DeviceBuffer is on the wrong CUDA device");
  }

  // ORT I/O binding marks the output ready before it reaches this boundary.
  // Retain the lease through Build so the pointer cannot be freed early.
  auto ready = owner->get_blocking_ready_lease();
  builder.AddTensor(
    name,
    nitros::NitrosTensorBuilder()
    .WithShape(nitros::NitrosTensorShape(ToNitrosShape(shape)))
    .WithDataType(ToNitrosType(dtype))
    .WithData(const_cast<uint8_t *>(ready.data()))
    .WithReleaseCallback([owner = std::move(owner)]() mutable {owner.reset();})
    .Build());
}
}  // namespace

const std::string & NitrosTensorBundleFormat()
{
  static const std::string format =
    nitros::nitros_tensor_list_nchw_rgb_f32_t::supported_type_name;
  return format;
}

NitrosToManagedTensorBundleAdapter::NitrosToManagedTensorBundleAdapter(int gpu_device_id)
: gpu_device_id_(gpu_device_id)
{
  CheckCuda(cudaSetDevice(gpu_device_id_), "cudaSetDevice");
  CheckCuda(
    cudaStreamCreateWithFlags(&readiness_stream_, cudaStreamNonBlocking),
    "cudaStreamCreateWithFlags");
}

NitrosToManagedTensorBundleAdapter::~NitrosToManagedTensorBundleAdapter()
{
  if (readiness_stream_ == nullptr) {return;}
  static_cast<void>(cudaSetDevice(gpu_device_id_));
  static_cast<void>(cudaStreamDestroy(readiness_stream_));
}

gpu_ros_managed::ManagedTensorBundle NitrosToManagedTensorBundleAdapter::Convert(
  const nitros::NitrosTensorListView & view) const
{
  std::vector<gpu_ros_managed::ManagedTensor> tensors;
  tensors.reserve(view.GetTensorCount());
  for (size_t index = 0; index < view.GetTensorCount(); ++index) {
    // A NitrosTensor copy retains its NitrosBuffer, which in turn retains the
    // original NITROS allocation and producer/reader event bookkeeping.
    auto owner = std::make_shared<nitros::NitrosTensor>(view.get_tensor(index));
    std::vector<int64_t> shape;
    for (const auto dimension : owner->shape().dims()) {
      shape.push_back(static_cast<int64_t>(dimension));
    }

    auto read = owner->get_read_handle(readiness_stream_);
    CheckCuda(cudaStreamSynchronize(readiness_stream_), "cudaStreamSynchronize input readiness");
    auto buffer = gpu_ros_managed::cuda::adopt_synchronized_external(
      const_cast<uint8_t *>(read.get_ptr()), owner->tensor_size(), gpu_device_id_, owner);
    tensors.emplace_back(
      owner->get_name(), ToManagedType(owner->data_type()), std::move(shape),
      std::move(buffer), owner->strides());
  }

  std_msgs::msg::Header header;
  header.stamp.sec = view.GetTimestampSeconds();
  header.stamp.nanosec = view.GetTimestampNanoseconds();
  header.frame_id = view.GetFrameId();
  return gpu_ros_managed::ManagedTensorBundle(std::move(header), std::move(tensors));
}

nitros::NitrosTensorList BuildNitrosTensorBundle(
  gpu_ros_managed::ManagedTensorBundleView input, int gpu_device_id)
{
  nitros::NitrosTensorListBuilder builder;
  builder.WithHeader(input.header());
  for (const auto & tensor : input.tensors()) {
    const auto * buffer = std::get_if<std::shared_ptr<gpu_ros_managed::DeviceBuffer>>(
      &tensor.storage());
    if (buffer == nullptr) {
      throw std::invalid_argument("Managed-to-NITROS requires CUDA device tensor payloads");
    }
    AddManagedTensor(
      builder, tensor.name(), tensor.data_type(), tensor.shape(), *buffer, gpu_device_id);
  }
  return builder.Build();
}

nitros::NitrosTensorList BuildNitrosTensorBundle(
  TensorBundleOutput && output, int gpu_device_id)
{
  std::vector<gpu_ros_managed::ManagedTensor> tensors;
  tensors.reserve(output.tensors.size());
  for (auto & tensor : output.tensors) {
    auto * buffer = std::get_if<std::shared_ptr<gpu_ros_managed::DeviceBuffer>>(&tensor.storage);
    if (buffer == nullptr) {
      throw std::invalid_argument("NITROS output requires CUDA device tensor payloads");
    }
    tensors.emplace_back(
      std::move(tensor.name), ToManagedType(tensor.dtype), std::move(tensor.shape),
      std::move(*buffer));
  }
  auto list = std::make_shared<gpu_ros_managed::ManagedTensorBundle>(
    std::move(output.header), std::move(tensors));
  return BuildNitrosTensorBundle(gpu_ros_managed::ManagedTensorBundleView(std::move(list)),
    gpu_device_id);
}

}  // namespace gpu_ros::onnx_inference
