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

#ifndef GPU_ROS_ONNX_INFERENCE__TENSOR_TYPES_HPP_
#define GPU_ROS_ONNX_INFERENCE__TENSOR_TYPES_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "onnxruntime_cxx_api.h" // NOLINT
#include "std_msgs/msg/header.hpp"
#include "gpu_ros_managed_core/buffer.hpp"

namespace gpu_ros::onnx_inference
{
enum class OutputPlacement
{
  kHost,
  kDevice
};

struct OutputTensor
{
  std::string name;
  ONNXTensorElementDataType dtype{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
  std::vector<int64_t> shape;
  std::variant<std::vector<uint8_t>, std::shared_ptr<gpu_ros_managed::DeviceBuffer>> storage;
};

struct DeviceOutputSpec
{
  std::string name;
  ONNXTensorElementDataType dtype;
  std::vector<int64_t> shape;
};

class DeviceOutputBatch
{
public:
  virtual ~DeviceOutputBatch() = default;
  virtual const std::vector<std::shared_ptr<gpu_ros_managed::DeviceBuffer>> & buffers() const = 0;
  virtual void * pointer(size_t index) const = 0;
  virtual void RetainOwner(std::shared_ptr<const void> owner) = 0;
  virtual void CompleteAfterSync() = 0;
  virtual void CancelBeforeSubmit() noexcept = 0;
  virtual void FailAfterSubmit() noexcept = 0;
  virtual void CopyFrom(const std::vector<OutputTensor> & tensors) = 0;
};

class DeviceOutputAllocator
{
public:
  virtual ~DeviceOutputAllocator() = default;
  virtual std::unique_ptr<DeviceOutputBatch> Allocate(
    const std_msgs::msg::Header & header, const std::vector<DeviceOutputSpec> & specs) = 0;
};

struct TensorBundleOutput
{
  std_msgs::msg::Header header;
  std::vector<OutputTensor> tensors;
};
} // namespace gpu_ros::onnx_inference
#endif // GPU_ROS_ONNX_INFERENCE__TENSOR_TYPES_HPP_
