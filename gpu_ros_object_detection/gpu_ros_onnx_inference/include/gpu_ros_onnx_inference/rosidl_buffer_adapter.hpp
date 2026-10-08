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

#ifndef GPU_ROS_ONNX_INFERENCE__ROSIDL_BUFFER_ADAPTER_HPP_
#define GPU_ROS_ONNX_INFERENCE__ROSIDL_BUFFER_ADAPTER_HPP_
#include "gpu_ros_rosidl_buffer/buffer_access.hpp"
#include "gpu_ros_onnx_inference/tensor_types.hpp"
#include "gpu_ros_onnx_inference/onnx_session.hpp"
#include "gpu_ros_tensor_bundle_msgs/msg/tensor_bundle.hpp"
namespace gpu_ros::onnx_inference
{
Ort::MemoryInfo ProviderMemoryInfo(ExecutionProvider provider, int device_id);
TensorBinding BindBuffer(const rosidl::Buffer<uint8_t> & buffer,
  const std::string & name, ONNXTensorElementDataType dtype,
  const std::vector<int64_t> & shape, size_t byte_offset,
  const std::vector<int64_t> & strides, std::shared_ptr<const void> message_owner,
  gpu_ros::rosidl_buffer::IBufferAccess & access, const Ort::MemoryInfo & memory);
TensorBindingBatch BindTensorBundle(
  gpu_ros_tensor_bundle_msgs::msg::TensorBundle::ConstSharedPtr message,
  ExecutionProvider provider, int device_id);
TensorBindingBatch BindTensorBundle(
  gpu_ros_tensor_bundle_msgs::msg::TensorBundle::ConstSharedPtr message,
  ExecutionProvider provider, int device_id, gpu_ros::rosidl_buffer::IBufferAccess & cpu_access,
  gpu_ros::rosidl_buffer::IBufferAccess * device_access);
std::unique_ptr<DeviceOutputAllocator> CreateBufferOutputAllocator(
  ExecutionProvider provider, int device_id);
// Every tensor in a Buffer output batch shares this original message envelope.
std::shared_ptr<gpu_ros_tensor_bundle_msgs::msg::TensorBundle> BufferOutputMessage(
  const std::vector<OutputTensor> & output);
} // namespace gpu_ros::onnx_inference
#endif
