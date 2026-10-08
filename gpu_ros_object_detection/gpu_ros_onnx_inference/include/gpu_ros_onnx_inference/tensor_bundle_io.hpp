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

#ifndef GPU_ROS_ONNX_INFERENCE__TENSOR_BUNDLE_IO_HPP_
#define GPU_ROS_ONNX_INFERENCE__TENSOR_BUNDLE_IO_HPP_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/header.hpp"

#include "gpu_ros_onnx_inference/tensor_types.hpp"

namespace gpu_ros::onnx_inference
{

// The callback owns every ORT view and its transport read lease.
class ITensorBundleIO
{
public:
  using Callback = std::function<void(TensorBindingBatch)>;

  virtual ~ITensorBundleIO() = default;
  virtual void Subscribe(Callback callback) = 0;
  virtual OutputPlacement output_placement() const noexcept = 0;
  virtual DeviceOutputAllocator * device_output_allocator() noexcept { return nullptr; }
  virtual bool shutdown(std::chrono::milliseconds timeout) noexcept
  {
    auto * allocator = device_output_allocator();
    return !allocator || allocator->shutdown(timeout);
  }
  virtual void Publish(TensorBundleOutput && output) = 0;
};

// Factory: std, managed, or rosidl_buffer.
std::unique_ptr<ITensorBundleIO> CreateTensorBundleIO(
  rclcpp::Node * node, const std::string & transport, bool publish_output = true);

} // namespace gpu_ros::onnx_inference

#endif // GPU_ROS_ONNX_INFERENCE__TENSOR_BUNDLE_IO_HPP_
