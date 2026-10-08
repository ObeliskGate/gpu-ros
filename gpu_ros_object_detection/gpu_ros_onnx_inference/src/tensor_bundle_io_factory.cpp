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

#include <stdexcept>
#include <string>

#include "gpu_ros_onnx_inference/tensor_bundle_io.hpp"

namespace gpu_ros::onnx_inference
{

std::unique_ptr<ITensorBundleIO> CreateStdTensorBundleIO(rclcpp::Node * node, bool publish_output);
std::unique_ptr<ITensorBundleIO> CreateManagedTensorBundleIO(
  rclcpp::Node * node, bool publish_output);
std::unique_ptr<ITensorBundleIO> CreateRosidlBufferTensorBundleIO(rclcpp::Node *, bool);
#ifdef BUILD_NATIVE_TENSOR_LIST_TRANSPORT
std::unique_ptr<ITensorBundleIO> CreateTensorListBufferIO(rclcpp::Node *, bool);
#endif

std::unique_ptr<ITensorBundleIO> CreateTensorBundleIO(
  rclcpp::Node * node, const std::string & transport, bool publish_output)
{
  if (transport == "std") {
    return CreateStdTensorBundleIO(node, publish_output);
  }
  if (transport == "managed") {
    return CreateManagedTensorBundleIO(node, publish_output);
  }
  if (transport == "rosidl_buffer") {
    const auto format = node->has_parameter("message_format") ?
      node->get_parameter("message_format").as_string() : "tensor_bundle";
    if (format == "tensor_bundle") {
      return CreateRosidlBufferTensorBundleIO(node, publish_output);
    }
#ifdef BUILD_NATIVE_TENSOR_LIST_TRANSPORT
    if (format == "tensor_list") { return CreateTensorListBufferIO(node, publish_output); }
#endif
    throw std::invalid_argument("unsupported Buffer message_format: " + format);
  }
  throw std::invalid_argument("Unknown transport: " + transport);
}

} // namespace gpu_ros::onnx_inference
