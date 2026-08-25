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

std::unique_ptr<ITensorBundleIO> CreateStdTensorBundleIO(
  rclcpp::Node * node, bool publish_output);
#ifdef BUILD_NITROS_TRANSPORT
std::unique_ptr<ITensorBundleIO> CreateNitrosTensorBundleIO(
  rclcpp::Node * node, bool publish_output);
#endif
std::unique_ptr<ITensorBundleIO> CreateManagedTensorBundleIO(
  rclcpp::Node * node, bool publish_output);

std::unique_ptr<ITensorBundleIO> CreateTensorBundleIO(
  rclcpp::Node * node, const std::string & transport, bool publish_output)
{
  if (transport == "std") {
    return CreateStdTensorBundleIO(node, publish_output);
  }
  if (transport == "nitros") {
#ifdef BUILD_NITROS_TRANSPORT
    return CreateNitrosTensorBundleIO(node, publish_output);
#else
    throw std::runtime_error(
            "transport=nitros requested but built without BUILD_NITROS_TRANSPORT.");
#endif
  }
  if (transport == "managed") {
    return CreateManagedTensorBundleIO(node, publish_output);
  }
  throw std::invalid_argument("Unknown transport: " + transport);
}

}  // namespace gpu_ros::onnx_inference
