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

#include "isaac_ros_onnx_inference/tensor_list_io.hpp"

namespace nvidia::isaac_ros::onnx_inference
{

std::unique_ptr<ITensorListIO> CreateStdTensorListIO(rclcpp::Node * node);
#ifdef BUILD_NITROS_TRANSPORT
std::unique_ptr<ITensorListIO> CreateNitrosTensorListIO(rclcpp::Node * node);
#endif
std::unique_ptr<ITensorListIO> CreateManagedTensorListIO(rclcpp::Node * node);

std::unique_ptr<ITensorListIO> CreateTensorListIO(
  rclcpp::Node * node, const std::string & transport)
{
  if (transport == "std") {
    return CreateStdTensorListIO(node);
  }
  if (transport == "nitros") {
#ifdef BUILD_NITROS_TRANSPORT
    return CreateNitrosTensorListIO(node);
#else
    throw std::runtime_error(
            "transport=nitros requested but built without BUILD_NITROS_TRANSPORT.");
#endif
  }
  if (transport == "managed") {
    return CreateManagedTensorListIO(node);
  }
  throw std::invalid_argument("Unknown transport: " + transport);
}

}  // namespace nvidia::isaac_ros::onnx_inference
