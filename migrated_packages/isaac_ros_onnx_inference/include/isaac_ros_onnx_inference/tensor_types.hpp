// Copyright 2026 Maintainer
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

#ifndef ISAAC_ROS_ONNX_INFERENCE__TENSOR_TYPES_HPP_
#define ISAAC_ROS_ONNX_INFERENCE__TENSOR_TYPES_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "std_msgs/msg/header.hpp"
#include "gpu_ros_managed_core/buffer.hpp"

namespace nvidia::isaac_ros::onnx_inference
{
enum class OutputPlacement {kHost, kDevice};

struct OutputTensor
{
  std::string name;
  ONNXTensorElementDataType dtype{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
  std::vector<int64_t> shape;
  std::variant<
    std::vector<uint8_t>,
    std::shared_ptr<gpu_ros_managed::DeviceBuffer>> storage;
};

struct TensorListOutput
{
  std_msgs::msg::Header header;
  std::vector<OutputTensor> tensors;
};
}  // namespace nvidia::isaac_ros::onnx_inference
#endif  // ISAAC_ROS_ONNX_INFERENCE__TENSOR_TYPES_HPP_
