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

#ifndef GPU_ROS_ONNX_INFERENCE__TENSOR_DTYPE_HPP_
#define GPU_ROS_ONNX_INFERENCE__TENSOR_DTYPE_HPP_

#include <cstdint>
#include <stdexcept>
#include <string>

#include "gpu_ros_tensor_bundle_msgs/msg/tensor.hpp"
#include "onnxruntime_cxx_api.h" // NOLINT

namespace gpu_ros::onnx_inference
{

// TensorBundle data_type is a project-owned wire enum and is intentionally
// converted at this boundary rather than exposing an external dtype schema.
constexpr uint8_t kTensorBundleFloat32 = gpu_ros_tensor_bundle_msgs::msg::Tensor::FLOAT32;
constexpr uint8_t kTensorBundleInt64 = gpu_ros_tensor_bundle_msgs::msg::Tensor::INT64;

inline ONNXTensorElementDataType BundleToOnnxDtype(uint8_t bundle_dtype)
{
  switch (bundle_dtype) {
    case kTensorBundleFloat32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case kTensorBundleInt64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    default:
      throw std::runtime_error(
        "OnnxInference: unsupported TensorBundle data_type " + std::to_string(bundle_dtype));
  }
}

inline uint8_t OnnxToBundleDtype(ONNXTensorElementDataType onnx_dtype)
{
  switch (onnx_dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return kTensorBundleFloat32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return kTensorBundleInt64;
    default:
      throw std::runtime_error(
        "OnnxInference: unsupported output dtype " + std::to_string(static_cast<int>(onnx_dtype)));
  }
}

} // namespace gpu_ros::onnx_inference

#endif // GPU_ROS_ONNX_INFERENCE__TENSOR_DTYPE_HPP_
