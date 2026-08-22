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

#ifndef ISAAC_ROS_ONNX_INFERENCE__ONNX_DTYPE_HPP_
#define ISAAC_ROS_ONNX_INFERENCE__ONNX_DTYPE_HPP_

#include <stdexcept>
#include <string>

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace nvidia::isaac_ros::onnx_inference
{

// isaac_ros_tensor_list_interfaces Tensor.data_type uses the GXF PrimitiveType
// enum, which differs from ONNXTensorElementDataType. Phase 1 supports the two
// dtypes RT-DETR needs: float32 and int64.
constexpr int kGxfFloat32 = 9;
constexpr int kGxfInt64 = 7;

inline ONNXTensorElementDataType GxfToOnnxDtype(int gxf_dtype)
{
  switch (gxf_dtype) {
    case kGxfFloat32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case kGxfInt64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    default:
      throw std::runtime_error(
              "OnnxInference: unsupported input data_type " + std::to_string(gxf_dtype));
  }
}

inline int OnnxToGxfDtype(ONNXTensorElementDataType onnx_dtype)
{
  switch (onnx_dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return kGxfFloat32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return kGxfInt64;
    default:
      throw std::runtime_error(
              "OnnxInference: unsupported output dtype " +
              std::to_string(static_cast<int>(onnx_dtype)));
  }
}

}  // namespace nvidia::isaac_ros::onnx_inference

#endif  // ISAAC_ROS_ONNX_INFERENCE__ONNX_DTYPE_HPP_
