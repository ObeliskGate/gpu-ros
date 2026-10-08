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

#ifndef GPU_ROS_ONNX_INFERENCE__MANAGED_TENSOR_BUNDLE_ADAPTER_HPP_
#define GPU_ROS_ONNX_INFERENCE__MANAGED_TENSOR_BUNDLE_ADAPTER_HPP_
#include "gpu_ros_managed_tensor_bundle/tensor_bundle.hpp"
#include "gpu_ros_onnx_inference/onnx_inference_core.hpp"
namespace gpu_ros::onnx_inference
{
TensorBindingBatch BindManagedTensorBundle(gpu_ros_managed::ManagedTensorBundleView,
  ExecutionProvider, int device_id, bool strict = false);
std::unique_ptr<DeviceOutputAllocator> CreateManagedOutputAllocator(const OnnxInferenceCore::Config &);
gpu_ros_managed::ManagedTensorBundle ManagedOutputMessage(TensorBundleOutput output);
} // namespace gpu_ros::onnx_inference
#endif
