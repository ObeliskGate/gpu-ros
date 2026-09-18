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

#ifndef GPU_ROS_ONNX_INFERENCE__NITROS_MANAGED_TENSOR_BUNDLE_ADAPTER_HPP_
#define GPU_ROS_ONNX_INFERENCE__NITROS_MANAGED_TENSOR_BUNDLE_ADAPTER_HPP_

#include <cuda_runtime_api.h>

#include <memory>
#include <string>

#include "gpu_ros_managed_tensor_bundle/tensor_bundle.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"
#include "gpu_ros_onnx_inference/tensor_types.hpp"

namespace gpu_ros::onnx_inference
{

// NITROS 4.5 exposes a producer-completion event only through a ReadHandle.
// This adapter waits on a private CUDA stream before turning the pointer into a
// ready Managed DeviceBuffer. The handoff does not copy tensor payload bytes.
class NitrosToManagedTensorBundleAdapter
{
public:
  explicit NitrosToManagedTensorBundleAdapter(int gpu_device_id);
  ~NitrosToManagedTensorBundleAdapter();
  NitrosToManagedTensorBundleAdapter(const NitrosToManagedTensorBundleAdapter &) = delete;
  NitrosToManagedTensorBundleAdapter & operator=(
    const NitrosToManagedTensorBundleAdapter &) = delete;

  gpu_ros_managed::ManagedTensorBundle Convert(
    const nvidia::isaac_ros::nitros::NitrosTensorListView & view) const;

private:
  int gpu_device_id_;
  cudaStream_t readiness_stream_{nullptr};
};

const std::string & NitrosTensorBundleFormat();

// Both overloads transfer the DeviceBuffer owner into the NITROS release
// callback. They never allocate or copy a tensor payload.
nvidia::isaac_ros::nitros::NitrosTensorList BuildNitrosTensorBundle(
  gpu_ros_managed::ManagedTensorBundleView input, int gpu_device_id);
nvidia::isaac_ros::nitros::NitrosTensorList BuildNitrosTensorBundle(
  TensorBundleOutput && output, int gpu_device_id);

} // namespace gpu_ros::onnx_inference

#endif // GPU_ROS_ONNX_INFERENCE__NITROS_MANAGED_TENSOR_BUNDLE_ADAPTER_HPP_
