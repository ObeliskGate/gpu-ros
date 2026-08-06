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

#ifndef ISAAC_ROS_ONNX_INFERENCE__NITROS_MANAGED_TENSOR_LIST_ADAPTER_HPP_
#define ISAAC_ROS_ONNX_INFERENCE__NITROS_MANAGED_TENSOR_LIST_ADAPTER_HPP_

#include <cuda_runtime_api.h>

#include <memory>
#include <string>

#include "gpu_ros_managed_tensor_list/tensor_list.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"
#include "isaac_ros_onnx_inference/tensor_types.hpp"

namespace nvidia::isaac_ros::onnx_inference
{

// NITROS 4.5 exposes a producer-completion event only through a ReadHandle.
// This adapter waits on a private CUDA stream before turning the pointer into a
// ready Managed DeviceBuffer. The handoff does not copy tensor payload bytes.
class NitrosToManagedTensorListAdapter
{
public:
  explicit NitrosToManagedTensorListAdapter(int gpu_device_id);
  ~NitrosToManagedTensorListAdapter();
  NitrosToManagedTensorListAdapter(const NitrosToManagedTensorListAdapter &) = delete;
  NitrosToManagedTensorListAdapter & operator=(const NitrosToManagedTensorListAdapter &) = delete;

  gpu_ros_managed::ManagedTensorList Convert(
    const nvidia::isaac_ros::nitros::NitrosTensorListView & view) const;

private:
  int gpu_device_id_;
  cudaStream_t readiness_stream_{nullptr};
};

const std::string & NitrosTensorListFormat();

// Both overloads transfer the DeviceBuffer owner into the NITROS release
// callback. They never allocate or copy a tensor payload.
nvidia::isaac_ros::nitros::NitrosTensorList BuildNitrosTensorList(
  gpu_ros_managed::ManagedTensorListView input, int gpu_device_id);
nvidia::isaac_ros::nitros::NitrosTensorList BuildNitrosTensorList(
  TensorListOutput && output, int gpu_device_id);

}  // namespace nvidia::isaac_ros::onnx_inference

#endif  // ISAAC_ROS_ONNX_INFERENCE__NITROS_MANAGED_TENSOR_LIST_ADAPTER_HPP_
