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

#ifndef GPU_ROS_NVIDIA_TENSOR_BUNDLE_COMPAT__TENSOR_BUNDLE_CONVERSION_HPP_
#define GPU_ROS_NVIDIA_TENSOR_BUNDLE_COMPAT__TENSOR_BUNDLE_CONVERSION_HPP_

#include "gpu_ros_tensor_bundle_msgs/msg/tensor_bundle.hpp"
#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"

namespace gpu_ros::nvidia_tensor_bundle_compat
{

using TensorBundle = gpu_ros_tensor_bundle_msgs::msg::TensorBundle;
using NvidiaTensorList = isaac_ros_tensor_list_interfaces::msg::TensorList;
using NvidiaTensor = isaac_ros_tensor_list_interfaces::msg::Tensor;

// Convert only contiguous C-order tensors. The old message's rank, shape, and
// byte strides are checked explicitly; an invalid or strided tensor throws.
TensorBundle ToTensorBundle(const NvidiaTensorList & source);
NvidiaTensorList ToNvidiaTensorList(const TensorBundle & source);

} // namespace gpu_ros::nvidia_tensor_bundle_compat

#endif // GPU_ROS_NVIDIA_TENSOR_BUNDLE_COMPAT__TENSOR_BUNDLE_CONVERSION_HPP_
