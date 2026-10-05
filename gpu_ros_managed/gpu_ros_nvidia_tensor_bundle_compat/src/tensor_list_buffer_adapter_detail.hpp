// Copyright 2026 gpu_ros_managed contributors
// Licensed under the Apache License, Version 2.0.
#ifndef GPU_ROS_NVIDIA_TENSOR_BUNDLE_COMPAT__ADAPTER_DETAIL_HPP_
#define GPU_ROS_NVIDIA_TENSOR_BUNDLE_COMPAT__ADAPTER_DETAIL_HPP_
// Private implementation boundary, not installed or exposed to ONNX.
#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"
#include "gpu_ros_managed_tensor_bundle/tensor_bundle.hpp"
namespace gpu_ros::nvidia_tensor_bundle_compat::detail
{
gpu_ros_managed::ManagedTensorBundle ImportTensorList(
  isaac_ros_tensor_list_interfaces::msg::TensorList::ConstSharedPtr message, int device);
isaac_ros_tensor_list_interfaces::msg::TensorList::ConstSharedPtr ReusableTensorList(
  const gpu_ros_managed::ManagedTensorBundle & bundle, int device);
}
#endif
