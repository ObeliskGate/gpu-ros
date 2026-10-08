// Copyright 2026 gpu_ros_managed contributors
// Licensed under the Apache License, Version 2.0.
#ifndef GPU_ROS_NVIDIA_TENSOR_BUNDLE_COMPAT__NATIVE_TENSOR_LIST_INTERNAL_HPP_
#define GPU_ROS_NVIDIA_TENSOR_BUNDLE_COMPAT__NATIVE_TENSOR_LIST_INTERNAL_HPP_
#include <cuda_runtime_api.h>
#include "gpu_ros_nvidia_tensor_bundle_compat/native_tensor_list.hpp"
#include "isaac_ros_tensor_msgs/msg/tensor_list.hpp"

namespace gpu_ros::nvidia_tensor_bundle_compat::native::detail
{
using Message = isaac_ros_tensor_msgs::msg::TensorList;
// The borrowed stream must remain valid through stream_owner. No ownership is
// transferred to CUDA; both entry points use the same backend import algorithm.
TensorList ImportTensorList(Message::ConstSharedPtr, int device);
TensorList ImportTensorList(Message::ConstSharedPtr, int device,
  cudaStream_t stream, std::shared_ptr<const void> stream_owner);
Message::ConstSharedPtr RawMessage(const TensorList &);
} // namespace gpu_ros::nvidia_tensor_bundle_compat::native::detail
#endif
