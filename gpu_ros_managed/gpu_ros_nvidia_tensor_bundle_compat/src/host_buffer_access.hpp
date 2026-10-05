// Copyright 2026 gpu_ros_managed contributors
// Licensed under the Apache License, Version 2.0.
#ifndef GPU_ROS_NVIDIA_TENSOR_BUNDLE_COMPAT__HOST_BUFFER_ACCESS_HPP_
#define GPU_ROS_NVIDIA_TENSOR_BUNDLE_COMPAT__HOST_BUFFER_ACCESS_HPP_
#include <vector>
#include "rosidl_buffer/buffer.hpp"
namespace gpu_ros::nvidia_tensor_bundle_compat::detail
{
template <typename T>
const std::vector<T> & HostValues(const rosidl::Buffer<T> & buffer, std::vector<T> & materialized)
{
  if (buffer.get_backend_type() == "cpu") {
    return static_cast<const std::vector<T> &>(buffer);
  }
  materialized = buffer.to_vector();
  return materialized;
}
}
#endif
