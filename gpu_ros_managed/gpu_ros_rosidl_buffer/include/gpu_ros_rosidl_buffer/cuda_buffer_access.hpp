// Copyright 2026 gpu_ros_managed contributors
// Licensed under the Apache License, Version 2.0.
#ifndef GPU_ROS_ROSIDL_BUFFER__CUDA_BUFFER_ACCESS_HPP_
#define GPU_ROS_ROSIDL_BUFFER__CUDA_BUFFER_ACCESS_HPP_

#include "gpu_ros_rosidl_buffer/buffer_access.hpp"

namespace gpu_ros::rosidl_buffer
{

// Available only from gpu_ros::rosidl_buffer_cuda. Requires a valid CUDA device
// ordinal. Accepts nonempty official cuda_buffer storage on that exact device;
// CPU buffers and other backends are rejected, never promoted. An owned stream
// waits for producer readiness and is synchronized before AcquireRead returns.
// Leases may outlive this access object. An unprovable synchronization failure
// quarantines affected storage and permanently rejects new reads on this object.
std::unique_ptr<IBufferAccess> CreateCudaBufferAccess(int device_id);

}  // namespace gpu_ros::rosidl_buffer

#endif  // GPU_ROS_ROSIDL_BUFFER__CUDA_BUFFER_ACCESS_HPP_
