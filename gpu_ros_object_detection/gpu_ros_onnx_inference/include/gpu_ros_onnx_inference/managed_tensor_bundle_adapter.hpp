// Copyright 2026 Boshen Chen
// Licensed under the Apache License, Version 2.0.
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
