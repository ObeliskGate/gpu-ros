// Copyright 2026 gpu_ros_managed contributors
// Licensed under the Apache License, Version 2.0.
#ifndef GPU_ROS_NVIDIA_TENSOR_BUNDLE_COMPAT__TENSOR_LIST_BUFFER_ADAPTER_HPP_
#define GPU_ROS_NVIDIA_TENSOR_BUNDLE_COMPAT__TENSOR_LIST_BUFFER_ADAPTER_HPP_

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "rclcpp/rclcpp.hpp"
#include "gpu_ros_managed_tensor_bundle/tensor_bundle.hpp"

namespace gpu_ros::nvidia_tensor_bundle_compat
{
struct NativeTensorSpec
{
  std::string name;
  gpu_ros_managed::TensorDataType dtype;
  std::vector<int64_t> shape;
};

// A transaction: cancel before submission; fail (retain uncertain storage) after submission.
class NativeOutputBatch
{
public:
  ~NativeOutputBatch();
  NativeOutputBatch(NativeOutputBatch &&) noexcept;
  NativeOutputBatch & operator=(NativeOutputBatch &&) noexcept;
  NativeOutputBatch(const NativeOutputBatch &) = delete;
  NativeOutputBatch & operator=(const NativeOutputBatch &) = delete;
  const std::vector<std::shared_ptr<gpu_ros_managed::DeviceBuffer>> & buffers() const;
  void * pointer(size_t index) const;
  void RetainOwner(std::shared_ptr<const void> owner);
  void CompleteAfterSync();
  void CancelBeforeSubmit() noexcept;
  void FailAfterSubmit() noexcept;
  // Generic bridge/fallback only, never the direct inference binding path.
  void CopyFrom(const gpu_ros_managed::ManagedTensorBundle & source);

private:
  struct Impl;
  explicit NativeOutputBatch(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
  friend class TensorListTransport;
};

// NVIDIA wire messages and backend handles are deliberately private to compat.
class TensorListTransport
{
public:
  using Callback = std::function<void(gpu_ros_managed::ManagedTensorBundleView)>;
  TensorListTransport(rclcpp::Node * node, int device_id,
    const std::string & output_topic = "tensor_output", bool publish_output = true);
  ~TensorListTransport();
  TensorListTransport(const TensorListTransport &) = delete;
  TensorListTransport & operator=(const TensorListTransport &) = delete;
  void Subscribe(Callback callback, const std::string & input_topic = "tensor_input");
  void Unsubscribe();
  void Publish(const gpu_ros_managed::ManagedTensorBundle & bundle);
  NativeOutputBatch Allocate(
    const std_msgs::msg::Header & header, const std::vector<NativeTensorSpec> & specs);

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};
} // namespace gpu_ros::nvidia_tensor_bundle_compat
#endif
