// Copyright 2026 gpu_ros_managed contributors
// Licensed under the Apache License, Version 2.0.
#ifndef GPU_ROS_NVIDIA_TENSOR_BUNDLE_COMPAT__NATIVE_TENSOR_LIST_HPP_
#define GPU_ROS_NVIDIA_TENSOR_BUNDLE_COMPAT__NATIVE_TENSOR_LIST_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/header.hpp"

namespace gpu_ros::nvidia_tensor_bundle_compat::native
{
struct TensorSpec
{
  std::string name;
  uint8_t dtype_code;
  uint8_t dtype_bits;
  uint16_t dtype_lanes;
  std::vector<int64_t> shape;
};
struct CopySource
{
  const void * data;
  size_t bytes;
  bool on_device;
};
namespace detail
{
struct State;
struct Access;
}
class TensorList
{
public:
  TensorList(const TensorList &) noexcept = default;
  TensorList & operator=(const TensorList &) noexcept = default;
  const std_msgs::msg::Header & header() const noexcept;
  const std::vector<TensorSpec> & tensors() const noexcept;
  const uint8_t * data(size_t index) const;
  int device_id() const noexcept;
  std::shared_ptr<const void> owner() const noexcept;
  bool same_message(const TensorList &) const noexcept;

private:
  explicit TensorList(std::shared_ptr<detail::State>);
  std::shared_ptr<detail::State> state_;
  friend struct detail::Access;
  friend class OutputBatch;
  friend class TensorListTransport;
};
class OutputBatch
{
public:
  ~OutputBatch();
  OutputBatch(OutputBatch &&) noexcept;
  OutputBatch & operator=(OutputBatch &&) noexcept;
  OutputBatch(const OutputBatch &) = delete;
  OutputBatch & operator=(const OutputBatch &) = delete;
  const std::vector<TensorSpec> & tensors() const noexcept;
  void * data(size_t index) const;
  std::shared_ptr<const void> owner() const noexcept;
  TensorList message() const;
  void RetainOwner(std::shared_ptr<const void>);
  void CopyFrom(const std::vector<CopySource> &, std::shared_ptr<const void> source_owner);
  void CompleteAfterSync();
  void CancelBeforeSubmit() noexcept;
  void FailAfterSubmit() noexcept;

private:
  explicit OutputBatch(std::shared_ptr<detail::State>);
  std::shared_ptr<detail::State> state_;
  friend class TensorListTransport;
};
class TensorListTransport
{
public:
  using Callback = std::function<void(TensorList)>;
  TensorListTransport(rclcpp::Node *, int device_id,
    const std::string & output_topic = "tensor_output", bool publish_output = true);
  ~TensorListTransport();
  TensorListTransport(const TensorListTransport &) = delete;
  TensorListTransport & operator=(const TensorListTransport &) = delete;
  void Subscribe(Callback, const std::string & input_topic = "tensor_input");
  void Unsubscribe();
  OutputBatch Allocate(const std_msgs::msg::Header &, const std::vector<TensorSpec> &);
  void Publish(const TensorList &);

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};
} // namespace gpu_ros::nvidia_tensor_bundle_compat::native
#endif
