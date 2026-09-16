// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright 2026 gpu_ros_managed contributors
// Licensed under the Apache License, Version 2.0.
// Modified from NVIDIA Isaac ROS NITROS Tensor sources for managed
// backend-neutral storage.
#ifndef GPU_ROS_MANAGED_TENSOR_BUNDLE__TENSOR_BUNDLE_HPP_
#define GPU_ROS_MANAGED_TENSOR_BUNDLE__TENSOR_BUNDLE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "std_msgs/msg/header.hpp"
#include "gpu_ros_tensor_bundle_msgs/msg/tensor.hpp"
#include "gpu_ros_managed_core/buffer.hpp"
#include "gpu_ros_managed_core/fixed_device_memory_pool.hpp"
#include "gpu_ros_managed_core/host_buffer.hpp"

namespace gpu_ros_managed
{
enum class TensorDataType : int32_t
{
  kInt8 = gpu_ros_tensor_bundle_msgs::msg::Tensor::INT8,
  kUInt8 = gpu_ros_tensor_bundle_msgs::msg::Tensor::UINT8,
  kInt16 = gpu_ros_tensor_bundle_msgs::msg::Tensor::INT16,
  kUInt16 = gpu_ros_tensor_bundle_msgs::msg::Tensor::UINT16,
  kInt32 = gpu_ros_tensor_bundle_msgs::msg::Tensor::INT32,
  kUInt32 = gpu_ros_tensor_bundle_msgs::msg::Tensor::UINT32,
  kInt64 = gpu_ros_tensor_bundle_msgs::msg::Tensor::INT64,
  kUInt64 = gpu_ros_tensor_bundle_msgs::msg::Tensor::UINT64,
  kFloat32 = gpu_ros_tensor_bundle_msgs::msg::Tensor::FLOAT32,
  kFloat64 = gpu_ros_tensor_bundle_msgs::msg::Tensor::FLOAT64
};

size_t bytes_per_element(TensorDataType dtype);
std::vector<uint64_t> contiguous_strides(const std::vector<int64_t> & shape, TensorDataType dtype);
size_t tensor_byte_size(const std::vector<int64_t> & shape, TensorDataType dtype);

using TensorStorage = std::variant<HostBuffer, std::shared_ptr<DeviceBuffer>>;

class ManagedTensor
{
public:
  ManagedTensor(std::string name, TensorDataType dtype, std::vector<int64_t> shape,
    TensorStorage storage, std::vector<uint64_t> strides = {});

  static ManagedTensor from_host_external(std::string name, TensorDataType dtype,
    std::vector<int64_t> shape, std::shared_ptr<const void> owner, const void * pointer,
    size_t size, std::vector<uint64_t> strides = {});
  static ManagedTensor from_host_copy(std::string name, TensorDataType dtype,
    std::vector<int64_t> shape, const void * pointer, size_t size);

  const std::string & name() const noexcept { return name_; }
  TensorDataType data_type() const noexcept { return dtype_; }
  const std::vector<int64_t> & shape() const noexcept { return shape_; }
  const std::vector<uint64_t> & strides() const noexcept { return strides_; }
  size_t byte_size() const noexcept { return byte_size_; }
  const TensorStorage & storage() const noexcept { return storage_; }
  bool is_host() const noexcept { return std::holds_alternative<HostBuffer>(storage_); }

private:
  std::string name_;
  TensorDataType dtype_;
  std::vector<int64_t> shape_;
  std::vector<uint64_t> strides_;
  TensorStorage storage_;
  size_t byte_size_{0};
};

class ManagedTensorBundle
{
public:
  ManagedTensorBundle() = default;
  ManagedTensorBundle(std_msgs::msg::Header header, std::vector<ManagedTensor> tensors);
  const std_msgs::msg::Header & header() const noexcept { return header_; }
  const std::vector<ManagedTensor> & tensors() const noexcept { return tensors_; }
  const ManagedTensor & get_tensor(size_t index) const { return tensors_.at(index); }
  const ManagedTensor & get_tensor(const std::string & name) const;

private:
  std_msgs::msg::Header header_;
  std::vector<ManagedTensor> tensors_;
};

class ManagedTensorBundleView
{
public:
  using MessageType = ManagedTensorBundle;
  explicit ManagedTensorBundleView(std::shared_ptr<const ManagedTensorBundle> message)
      : message_(std::move(message))
  {
    if (!message_) {
      throw std::invalid_argument("ManagedTensorBundleView message is null");
    }
  }
  const ManagedTensorBundle & get() const noexcept { return *message_; }
  const std_msgs::msg::Header & header() const noexcept { return message_->header(); }
  const std::vector<ManagedTensor> & tensors() const noexcept { return message_->tensors(); }
  const ManagedTensor & get_tensor(size_t index) const { return message_->get_tensor(index); }
  const ManagedTensor & get_tensor(const std::string & name) const
  {
    return message_->get_tensor(name);
  }
  const std::shared_ptr<const ManagedTensorBundle> & owner() const noexcept { return message_; }

private:
  std::shared_ptr<const ManagedTensorBundle> message_;
};

class ManagedTensorBundleBuilder
{
public:
  ManagedTensorBundleBuilder & with_header(std_msgs::msg::Header header)
  {
    header_ = std::move(header);
    return *this;
  }
  ManagedTensorBundleBuilder & add_tensor(ManagedTensor tensor)
  {
    tensors_.push_back(std::move(tensor));
    return *this;
  }
  ManagedTensorBundle build() &&
  {
    return ManagedTensorBundle(std::move(header_), std::move(tensors_));
  }

private:
  std_msgs::msg::Header header_;
  std::vector<ManagedTensor> tensors_;
};

struct PooledTensor
{
  ManagedTensor tensor;
  WriteHandle writer;
  PooledTensor(ManagedTensor t, WriteHandle w) : tensor(std::move(t)), writer(std::move(w)) {}
  PooledTensor(PooledTensor &&) noexcept = default;
  PooledTensor & operator=(PooledTensor &&) = delete;
};

PooledTensor tensor_from_pool(std::string name, TensorDataType dtype, std::vector<int64_t> shape,
  FixedDeviceMemoryPool & pool, const DeviceStream & stream);
PooledTensor tensor_from_external(std::string name, TensorDataType dtype,
  std::vector<int64_t> shape, std::shared_ptr<DeviceBuffer> buffer, const DeviceStream & stream);
} // namespace gpu_ros_managed
#endif
