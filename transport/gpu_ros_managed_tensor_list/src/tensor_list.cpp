// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright 2026 gpu_ros_managed contributors
// Licensed under the Apache License, Version 2.0.
// Modified from NVIDIA Isaac ROS NITROS TensorList sources for managed
// backend-neutral storage.
#include "gpu_ros_managed_tensor_list/tensor_list.hpp"

#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace gpu_ros_managed
{
size_t bytes_per_element(TensorDataType dtype)
{
  switch (dtype) {
    case TensorDataType::kInt8:
    case TensorDataType::kUInt8: return 1;
    case TensorDataType::kInt16:
    case TensorDataType::kUInt16: return 2;
    case TensorDataType::kInt32:
    case TensorDataType::kUInt32:
    case TensorDataType::kFloat32: return 4;
    case TensorDataType::kInt64:
    case TensorDataType::kUInt64:
    case TensorDataType::kFloat64: return 8;
  }
  throw std::invalid_argument("Unsupported TensorDataType");
}

size_t tensor_byte_size(const std::vector<int64_t> & shape, TensorDataType dtype)
{
  if (shape.empty()) {throw std::invalid_argument("Tensor rank must be at least one");}
  size_t count = 1;
  for (const int64_t dimension : shape) {
    if (dimension <= 0) {throw std::invalid_argument("Tensor dimensions must be positive");}
    const auto value = static_cast<size_t>(dimension);
    if (count > std::numeric_limits<size_t>::max() / value) {
      throw std::overflow_error("Tensor element count overflows size_t");
    }
    count *= value;
  }
  const size_t element = bytes_per_element(dtype);
  if (count > std::numeric_limits<size_t>::max() / element) {
    throw std::overflow_error("Tensor byte size overflows size_t");
  }
  return count * element;
}

std::vector<uint64_t> contiguous_strides(
  const std::vector<int64_t> & shape, TensorDataType dtype)
{
  static_cast<void>(tensor_byte_size(shape, dtype));
  std::vector<uint64_t> strides(shape.size());
  strides.back() = bytes_per_element(dtype);
  for (size_t i = shape.size() - 1; i > 0; --i) {
    const auto next = static_cast<uint64_t>(shape[i]);
    if (strides[i] > std::numeric_limits<uint64_t>::max() / next) {
      throw std::overflow_error("Tensor stride overflows uint64_t");
    }
    strides[i - 1] = strides[i] * next;
  }
  return strides;
}

ManagedTensor::ManagedTensor(
  std::string name, TensorDataType dtype, std::vector<int64_t> shape,
  TensorStorage storage, std::vector<uint64_t> strides)
: name_(std::move(name)), dtype_(dtype), shape_(std::move(shape)),
  strides_(strides.empty() ? contiguous_strides(shape_, dtype_) : std::move(strides)),
  storage_(std::move(storage)), byte_size_(tensor_byte_size(shape_, dtype_))
{
  if (strides_ != contiguous_strides(shape_, dtype_)) {
    throw std::invalid_argument("Only contiguous tensor strides are supported");
  }
  const size_t storage_size = std::visit([](const auto & value) {
      if constexpr (std::is_same_v<std::decay_t<decltype(value)>, HostBuffer>) {
        return value.size();
      } else {
        if (!value) {throw std::invalid_argument("DeviceBuffer storage is null");}
        return value->size();
      }
    }, storage_);
  if (storage_size < byte_size_) {throw std::invalid_argument("Tensor storage is too small");}
}

ManagedTensor ManagedTensor::from_host_external(
  std::string name, TensorDataType dtype, std::vector<int64_t> shape,
  std::shared_ptr<const void> owner, const void * pointer, size_t size,
  std::vector<uint64_t> strides)
{
  return ManagedTensor(
    std::move(name), dtype, std::move(shape),
    HostBuffer(std::move(owner), pointer, size), std::move(strides));
}
ManagedTensor ManagedTensor::from_host_copy(
  std::string name, TensorDataType dtype, std::vector<int64_t> shape,
  const void * pointer, size_t size)
{
  return ManagedTensor(
    std::move(name), dtype, std::move(shape), HostBuffer::copy(pointer, size));
}

ManagedTensorList::ManagedTensorList(
  std_msgs::msg::Header header, std::vector<ManagedTensor> tensors)
: header_(std::move(header)), tensors_(std::move(tensors))
{
  if (tensors_.empty()) {return;}
  const bool host = tensors_.front().is_host();
  DeviceId device{};
  if (!host) {
    device = std::get<std::shared_ptr<DeviceBuffer>>(tensors_.front().storage())->device_id();
  }
  for (const auto & tensor : tensors_) {
    if (tensor.is_host() != host) {
      throw std::invalid_argument("A TensorList cannot mix host and device tensors");
    }
    if (!host &&
      std::get<std::shared_ptr<DeviceBuffer>>(tensor.storage())->device_id() != device)
    {
      throw std::invalid_argument("A TensorList cannot mix backend/device allocations");
    }
  }
}
const ManagedTensor & ManagedTensorList::get_tensor(const std::string & name) const
{
  for (const auto & tensor : tensors_) {
    if (tensor.name() == name) {return tensor;}
  }
  throw std::out_of_range("Tensor with name '" + name + "' not found");
}

PooledTensor tensor_from_pool(
  std::string name, TensorDataType dtype, std::vector<int64_t> shape,
  FixedDeviceMemoryPool & pool, const DeviceStream & stream)
{
  auto block = pool.acquire(stream);
  ManagedTensor tensor(
    std::move(name), dtype, std::move(shape), block.buffer);
  return PooledTensor(std::move(tensor), std::move(block.writer));
}
PooledTensor tensor_from_external(
  std::string name, TensorDataType dtype, std::vector<int64_t> shape,
  std::shared_ptr<DeviceBuffer> buffer, const DeviceStream & stream)
{
  ManagedTensor tensor(std::move(name), dtype, std::move(shape), buffer);
  return PooledTensor(std::move(tensor), buffer->get_write_handle(stream));
}
}  // namespace gpu_ros_managed
