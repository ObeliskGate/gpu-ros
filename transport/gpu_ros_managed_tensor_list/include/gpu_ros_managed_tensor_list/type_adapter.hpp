// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright 2026 gpu_ros_managed contributors
// Licensed under the Apache License, Version 2.0.
#ifndef GPU_ROS_MANAGED_TENSOR_LIST__TYPE_ADAPTER_HPP_
#define GPU_ROS_MANAGED_TENSOR_LIST__TYPE_ADAPTER_HPP_

#include <algorithm>
#include <memory>
#include <vector>

#include "rclcpp/type_adapter.hpp"
#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"
#include "gpu_ros_managed_core/detail/backend_ops.hpp"
#include "gpu_ros_managed_tensor_list/tensor_list.hpp"

template<>
struct rclcpp::TypeAdapter<
  gpu_ros_managed::ManagedTensorList,
  isaac_ros_tensor_list_interfaces::msg::TensorList>
{
  using is_specialized = std::true_type;
  using custom_type = gpu_ros_managed::ManagedTensorList;
  using ros_message_type = isaac_ros_tensor_list_interfaces::msg::TensorList;

  static void convert_to_ros_message(const custom_type & source, ros_message_type & destination)
  {
    destination.header = source.header();
    destination.tensors.clear();
    destination.tensors.reserve(source.tensors().size());
    for (const auto & tensor : source.tensors()) {
      auto & output = destination.tensors.emplace_back();
      output.name = tensor.name();
      output.data_type = static_cast<int32_t>(tensor.data_type());
      output.shape.rank = static_cast<uint8_t>(tensor.shape().size());
      output.shape.dims.assign(tensor.shape().begin(), tensor.shape().end());
      output.strides = tensor.strides();
      output.data.resize(tensor.byte_size());
      if (const auto * host = std::get_if<gpu_ros_managed::HostBuffer>(&tensor.storage())) {
        if (tensor.byte_size() != 0) {
          std::copy_n(host->data(), tensor.byte_size(), output.data.data());
        }
      } else {
        const auto & device = std::get<std::shared_ptr<gpu_ros_managed::DeviceBuffer>>(
          tensor.storage());
        gpu_ros_managed::detail::DeviceBufferFactory::copy_to_host_blocking(
          *device, output.data.data(), output.data.size());
      }
    }
  }

  static void convert_to_custom(const ros_message_type & source, custom_type & destination)
  {
    std::vector<gpu_ros_managed::ManagedTensor> tensors;
    tensors.reserve(source.tensors.size());
    for (const auto & tensor : source.tensors) {
      std::vector<int64_t> shape(tensor.shape.dims.begin(), tensor.shape.dims.end());
      if (tensor.shape.rank != shape.size()) {
        throw std::invalid_argument("ROS Tensor rank does not match dims");
      }
      tensors.push_back(gpu_ros_managed::ManagedTensor::from_host_copy(
          tensor.name, static_cast<gpu_ros_managed::TensorDataType>(tensor.data_type),
          std::move(shape), tensor.data.data(), tensor.data.size()));
      if (!tensor.strides.empty() && tensor.strides != tensors.back().strides()) {
        throw std::invalid_argument("ROS Tensor is non-contiguous");
      }
    }
    destination = custom_type(source.header, std::move(tensors));
  }
};

RCLCPP_USING_CUSTOM_TYPE_AS_ROS_MESSAGE_TYPE(
  gpu_ros_managed::ManagedTensorList,
  isaac_ros_tensor_list_interfaces::msg::TensorList);
#endif
