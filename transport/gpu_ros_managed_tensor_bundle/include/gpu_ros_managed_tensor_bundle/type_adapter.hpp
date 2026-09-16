// SPDX-FileCopyrightText: Boshen Chen
// Copyright 2026 Boshen Chen
// Licensed under the Apache License, Version 2.0.
#ifndef GPU_ROS_MANAGED_TENSOR_BUNDLE__TYPE_ADAPTER_HPP_
#define GPU_ROS_MANAGED_TENSOR_BUNDLE__TYPE_ADAPTER_HPP_

#include <algorithm>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include "rclcpp/type_adapter.hpp"
#include "gpu_ros_tensor_bundle_msgs/msg/tensor.hpp"
#include "gpu_ros_tensor_bundle_msgs/msg/tensor_bundle.hpp"
#include "gpu_ros_managed_tensor_bundle/tensor_bundle.hpp"

template <>
struct rclcpp::TypeAdapter<gpu_ros_managed::ManagedTensorBundle,
  gpu_ros_tensor_bundle_msgs::msg::TensorBundle>
{
  using is_specialized = std::true_type;
  using custom_type = gpu_ros_managed::ManagedTensorBundle;
  using ros_message_type = gpu_ros_tensor_bundle_msgs::msg::TensorBundle;

  static void convert_to_ros_message(const custom_type & source, ros_message_type & destination)
  {
    destination.header = source.header();
    destination.tensors.clear();
    destination.tensors.reserve(source.tensors().size());
    for (const auto & tensor : source.tensors()) {
      auto & output = destination.tensors.emplace_back();
      output.name = tensor.name();
      output.data_type = static_cast<uint8_t>(tensor.data_type());
      output.shape.assign(tensor.shape().begin(), tensor.shape().end());
      output.data.resize(tensor.byte_size());
      if (const auto * host = std::get_if<gpu_ros_managed::HostBuffer>(&tensor.storage())) {
        if (tensor.byte_size() != 0) {
          std::copy_n(host->data(), tensor.byte_size(), output.data.data());
        }
      } else {
        const auto & device =
          std::get<std::shared_ptr<gpu_ros_managed::DeviceBuffer>>(tensor.storage());
        device->copy_to_host_blocking(output.data.data(), output.data.size());
      }
    }
  }

  static void convert_to_custom(const ros_message_type & source, custom_type & destination)
  {
    std::vector<gpu_ros_managed::ManagedTensor> tensors;
    tensors.reserve(source.tensors.size());
    for (const auto & tensor : source.tensors) {
      std::vector<int64_t> shape(tensor.shape.begin(), tensor.shape.end());
      tensors.push_back(gpu_ros_managed::ManagedTensor::from_host_copy(tensor.name,
        static_cast<gpu_ros_managed::TensorDataType>(tensor.data_type), std::move(shape),
        tensor.data.data(), tensor.data.size()));
    }
    destination = custom_type(source.header, std::move(tensors));
  }
};

RCLCPP_USING_CUSTOM_TYPE_AS_ROS_MESSAGE_TYPE(
  gpu_ros_managed::ManagedTensorBundle, gpu_ros_tensor_bundle_msgs::msg::TensorBundle);
#endif
