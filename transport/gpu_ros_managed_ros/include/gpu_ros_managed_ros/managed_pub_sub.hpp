// Modified from NVIDIA Isaac ROS NITROS managed publisher/subscriber sources for the managed backend-neutral API.

// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright 2026 gpu_ros_managed contributors
// Licensed under the Apache License, Version 2.0.
#ifndef GPU_ROS_MANAGED_ROS__MANAGED_PUB_SUB_HPP_
#define GPU_ROS_MANAGED_ROS__MANAGED_PUB_SUB_HPP_

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include "rclcpp/rclcpp.hpp"

namespace gpu_ros_managed
{
template<typename MessageT>
class ManagedPublisher
{
public:
  ManagedPublisher(
    rclcpp::Node * node, const std::string & topic,
    const rclcpp::QoS & qos = rclcpp::QoS(1))
  {
    rclcpp::PublisherOptions options;
    options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
    publisher_ = node->create_publisher<MessageT>(topic, qos, options);
  }
  void publish(MessageT && message) {publisher_->publish(std::move(message));}
  void publish(std::unique_ptr<MessageT> message) {publisher_->publish(std::move(message));}

private:
  typename rclcpp::Publisher<MessageT>::SharedPtr publisher_;
};

template<typename ViewT>
class ManagedSubscriber
{
public:
  using MessageT = typename ViewT::MessageType;
  using Callback = std::function<void(ViewT)>;
  ManagedSubscriber(
    rclcpp::Node * node, const std::string & topic, Callback callback,
    const rclcpp::QoS & qos = rclcpp::QoS(1))
  {
    rclcpp::SubscriptionOptions options;
    options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
    subscription_ = node->create_subscription<MessageT>(
      topic, qos,
      [callback = std::move(callback)](std::shared_ptr<const MessageT> message) mutable {
        callback(ViewT(std::move(message)));
      }, options);
  }

private:
  typename rclcpp::Subscription<MessageT>::SharedPtr subscription_;
};
}  // namespace gpu_ros_managed
#endif
