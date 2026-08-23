// Copyright 2026 Boshen Chen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef GPU_ROS_RTDETR__RTDETR_MANAGED_HIP_NODES_HPP_
#define GPU_ROS_RTDETR__RTDETR_MANAGED_HIP_NODES_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "gpu_ros_managed_core/fixed_device_memory_pool.hpp"
#include "gpu_ros_managed_hip/hip_backend.hpp"
#include "gpu_ros_managed_ros/managed_pub_sub.hpp"
#include "gpu_ros_managed_tensor_bundle/tensor_bundle.hpp"
#include "gpu_ros_managed_tensor_bundle/type_adapter.hpp"
#include "gpu_ros_rtdetr/rtdetr_decoder.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"

namespace gpu_ros::rtdetr
{

class RtDetrManagedHipImageEncoderNode final : public rclcpp::Node
{
public:
  explicit RtDetrManagedHipImageEncoderNode(
    const rclcpp::NodeOptions options = rclcpp::NodeOptions());
  ~RtDetrManagedHipImageEncoderNode() override;

private:
  void InputCallback(sensor_msgs::msg::Image::ConstSharedPtr message);
  std::string tensor_name_;
  int64_t output_width_{640};
  int64_t output_height_{640};
  int gpu_device_id_{0};
  size_t pool_capacity_{16};
  std::chrono::milliseconds pool_timeout_{100};
  size_t max_input_bytes_{16U * 1024U * 1024U};
  std::unique_ptr<gpu_ros_managed::hip::HipStream> stream_;
  std::unique_ptr<gpu_ros_managed::FixedDeviceMemoryPool> raw_pool_;
  std::unique_ptr<gpu_ros_managed::FixedDeviceMemoryPool> output_pool_;
  gpu_ros_managed::ManagedPublisher<gpu_ros_managed::ManagedTensorBundle> publisher_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  size_t pool_exhaustion_drops_{0};
};

class RtDetrManagedHipPreprocessorNode final : public rclcpp::Node
{
public:
  explicit RtDetrManagedHipPreprocessorNode(
    const rclcpp::NodeOptions options = rclcpp::NodeOptions());
  ~RtDetrManagedHipPreprocessorNode() override;

private:
  void InputCallback(gpu_ros_managed::ManagedTensorBundleView message);
  std::string input_image_tensor_name_;
  std::string output_image_tensor_name_;
  std::string output_size_tensor_name_;
  int64_t image_width_{640};
  int64_t image_height_{480};
  int64_t model_input_width_{640};
  int64_t model_input_height_{640};
  bool use_max_dim_for_orig_size_{true};
  int gpu_device_id_{0};
  size_t pool_capacity_{16};
  std::chrono::milliseconds pool_timeout_{100};
  gpu_ros_managed::hip::HipStream stream_;
  std::unique_ptr<gpu_ros_managed::FixedDeviceMemoryPool> size_pool_;
  gpu_ros_managed::ManagedPublisher<gpu_ros_managed::ManagedTensorBundle> publisher_;
  std::unique_ptr<gpu_ros_managed::ManagedSubscriber<
      gpu_ros_managed::ManagedTensorBundleView>> subscriber_;
  size_t pool_exhaustion_drops_{0};
};

class RtDetrManagedHipDecoderNode final : public rclcpp::Node
{
public:
  explicit RtDetrManagedHipDecoderNode(
    const rclcpp::NodeOptions options = rclcpp::NodeOptions());

private:
  void InputCallback(gpu_ros_managed::ManagedTensorBundleView message);
  gpu_ros_managed::hip::HipStream read_stream_;
  RtDetrDecoderConfig config_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr publisher_;
  std::unique_ptr<gpu_ros_managed::ManagedSubscriber<
      gpu_ros_managed::ManagedTensorBundleView>> subscriber_;
};

}  // namespace gpu_ros::rtdetr

#endif  // GPU_ROS_RTDETR__RTDETR_MANAGED_HIP_NODES_HPP_
