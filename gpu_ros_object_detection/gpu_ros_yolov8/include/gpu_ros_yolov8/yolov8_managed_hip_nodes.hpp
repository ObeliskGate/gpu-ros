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

#ifndef GPU_ROS_YOLOV8__YOLOV8_MANAGED_HIP_NODES_HPP_
#define GPU_ROS_YOLOV8__YOLOV8_MANAGED_HIP_NODES_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gpu_ros_managed_core/fixed_device_memory_pool.hpp"
#include "gpu_ros_managed_hip/hip_backend.hpp"
#include "gpu_ros_managed_ros/managed_pub_sub.hpp"
#include "gpu_ros_managed_tensor_bundle/tensor_bundle.hpp"
#include "gpu_ros_managed_tensor_bundle/type_adapter.hpp"
#include "gpu_ros_yolov8/yolov8_decoder.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"

namespace gpu_ros::yolov8
{

class YoloV8ManagedHipImageEncoderNode final : public rclcpp::Node
{
public:
  explicit YoloV8ManagedHipImageEncoderNode(
    const rclcpp::NodeOptions options = rclcpp::NodeOptions());
  ~YoloV8ManagedHipImageEncoderNode() override;

private:
  void InputCallback(sensor_msgs::msg::Image::ConstSharedPtr message);
  using TensorBundle = gpu_ros_managed::ManagedTensorBundle;

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
  gpu_ros_managed::ManagedPublisher<TensorBundle> publisher_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  size_t pool_exhaustion_drops_{0};
};

class YoloV8ManagedHipDecoderNode final : public rclcpp::Node
{
public:
  explicit YoloV8ManagedHipDecoderNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());

private:
  void InputCallback(gpu_ros_managed::ManagedTensorBundleView message);
  gpu_ros_managed::hip::HipStream read_stream_;
  YoloV8DecoderConfig config_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr publisher_;
  std::unique_ptr<gpu_ros_managed::ManagedSubscriber<gpu_ros_managed::ManagedTensorBundleView>>
    subscriber_;
};

} // namespace gpu_ros::yolov8

#endif // GPU_ROS_YOLOV8__YOLOV8_MANAGED_HIP_NODES_HPP_
