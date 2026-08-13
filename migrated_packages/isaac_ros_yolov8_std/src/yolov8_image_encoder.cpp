// Copyright 2026 Maintainer
// Licensed under the Apache License, Version 2.0.

#include "isaac_ros_yolov8_std/yolov8_image_encoder.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "isaac_ros_detection_common/image_preprocess.hpp"

namespace nvidia::isaac_ros::yolov8_std
{
namespace
{
constexpr int kGxfFloat32 = 9;
constexpr int kChannels = 3;
}

isaac_ros_tensor_list_interfaces::msg::TensorList EncodeYoloV8Image(
  const sensor_msgs::msg::Image & image,
  const YoloV8ImageEncoderConfig & config)
{
  if (config.tensor_name.empty()) {
    throw std::invalid_argument("YOLOv8 tensor_name must not be empty");
  }
  const auto plan = detection_common::MakeImagePreprocessPlan(
    image, config.output_width, config.output_height,
    detection_common::PreprocessNormalization::kUnitRange);
  const auto values = detection_common::ExecuteCpuPreprocess(image, plan);

  isaac_ros_tensor_list_interfaces::msg::Tensor tensor;
  tensor.name = config.tensor_name;
  tensor.data_type = kGxfFloat32;
  tensor.shape.rank = 4;
  tensor.shape.dims = {
    1U, static_cast<uint32_t>(kChannels),
    static_cast<uint32_t>(config.output_height),
    static_cast<uint32_t>(config.output_width)};
  tensor.data.resize(values.size() * sizeof(float));
  std::memcpy(tensor.data.data(), values.data(), tensor.data.size());

  isaac_ros_tensor_list_interfaces::msg::TensorList output;
  output.header = image.header;
  output.tensors.push_back(std::move(tensor));
  return output;
}

}  // namespace nvidia::isaac_ros::yolov8_std
