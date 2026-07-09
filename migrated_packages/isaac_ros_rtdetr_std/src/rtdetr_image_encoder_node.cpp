// Copyright 2026 Maintainer
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

#include "isaac_ros_rtdetr_std/rtdetr_image_encoder_node.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/image_encodings.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace rtdetr_std
{

namespace
{

constexpr int kGxfFloat32 = 9;
constexpr int kChannels = 3;

void ValidateImageStorage(
  const sensor_msgs::msg::Image & msg,
  size_t bytes_per_pixel)
{
  const size_t min_step = static_cast<size_t>(msg.width) * bytes_per_pixel;
  if (msg.step < min_step) {
    throw std::runtime_error("image step is smaller than width * bytes_per_pixel");
  }
  const size_t min_data_size =
    static_cast<size_t>(msg.step) * (static_cast<size_t>(msg.height) - 1U) + min_step;
  if (msg.data.size() < min_data_size) {
    throw std::runtime_error("image data buffer is smaller than height/step metadata requires");
  }
}

cv::Mat ImageToRgbMat(const sensor_msgs::msg::Image & msg)
{
  namespace enc = sensor_msgs::image_encodings;

  if (msg.height == 0 || msg.width == 0 || msg.data.empty()) {
    throw std::runtime_error("empty image");
  }

  if (msg.encoding == enc::RGB8) {
    ValidateImageStorage(msg, 3U);
    return cv::Mat(
      static_cast<int>(msg.height), static_cast<int>(msg.width), CV_8UC3,
      const_cast<uint8_t *>(msg.data.data()), msg.step);
  }

  cv::Mat src;
  cv::Mat rgb;
  if (msg.encoding == enc::BGR8) {
    ValidateImageStorage(msg, 3U);
    src = cv::Mat(
      static_cast<int>(msg.height), static_cast<int>(msg.width), CV_8UC3,
      const_cast<uint8_t *>(msg.data.data()), msg.step);
    cv::cvtColor(src, rgb, cv::COLOR_BGR2RGB);
  } else if (msg.encoding == enc::RGBA8) {
    ValidateImageStorage(msg, 4U);
    src = cv::Mat(
      static_cast<int>(msg.height), static_cast<int>(msg.width), CV_8UC4,
      const_cast<uint8_t *>(msg.data.data()), msg.step);
    cv::cvtColor(src, rgb, cv::COLOR_RGBA2RGB);
  } else if (msg.encoding == enc::BGRA8) {
    ValidateImageStorage(msg, 4U);
    src = cv::Mat(
      static_cast<int>(msg.height), static_cast<int>(msg.width), CV_8UC4,
      const_cast<uint8_t *>(msg.data.data()), msg.step);
    cv::cvtColor(src, rgb, cv::COLOR_BGRA2RGB);
  } else if (msg.encoding == enc::MONO8) {
    ValidateImageStorage(msg, 1U);
    src = cv::Mat(
      static_cast<int>(msg.height), static_cast<int>(msg.width), CV_8UC1,
      const_cast<uint8_t *>(msg.data.data()), msg.step);
    cv::cvtColor(src, rgb, cv::COLOR_GRAY2RGB);
  } else {
    throw std::runtime_error("unsupported image encoding: " + msg.encoding);
  }
  return rgb;
}

cv::Mat ResizeAndPad(const cv::Mat & rgb, int output_width, int output_height)
{
  const double scale = std::min(
    static_cast<double>(output_width) / static_cast<double>(rgb.cols),
    static_cast<double>(output_height) / static_cast<double>(rgb.rows));
  const int resized_width = std::clamp(
    static_cast<int>(std::round(rgb.cols * scale)), 1, output_width);
  const int resized_height = std::clamp(
    static_cast<int>(std::round(rgb.rows * scale)), 1, output_height);

  cv::Mat resized;
  cv::resize(rgb, resized, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_LINEAR);

  cv::Mat padded(output_height, output_width, CV_8UC3, cv::Scalar(0, 0, 0));
  resized.copyTo(padded(cv::Rect(0, 0, resized_width, resized_height)));
  return padded;
}

std::vector<uint8_t> HwcRgbU8ToNchwF32Bytes(const cv::Mat & padded)
{
  const int height = padded.rows;
  const int width = padded.cols;
  std::vector<uint8_t> bytes(
    sizeof(float) * static_cast<size_t>(kChannels) * height * width);
  auto * out = reinterpret_cast<float *>(bytes.data());

  for (int c = 0; c < kChannels; ++c) {
    for (int y = 0; y < height; ++y) {
      const auto * row = padded.ptr<cv::Vec3b>(y);
      for (int x = 0; x < width; ++x) {
        const size_t index =
          static_cast<size_t>(c) * height * width +
          static_cast<size_t>(y) * width +
          static_cast<size_t>(x);
        out[index] = static_cast<float>(row[x][c]);
      }
    }
  }

  return bytes;
}

}  // namespace

RtDetrImageEncoderNode::RtDetrImageEncoderNode(const rclcpp::NodeOptions options)
: rclcpp::Node("rtdetr_image_encoder_node", options),
  tensor_name_{declare_parameter<std::string>("tensor_name", "input_tensor")},
  output_height_{declare_parameter<int64_t>("output_height", 640)},
  output_width_{declare_parameter<int64_t>("output_width", 640)}
{
  if (output_height_ <= 0 || output_width_ <= 0) {
    throw std::invalid_argument("output_height and output_width must be positive");
  }

  pub_ = create_publisher<TensorList>("encoded_tensor", 10);
  sub_ = create_subscription<Image>(
    "image", 10,
    std::bind(&RtDetrImageEncoderNode::InputCallback, this, std::placeholders::_1));
}

void RtDetrImageEncoderNode::InputCallback(const Image::ConstSharedPtr msg)
{
  try {
    const cv::Mat rgb = ImageToRgbMat(*msg);
    const cv::Mat padded = ResizeAndPad(
      rgb, static_cast<int>(output_width_), static_cast<int>(output_height_));

    isaac_ros_tensor_list_interfaces::msg::Tensor tensor;
    tensor.name = tensor_name_;
    tensor.data_type = kGxfFloat32;
    tensor.shape.rank = 4;
    tensor.shape.dims = {1, kChannels, output_height_, output_width_};
    tensor.data = HwcRgbU8ToNchwF32Bytes(padded);

    TensorList out_msg;
    out_msg.header = msg->header;
    out_msg.tensors.push_back(std::move(tensor));
    pub_->publish(out_msg);
  } catch (const std::exception & e) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "RT-DETR image encoder dropped frame: %s", e.what());
  }
}

}  // namespace rtdetr_std
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::rtdetr_std::RtDetrImageEncoderNode)
