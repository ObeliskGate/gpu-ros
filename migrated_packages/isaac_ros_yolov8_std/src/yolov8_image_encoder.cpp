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

#include "isaac_ros_yolov8_std/yolov8_image_encoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "sensor_msgs/image_encodings.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace yolov8_std
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

  const size_t row_count = static_cast<size_t>(msg.height) - 1U;
  if (
    static_cast<size_t>(msg.step) != 0U &&
    row_count > (std::numeric_limits<size_t>::max() - min_step) /
    static_cast<size_t>(msg.step))
  {
    throw std::runtime_error("image height/step metadata overflows the buffer size calculation");
  }
  const size_t min_data_size =
    static_cast<size_t>(msg.step) * row_count + min_step;
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
  if (msg.height > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
    msg.width > static_cast<uint32_t>(std::numeric_limits<int>::max()))
  {
    throw std::runtime_error("image dimensions do not fit in OpenCV's int dimensions");
  }

  const int height = static_cast<int>(msg.height);
  const int width = static_cast<int>(msg.width);
  if (msg.encoding == enc::RGB8) {
    ValidateImageStorage(msg, 3U);
    return cv::Mat(
      height, width, CV_8UC3,
      const_cast<uint8_t *>(msg.data.data()), msg.step);
  }

  cv::Mat src;
  cv::Mat rgb;
  if (msg.encoding == enc::BGR8) {
    ValidateImageStorage(msg, 3U);
    src = cv::Mat(
      height, width, CV_8UC3,
      const_cast<uint8_t *>(msg.data.data()), msg.step);
    cv::cvtColor(src, rgb, cv::COLOR_BGR2RGB);
  } else if (msg.encoding == enc::RGBA8) {
    ValidateImageStorage(msg, 4U);
    src = cv::Mat(
      height, width, CV_8UC4,
      const_cast<uint8_t *>(msg.data.data()), msg.step);
    cv::cvtColor(src, rgb, cv::COLOR_RGBA2RGB);
  } else if (msg.encoding == enc::BGRA8) {
    ValidateImageStorage(msg, 4U);
    src = cv::Mat(
      height, width, CV_8UC4,
      const_cast<uint8_t *>(msg.data.data()), msg.step);
    cv::cvtColor(src, rgb, cv::COLOR_BGRA2RGB);
  } else if (msg.encoding == enc::MONO8) {
    ValidateImageStorage(msg, 1U);
    src = cv::Mat(
      height, width, CV_8UC1,
      const_cast<uint8_t *>(msg.data.data()), msg.step);
    cv::cvtColor(src, rgb, cv::COLOR_GRAY2RGB);
  } else {
    throw std::runtime_error("unsupported image encoding: " + msg.encoding);
  }
  return rgb;
}

cv::Mat ResizeAndPad(
  const cv::Mat & rgb,
  int output_width,
  int output_height)
{
  const double scale = std::min(
    static_cast<double>(output_width) / static_cast<double>(rgb.cols),
    static_cast<double>(output_height) / static_cast<double>(rgb.rows));
  const int resized_width = std::clamp(
    static_cast<int>(std::round(rgb.cols * scale)), 1, output_width);
  const int resized_height = std::clamp(
    static_cast<int>(std::round(rgb.rows * scale)), 1, output_height);

  cv::Mat resized;
  cv::resize(
    rgb, resized, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_LINEAR);

  cv::Mat padded(output_height, output_width, CV_8UC3, cv::Scalar(0, 0, 0));
  resized.copyTo(padded(cv::Rect(0, 0, resized_width, resized_height)));
  return padded;
}

std::vector<uint8_t> HwcRgbU8ToNchwF32Bytes(const cv::Mat & padded)
{
  const int height = padded.rows;
  const int width = padded.cols;
  const size_t element_count =
    static_cast<size_t>(kChannels) * static_cast<size_t>(height) * static_cast<size_t>(width);
  std::vector<float> values(element_count);

  for (int c = 0; c < kChannels; ++c) {
    for (int y = 0; y < height; ++y) {
      const auto * row = padded.ptr<cv::Vec3b>(y);
      for (int x = 0; x < width; ++x) {
        const size_t index =
          static_cast<size_t>(c) * static_cast<size_t>(height) * static_cast<size_t>(width) +
          static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
        values[index] = static_cast<float>(row[x][c]) / 255.0F;
      }
    }
  }

  std::vector<uint8_t> bytes(values.size() * sizeof(float));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

}  // namespace

isaac_ros_tensor_list_interfaces::msg::TensorList EncodeYoloV8Image(
  const sensor_msgs::msg::Image & image,
  const YoloV8ImageEncoderConfig & config)
{
  if (config.tensor_name.empty()) {
    throw std::invalid_argument("YOLOv8 tensor_name must not be empty");
  }
  if (
    config.output_width <= 0 || config.output_height <= 0 ||
    config.output_width > std::numeric_limits<int>::max() ||
    config.output_height > std::numeric_limits<int>::max())
  {
    throw std::invalid_argument(
            "YOLOv8 output_width and output_height must be positive and fit in an int");
  }

  const cv::Mat rgb = ImageToRgbMat(image);
  const cv::Mat padded = ResizeAndPad(
    rgb, static_cast<int>(config.output_width), static_cast<int>(config.output_height));

  isaac_ros_tensor_list_interfaces::msg::Tensor tensor;
  tensor.name = config.tensor_name;
  tensor.data_type = kGxfFloat32;
  tensor.shape.rank = 4;
  tensor.shape.dims = {
    1U,
    static_cast<uint32_t>(kChannels),
    static_cast<uint32_t>(config.output_height),
    static_cast<uint32_t>(config.output_width)};
  tensor.data = HwcRgbU8ToNchwF32Bytes(padded);

  isaac_ros_tensor_list_interfaces::msg::TensorList output;
  output.header = image.header;
  output.tensors.push_back(std::move(tensor));
  return output;
}

}  // namespace yolov8_std
}  // namespace isaac_ros
}  // namespace nvidia
