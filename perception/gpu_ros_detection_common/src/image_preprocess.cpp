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

#include "gpu_ros_detection_common/image_preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "sensor_msgs/image_encodings.hpp"

namespace gpu_ros::detection_common
{
namespace
{

ImageEncoding ParseEncoding(const std::string & encoding)
{
  namespace enc = sensor_msgs::image_encodings;
  if (encoding == enc::RGB8) {return ImageEncoding::kRgb8;}
  if (encoding == enc::BGR8) {return ImageEncoding::kBgr8;}
  if (encoding == enc::RGBA8) {return ImageEncoding::kRgba8;}
  if (encoding == enc::BGRA8) {return ImageEncoding::kBgra8;}
  if (encoding == enc::MONO8) {return ImageEncoding::kMono8;}
  throw std::invalid_argument("unsupported image encoding: " + encoding);
}

uint32_t ChannelsFor(ImageEncoding encoding)
{
  return encoding == ImageEncoding::kMono8 ? 1U :
    (encoding == ImageEncoding::kRgb8 || encoding == ImageEncoding::kBgr8 ? 3U : 4U);
}

void ValidateImageStorage(const sensor_msgs::msg::Image & image, size_t bytes_per_pixel)
{
  if (image.height == 0 || image.width == 0 || image.data.empty()) {
    throw std::invalid_argument("image is empty");
  }
  if (image.height > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
    image.width > static_cast<uint32_t>(std::numeric_limits<int>::max()))
  {
    throw std::invalid_argument("image dimensions do not fit in OpenCV int dimensions");
  }
  if (bytes_per_pixel != 0U && static_cast<size_t>(image.width) >
    std::numeric_limits<size_t>::max() / bytes_per_pixel)
  {
    throw std::overflow_error("image width/bytes-per-pixel overflows size_t");
  }
  const size_t min_step = static_cast<size_t>(image.width) * bytes_per_pixel;
  if (image.step < min_step) {
    throw std::invalid_argument("image step is smaller than width * bytes_per_pixel");
  }
  const size_t rows_before_last = static_cast<size_t>(image.height) - 1U;
  if (rows_before_last >
    (std::numeric_limits<size_t>::max() - min_step) / static_cast<size_t>(image.step))
  {
    throw std::overflow_error("image height/step metadata overflows the buffer size");
  }
  const size_t required = static_cast<size_t>(image.step) * rows_before_last + min_step;
  if (image.data.size() < required) {
    throw std::invalid_argument("image data is smaller than height/step metadata requires");
  }
}

cv::Mat ToRgb(const sensor_msgs::msg::Image & image, ImageEncoding encoding)
{
  const int height = static_cast<int>(image.height);
  const int width = static_cast<int>(image.width);
  cv::Mat source;
  cv::Mat rgb;
  switch (encoding) {
    case ImageEncoding::kRgb8:
      ValidateImageStorage(image, 3U);
      return cv::Mat(
        height, width, CV_8UC3, const_cast<uint8_t *>(image.data.data()), image.step);
    case ImageEncoding::kBgr8:
      ValidateImageStorage(image, 3U);
      source = cv::Mat(
        height, width, CV_8UC3, const_cast<uint8_t *>(image.data.data()), image.step);
      cv::cvtColor(source, rgb, cv::COLOR_BGR2RGB);
      break;
    case ImageEncoding::kRgba8:
      ValidateImageStorage(image, 4U);
      source = cv::Mat(
        height, width, CV_8UC4, const_cast<uint8_t *>(image.data.data()), image.step);
      cv::cvtColor(source, rgb, cv::COLOR_RGBA2RGB);
      break;
    case ImageEncoding::kBgra8:
      ValidateImageStorage(image, 4U);
      source = cv::Mat(
        height, width, CV_8UC4, const_cast<uint8_t *>(image.data.data()), image.step);
      cv::cvtColor(source, rgb, cv::COLOR_BGRA2RGB);
      break;
    case ImageEncoding::kMono8:
      ValidateImageStorage(image, 1U);
      source = cv::Mat(
        height, width, CV_8UC1, const_cast<uint8_t *>(image.data.data()), image.step);
      cv::cvtColor(source, rgb, cv::COLOR_GRAY2RGB);
      break;
  }
  return rgb;
}

}  // namespace

const char * ImageEncodingName(ImageEncoding encoding) noexcept
{
  switch (encoding) {
    case ImageEncoding::kRgb8: return "rgb8";
    case ImageEncoding::kBgr8: return "bgr8";
    case ImageEncoding::kRgba8: return "rgba8";
    case ImageEncoding::kBgra8: return "bgra8";
    case ImageEncoding::kMono8: return "mono8";
  }
  return "unknown";
}

size_t SourceRowBytes(const ImagePreprocessPlan & plan)
{
  if (plan.input_channels != 0U && static_cast<size_t>(plan.input_width) >
    std::numeric_limits<size_t>::max() / plan.input_channels)
  {
    throw std::overflow_error("preprocess source row bytes overflow size_t");
  }
  return static_cast<size_t>(plan.input_width) * plan.input_channels;
}

ImagePreprocessPlan MakeImagePreprocessPlan(
  const sensor_msgs::msg::Image & image,
  int64_t output_width,
  int64_t output_height,
  PreprocessNormalization normalization)
{
  if (output_width <= 0 || output_height <= 0 ||
    output_width > std::numeric_limits<int>::max() ||
    output_height > std::numeric_limits<int>::max())
  {
    throw std::invalid_argument("preprocess output dimensions must be positive and fit in int");
  }
  const auto encoding = ParseEncoding(image.encoding);
  const uint32_t channels = ChannelsFor(encoding);
  ValidateImageStorage(image, channels);
  const double scale = std::min(
    static_cast<double>(output_width) / static_cast<double>(image.width),
    static_cast<double>(output_height) / static_cast<double>(image.height));
  const double rounded_width = std::round(static_cast<double>(image.width) * scale);
  const double rounded_height = std::round(static_cast<double>(image.height) * scale);
  const int resized_width = static_cast<int>(std::clamp(
      rounded_width, 1.0, static_cast<double>(output_width)));
  const int resized_height = static_cast<int>(std::clamp(
      rounded_height, 1.0, static_cast<double>(output_height)));
  return ImagePreprocessPlan{
    encoding, image.width, image.height, image.step, channels,
    static_cast<int>(output_width), static_cast<int>(output_height),
    resized_width, resized_height, normalization};
}

std::vector<float> ExecuteCpuPreprocess(
  const sensor_msgs::msg::Image & image,
  const ImagePreprocessPlan & plan)
{
  if (image.width != plan.input_width || image.height != plan.input_height ||
    image.step != plan.input_step || ParseEncoding(image.encoding) != plan.encoding)
  {
    throw std::invalid_argument("image no longer matches its ImagePreprocessPlan");
  }
  const cv::Mat rgb = ToRgb(image, plan.encoding);
  cv::Mat resized;
  cv::resize(
    rgb, resized, cv::Size(plan.resized_width, plan.resized_height),
    0.0, 0.0, cv::INTER_LINEAR);
  cv::Mat padded(plan.output_height, plan.output_width, CV_8UC3, cv::Scalar(0, 0, 0));
  resized.copyTo(padded(cv::Rect(0, 0, plan.resized_width, plan.resized_height)));

  if (static_cast<size_t>(plan.output_width) >
    std::numeric_limits<size_t>::max() / static_cast<size_t>(plan.output_height))
  {
    throw std::overflow_error("preprocess output plane overflows size_t");
  }
  const size_t plane = static_cast<size_t>(plan.output_width) * plan.output_height;
  if (plane > std::numeric_limits<size_t>::max() / 3U) {
    throw std::overflow_error("preprocess output element count overflows size_t");
  }
  const size_t element_count = 3U * plane;
  if (element_count > std::numeric_limits<size_t>::max() / sizeof(float)) {
    throw std::overflow_error("preprocess output byte size overflows size_t");
  }
  std::vector<float> output(element_count, 0.0F);
  const float scale = plan.normalization == PreprocessNormalization::kUnitRange ?
    1.0F / 255.0F : 1.0F;
  for (int channel = 0; channel < 3; ++channel) {
    for (int y = 0; y < plan.output_height; ++y) {
      const auto * row = padded.ptr<cv::Vec3b>(y);
      for (int x = 0; x < plan.output_width; ++x) {
        const size_t index = static_cast<size_t>(channel) * plane +
          static_cast<size_t>(y) * plan.output_width + x;
        output[index] = static_cast<float>(row[x][channel]) * scale;
      }
    }
  }
  return output;
}

}  // namespace gpu_ros::detection_common
