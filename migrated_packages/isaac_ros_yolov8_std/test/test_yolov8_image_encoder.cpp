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

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

#include "isaac_ros_yolov8_std/yolov8_image_encoder.hpp"
#include "sensor_msgs/image_encodings.hpp"

namespace
{

using nvidia::isaac_ros::yolov8_std::EncodeYoloV8Image;
using nvidia::isaac_ros::yolov8_std::YoloV8ImageEncoderConfig;

sensor_msgs::msg::Image MakeImage(
  const std::string & encoding,
  uint32_t width,
  uint32_t height,
  uint32_t step,
  const std::vector<uint8_t> & data)
{
  sensor_msgs::msg::Image image;
  image.header.frame_id = "camera";
  image.header.stamp.sec = 12;
  image.header.stamp.nanosec = 34;
  image.encoding = encoding;
  image.width = width;
  image.height = height;
  image.step = step;
  image.data = data;
  return image;
}

std::vector<float> TensorValues(
  const isaac_ros_tensor_list_interfaces::msg::Tensor & tensor)
{
  std::vector<float> values(tensor.data.size() / sizeof(float));
  std::memcpy(values.data(), tensor.data.data(), tensor.data.size());
  return values;
}

YoloV8ImageEncoderConfig Config(int64_t width = 2, int64_t height = 2)
{
  YoloV8ImageEncoderConfig config;
  config.tensor_name = "images";
  config.output_width = width;
  config.output_height = height;
  return config;
}

}  // namespace

TEST(YoloV8ImageEncoderTest, ConvertsRgbToNormalizedNchwAndPreservesHeader)
{
  const auto image = MakeImage(
    sensor_msgs::image_encodings::RGB8, 1, 1, 3, {10U, 20U, 30U});

  const auto output = EncodeYoloV8Image(image, Config());

  ASSERT_EQ(output.header.frame_id, "camera");
  EXPECT_EQ(output.header.stamp.sec, 12);
  ASSERT_EQ(output.tensors.size(), 1U);
  const auto & tensor = output.tensors.front();
  EXPECT_EQ(tensor.name, "images");
  EXPECT_EQ(tensor.data_type, 9);
  EXPECT_EQ(tensor.shape.rank, 4);
  EXPECT_EQ(tensor.shape.dims, (std::vector<uint32_t>{1U, 3U, 2U, 2U}));

  const auto values = TensorValues(tensor);
  ASSERT_EQ(values.size(), 12U);
  EXPECT_FLOAT_EQ(values.at(0), 10.0F / 255.0F);
  EXPECT_FLOAT_EQ(values.at(4), 20.0F / 255.0F);
  EXPECT_FLOAT_EQ(values.at(8), 30.0F / 255.0F);
  EXPECT_FLOAT_EQ(values.at(1), values.at(0));
  EXPECT_FLOAT_EQ(values.at(5), values.at(4));
  EXPECT_FLOAT_EQ(values.at(9), values.at(8));
}

TEST(YoloV8ImageEncoderTest, ConvertsBgrToRgb)
{
  const auto image = MakeImage(
    sensor_msgs::image_encodings::BGR8, 1, 1, 3, {1U, 2U, 3U});

  const auto output = EncodeYoloV8Image(image, Config(1, 1));
  const auto values = TensorValues(output.tensors.front());

  ASSERT_EQ(values.size(), 3U);
  EXPECT_FLOAT_EQ(values.at(0), 3.0F / 255.0F);
  EXPECT_FLOAT_EQ(values.at(1), 2.0F / 255.0F);
  EXPECT_FLOAT_EQ(values.at(2), 1.0F / 255.0F);
}

TEST(YoloV8ImageEncoderTest, UsesLinearResizeAndBottomPadding)
{
  const auto image = MakeImage(
    sensor_msgs::image_encodings::RGB8, 2, 1, 6,
    {10U, 20U, 30U, 40U, 50U, 60U});

  const auto output = EncodeYoloV8Image(image, Config(4, 4));
  const auto values = TensorValues(output.tensors.front());

  ASSERT_EQ(values.size(), 48U);
  EXPECT_FLOAT_EQ(values.at(0), 10.0F / 255.0F);
  EXPECT_FLOAT_EQ(values.at(1), 17.0F / 255.0F);
  EXPECT_FLOAT_EQ(values.at(2), 32.0F / 255.0F);
  EXPECT_FLOAT_EQ(values.at(3), 40.0F / 255.0F);
  EXPECT_FLOAT_EQ(values.at(4), 10.0F / 255.0F);
  EXPECT_FLOAT_EQ(values.at(5), 17.0F / 255.0F);
  EXPECT_FLOAT_EQ(values.at(8), 0.0F);
  EXPECT_FLOAT_EQ(values.at(12), 0.0F);
}

TEST(YoloV8ImageEncoderTest, RejectsInvalidImageStorage)
{
  const auto image = MakeImage(
    sensor_msgs::image_encodings::RGB8, 2, 1, 5, {1U, 2U, 3U, 4U, 5U});

  EXPECT_THROW(EncodeYoloV8Image(image, Config()), std::invalid_argument);
}
