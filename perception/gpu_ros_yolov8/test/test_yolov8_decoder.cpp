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

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_ros_yolov8/yolov8_decoder.hpp"

namespace
{

using gpu_ros_tensor_bundle_msgs::msg::Tensor;
using gpu_ros_tensor_bundle_msgs::msg::TensorBundle;
using gpu_ros::yolov8::DecodeYoloV8TensorBundle;
using gpu_ros::yolov8::YoloV8DecoderConfig;

Tensor MakeTensor(
  const std::string & name,
  const std::vector<uint32_t> & shape,
  const std::vector<float> & values,
  uint8_t data_type = Tensor::FLOAT32)
{
  Tensor tensor;
  tensor.name = name;
  tensor.data_type = data_type;
  tensor.shape.assign(shape.begin(), shape.end());
  tensor.data.resize(values.size() * sizeof(float));
  std::memcpy(tensor.data.data(), values.data(), tensor.data.size());
  return tensor;
}

TensorBundle MakeTensorBundle(const Tensor & tensor)
{
  TensorBundle msg;
  msg.header.frame_id = "camera";
  msg.tensors.push_back(tensor);
  return msg;
}

std::vector<float> MakeYoloOutput(
  const std::vector<float> & xs,
  const std::vector<float> & ys,
  const std::vector<float> & ws,
  const std::vector<float> & hs,
  const std::vector<float> & class0,
  const std::vector<float> & class1)
{
  std::vector<float> values;
  values.insert(values.end(), xs.begin(), xs.end());
  values.insert(values.end(), ys.begin(), ys.end());
  values.insert(values.end(), ws.begin(), ws.end());
  values.insert(values.end(), hs.begin(), hs.end());
  values.insert(values.end(), class0.begin(), class0.end());
  values.insert(values.end(), class1.begin(), class1.end());
  return values;
}

YoloV8DecoderConfig Config()
{
  YoloV8DecoderConfig config;
  config.tensor_name = "output_tensor";
  config.num_classes = 2;
  config.confidence_threshold = 0.25;
  config.nms_threshold = 0.45;
  return config;
}

}  // namespace

TEST(YoloV8DecoderTest, DecodesSingleDetection)
{
  const auto values = MakeYoloOutput(
    {100.0F, 300.0F}, {120.0F, 320.0F}, {40.0F, 30.0F}, {50.0F, 20.0F},
    {0.10F, 0.20F}, {0.90F, 0.05F});
  const auto msg = MakeTensorBundle(MakeTensor("output_tensor", {1, 6, 2}, values));

  const auto detections = DecodeYoloV8TensorBundle(msg, Config());

  ASSERT_EQ(detections.detections.size(), 1U);
  const auto & detection = detections.detections.front();
  EXPECT_EQ(detection.header.frame_id, "camera");
  EXPECT_DOUBLE_EQ(detection.bbox.center.position.x, 100.0);
  EXPECT_DOUBLE_EQ(detection.bbox.center.position.y, 120.0);
  EXPECT_DOUBLE_EQ(detection.bbox.size_x, 40.0);
  EXPECT_DOUBLE_EQ(detection.bbox.size_y, 50.0);
  ASSERT_EQ(detection.results.size(), 1U);
  EXPECT_EQ(detection.results.front().hypothesis.class_id, "1");
  EXPECT_FLOAT_EQ(detection.results.front().hypothesis.score, 0.90F);
}

TEST(YoloV8DecoderTest, AppliesConfidenceThreshold)
{
  const auto values = MakeYoloOutput(
    {100.0F}, {120.0F}, {40.0F}, {50.0F},
    {0.10F}, {0.20F});
  const auto msg = MakeTensorBundle(MakeTensor("output_tensor", {1, 6, 1}, values));

  const auto detections = DecodeYoloV8TensorBundle(msg, Config());

  EXPECT_TRUE(detections.detections.empty());
}

TEST(YoloV8DecoderTest, AppliesNms)
{
  const auto values = MakeYoloOutput(
    {100.0F, 102.0F, 300.0F},
    {100.0F, 102.0F, 300.0F},
    {50.0F, 50.0F, 30.0F},
    {50.0F, 50.0F, 30.0F},
    {0.90F, 0.80F, 0.70F},
    {0.10F, 0.10F, 0.10F});
  const auto msg = MakeTensorBundle(MakeTensor("output_tensor", {1, 6, 3}, values));

  const auto detections = DecodeYoloV8TensorBundle(msg, Config());

  ASSERT_EQ(detections.detections.size(), 2U);
  EXPECT_DOUBLE_EQ(detections.detections.at(0).bbox.center.position.x, 100.0);
  EXPECT_DOUBLE_EQ(detections.detections.at(1).bbox.center.position.x, 300.0);
}

TEST(YoloV8DecoderTest, MissingTensorThrows)
{
  const auto values = MakeYoloOutput(
    {100.0F}, {120.0F}, {40.0F}, {50.0F},
    {0.10F}, {0.90F});
  const auto msg = MakeTensorBundle(MakeTensor("other_tensor", {1, 6, 1}, values));

  EXPECT_THROW(DecodeYoloV8TensorBundle(msg, Config()), std::runtime_error);
}

TEST(YoloV8DecoderTest, BadDtypeThrows)
{
  const auto values = MakeYoloOutput(
    {100.0F}, {120.0F}, {40.0F}, {50.0F},
    {0.10F}, {0.90F});
  const auto msg = MakeTensorBundle(MakeTensor("output_tensor", {1, 6, 1}, values, 7));

  EXPECT_THROW(DecodeYoloV8TensorBundle(msg, Config()), std::runtime_error);
}

TEST(YoloV8DecoderTest, BadShapeThrows)
{
  const auto values = MakeYoloOutput(
    {100.0F}, {120.0F}, {40.0F}, {50.0F},
    {0.10F}, {0.90F});
  const auto msg = MakeTensorBundle(MakeTensor("output_tensor", {6, 1}, values));

  EXPECT_THROW(DecodeYoloV8TensorBundle(msg, Config()), std::runtime_error);
}

TEST(YoloV8DecoderTest, BadByteSizeThrows)
{
  auto tensor = MakeTensor("output_tensor", {1, 6, 1}, {1.0F, 2.0F});
  const auto msg = MakeTensorBundle(tensor);

  EXPECT_THROW(DecodeYoloV8TensorBundle(msg, Config()), std::runtime_error);
}
