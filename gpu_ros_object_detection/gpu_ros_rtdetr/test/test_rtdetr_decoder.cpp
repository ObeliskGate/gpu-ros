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
#include <limits>
#include <stdexcept>
#include <vector>

#include "gpu_ros_rtdetr/rtdetr_decoder.hpp"

namespace
{
using gpu_ros::rtdetr::DecodeRtDetrValues;
using gpu_ros::rtdetr::RtDetrDecoderConfig;

RtDetrDecoderConfig Config()
{
  RtDetrDecoderConfig config;
  config.confidence_threshold = 0.5;
  return config;
}
} // namespace

TEST(RtDetrDecoderTest, DecodesValidBoxesAndFiltersScores)
{
  const std::vector<int64_t> labels{3, 7};
  const std::vector<float> boxes{10.0F, 20.0F, 50.0F, 80.0F, 0.0F, 0.0F, 1.0F, 1.0F};
  const std::vector<float> scores{0.9F, 0.5F};
  std_msgs::msg::Header header;
  header.frame_id = "camera";

  const auto output = DecodeRtDetrValues(header, labels.data(), labels.size(), boxes.data(),
    boxes.size(), scores.data(), scores.size(), Config());

  ASSERT_EQ(output.detections.size(), 1U);
  const auto & detection = output.detections.front();
  EXPECT_EQ(detection.results.front().hypothesis.class_id, "3");
  EXPECT_FLOAT_EQ(detection.results.front().hypothesis.score, 0.9F);
  EXPECT_DOUBLE_EQ(detection.bbox.center.position.x, 30.0);
  EXPECT_DOUBLE_EQ(detection.bbox.center.position.y, 50.0);
  EXPECT_DOUBLE_EQ(detection.bbox.size_x, 40.0);
  EXPECT_DOUBLE_EQ(detection.bbox.size_y, 60.0);
}

TEST(RtDetrDecoderTest, RejectsInconsistentSpansAndNonFiniteBoxes)
{
  const std::vector<int64_t> labels{1};
  const std::vector<float> boxes{0.0F, 0.0F, 1.0F, 1.0F};
  const std::vector<float> scores{0.9F};
  EXPECT_THROW(DecodeRtDetrValues(std_msgs::msg::Header{}, labels.data(), labels.size(),
                 boxes.data(), 3U, scores.data(), scores.size(), Config()),
    std::invalid_argument);

  const std::vector<float> nonfinite_boxes{
    0.0F, 0.0F, std::numeric_limits<float>::quiet_NaN(), 1.0F};
  const auto output = DecodeRtDetrValues(std_msgs::msg::Header{}, labels.data(), labels.size(),
    nonfinite_boxes.data(), nonfinite_boxes.size(), scores.data(), scores.size(), Config());
  EXPECT_TRUE(output.detections.empty());
}
